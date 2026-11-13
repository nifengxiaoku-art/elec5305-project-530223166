"""Independent artifact audit: content duplicates, cache integrity and LR refits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset import CATEGORIES, load_dataset, sha256
from .evaluate import metrics
from .extract_embeddings import FrozenCLAP, normalized_matrix, write_json
from .few_shot import fit_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--data-root",default="data/ESC-50")
    parser.add_argument("--cache",type=Path,default=Path("cache/embeddings"))
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    out=args.run/"verification.json"
    if out.exists():
        raise FileExistsError("Verification output already exists; preserve previous audit")
    config=json.loads((args.run/"config.json").read_text())
    full,rows=load_dataset(args.data_root)
    model=json.loads((args.run/"model.json").read_text())
    cache_dirs={}
    for directory in args.cache.iterdir():
        manifest=json.loads((directory/"manifest.json").read_text())
        if all(manifest.get(k)==v for k,v in model.items()):
            cache_dirs["clean" if manifest["condition"].get("clean") else "noise"]=directory
    vectors={}
    checked=0
    content={}
    for method,index_file in [("clean","clean_cache.csv"),("noise","noisy_cache.csv")]:
        index=pd.read_csv(args.run/"manifests"/index_file,dtype={"src_file":str})
        vectors[method]={}
        for r in index.itertuples():
            row=rows[rows.filename==r.filename].iloc[0]
            if sha256(row.path)!=r.audio_sha256 or row.category!=r.category or int(row.fold)!=r.fold:
                raise ValueError("Cache input identity mismatch")
            vector=np.load(cache_dirs[method]/(r.cache_key+".npy"),allow_pickle=False)
            if vector.shape!=(1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:
                raise ValueError("Invalid cache vector")
            vectors[method][row.name]=vector
            checked+=1
            if method=="clean":
                content.setdefault(r.audio_sha256,[]).append(r.filename)
    duplicate_audio=[v for v in content.values() if len(v)>1]
    # Exact duplicate audio across different source IDs would defeat source checks.
    cross_source_duplicates=[]
    for names in duplicate_audio:
        if rows[rows.filename.isin(names)].src_file.nunique()>1:
            cross_source_duplicates.append(names)
    if cross_source_duplicates:
        raise ValueError(f"Cross-source exact duplicate audio: {cross_source_duplicates}")
    records=pd.read_csv(args.run/"metrics"/"runs.csv")
    selections=pd.read_csv(args.run/"manifests"/"selected_training.csv")
    predicted=pd.read_csv(args.run/"predictions"/"few_shot.csv")
    refits=0
    for record in records.itertuples():
        chosen=selections[selections.run_id==record.run_id]
        train=rows.set_index("filename",drop=False).loc[chosen.filename].copy()
        train.index=rows.set_index("filename").loc[chosen.filename].index.map(dict(zip(rows.filename,rows.index)))
        val_names=pd.read_csv(args.run/"manifests"/f"fold{record.test_fold}_validation.csv").filename
        val=rows[rows.filename.isin(val_names)]
        test=rows[rows.fold==record.test_fold]
        x=normalized_matrix(vectors["clean"],train.index)
        y=train.category.to_numpy()
        weights=None
        if record.method=="noise":
            x=np.concatenate([x,normalized_matrix(vectors["noise"],train.index)])
            y=np.tile(y,2)
            weights=np.full(len(y),.5)
        classifier,c,search=fit_classifier(x,y,normalized_matrix(vectors["clean"],val.index),
                                         val.category.to_numpy(),config["C_grid"],weights)
        if c!=record.C:
            raise ValueError("Validation-selected C not reproducible")
        labels=classifier.predict(normalized_matrix(vectors["clean"],test.index))
        stored=predicted[predicted.run_id==record.run_id].set_index("filename").loc[test.filename].predicted_label.to_numpy()
        np.testing.assert_array_equal(labels,stored)
        refits+=1
    extractor=FrozenCLAP(config,model["checkpoint_path"],args.offline)
    if extractor.manifest["checkpoint_sha256"]!=model["checkpoint_sha256"]:
        raise ValueError("Checkpoint fingerprint changed")
    import torch
    state=torch.load(extractor.model.model_fp,map_location="cpu")["model"]
    # Legacy GPT-2 checkpoints serialize deterministic attention masks; current
    # Transformers keeps causal masks nonpersistent and removes masked_bias.
    # Validate those exact constants rather than accepting arbitrary extra keys.
    expected_keys=set(extractor.model.clap.state_dict())
    ignored=[]
    for key in sorted(set(state)-expected_keys):
        prefix="caption_encoder.base.h."
        tail=key[len(prefix):].split(".") if key.startswith(prefix) else []
        if len(tail)!=3 or not tail[0].isdigit() or int(tail[0]) not in range(12) or tail[1]!="attn":
            raise ValueError(f"Unexpected checkpoint parameter: {key}")
        value=state[key]
        if tail[2]=="bias":
            expected=torch.tril(torch.ones((1024,1024),dtype=value.dtype)).reshape(1,1,1024,1024)
            if not torch.equal(value,expected):
                raise ValueError("Legacy GPT-2 causal mask differs from expected constant")
        elif tail[2]=="masked_bias":
            if value.numel()!=1 or value.item()!=-10000:
                raise ValueError("Legacy GPT-2 masked_bias differs from expected constant")
        else:
            raise ValueError(f"Unrecognized extra checkpoint key: {key}")
        ignored.append(key)
        del state[key]
    extractor.model.clap.load_state_dict(state,strict=True)
    del state
    examples=rows.iloc[[0,399]]
    public=extractor.model.get_audio_embeddings(examples.path.tolist()).detach().cpu().numpy()
    cached=np.stack([vectors["clean"][i] for i in examples.index])
    np.testing.assert_allclose(public,cached,rtol=1e-4,atol=1e-4)
    text=extractor.text_embeddings()
    np.testing.assert_allclose(text,np.load(args.run/"manifests"/"text_embeddings.npy"),rtol=1e-5,atol=1e-5)
    report={"status":"PASS", "cache_vectors_checked":checked,
            "exact_audio_duplicate_groups":duplicate_audio,"cross_source_exact_duplicates":cross_source_duplicates,
            "all_validation_searches_and_test_predictions_refit":refits,
            "strict_checkpoint_keys":"all learned parameters matched; legacy constant masks explicitly validated",
            "legacy_gpt2_constant_buffers":ignored,
            "public_api_audio_max_abs_difference":float(np.max(np.abs(public-cached))),
            "public_api_examples":examples.filename.tolist(),"text_embeddings_reproduced":True,
            "limitation":"Does not establish absence of near duplicates or upstream pretraining overlap"}
    write_json(out,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
