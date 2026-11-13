"""Exploratory fixed-C=1 diagnostic; no validation selection, unchanged primary study."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset import load_dataset
from .extract_embeddings import normalized_matrix, write_json
from .few_shot import fit_classifier
from .evaluate import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--data-root",default="data/ESC-50")
    parser.add_argument("--cache",type=Path,default=Path("cache/embeddings"))
    args=parser.parse_args()
    out=args.run/"sensitivity_fixed_c1"
    out.mkdir(exist_ok=False)
    _,rows=load_dataset(args.data_root)
    model=json.loads((args.run/"model.json").read_text())
    directories={}
    for directory in args.cache.iterdir():
        manifest=json.loads((directory/"manifest.json").read_text())
        if all(manifest.get(k)==v for k,v in model.items()):
            directories["clean" if manifest["condition"].get("clean") else "noise"]=directory
    vectors={}
    for method,index_file in [("clean","clean_cache.csv"),("noise","noisy_cache.csv")]:
        frame=pd.read_csv(args.run/"manifests"/index_file)
        lookup=rows.reset_index().set_index("filename")["index"].to_dict()
        vectors[method]={lookup[r.filename]:np.load(directories[method]/(r.cache_key+".npy"),allow_pickle=False)
                         for r in frame.itertuples()}
    runs=pd.read_csv(args.run/"metrics"/"runs.csv")
    selected=pd.read_csv(args.run/"manifests"/"selected_training.csv")
    records,predictions=[],[]
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    for run in runs[~runs.pilot].itertuples():
        selection=selected[selected.run_id==run.run_id]
        train=rows.set_index("filename",drop=False).loc[selection.filename]
        indices=[lookup[f] for f in train.filename]
        x=normalized_matrix(vectors["clean"],indices)
        y=train.category.to_numpy()
        weights=None
        if run.method=="noise":
            x=np.concatenate([x,normalized_matrix(vectors["noise"],indices)])
            y=np.tile(y,2)
            weights=np.full(len(y),.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            classifier=LogisticRegression(C=1,solver="lbfgs",max_iter=3000,tol=1e-6).fit(x,y,sample_weight=weights)
        test=rows[rows.fold==run.test_fold]
        predicted=classifier.predict(normalized_matrix(vectors["clean"],test.index))
        result=metrics(test.category,predicted)
        identity={"run_id":run.run_id,"test_fold":run.test_fold,"n":run.n,"seed":run.seed,"method":run.method}
        records.append({**identity,"C":1,"validation_labels_used":0,
                        "macro_f1":result["macro_f1"],"accuracy":result["accuracy"]})
        for filename,true,label in zip(test.filename,test.category,predicted):
            predictions.append({**identity,"filename":filename,"true_label":true,"predicted_label":label})
    data=pd.DataFrame(records)
    data.to_csv(out/"runs.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"predictions.csv",index=False)
    folds=data.groupby(["method","n","test_fold"],as_index=False)[["macro_f1","accuracy"]].mean()
    summary=folds.groupby(["method","n"]).agg(macro_f1_mean=("macro_f1","mean"),fold_sd=("macro_f1","std"),folds=("test_fold","nunique"))
    summary.to_csv(out/"summary.csv")
    pairs=data[data.method=="noise"].merge(data[data.method=="clean"],on=["test_fold","n","seed"],suffixes=("_noise","_clean"),validate="one_to_one")
    pairs["delta_macro_f1"]=pairs.macro_f1_noise-pairs.macro_f1_clean
    pairs.to_csv(out/"paired_deltas.csv",index=False)
    pairs.groupby(["n","test_fold"],as_index=False).delta_macro_f1.mean().to_csv(out/"paired_fold_means.csv",index=False)
    for run_id,g in pd.DataFrame(predictions).groupby("run_id"):
        result=metrics(g.true_label,g.predicted_label)
        np.testing.assert_allclose(result["macro_f1"],data[data.run_id==run_id].iloc[0].macro_f1,atol=1e-12)
    write_json(out/"audit.json",{"status":"PASS","prediction_runs_recomputed":len(data),"C":1,
               "validation_labels_used":0,"status_of_design":"exploratory, specified after primary scores; not used to change primary method"})
    print(summary.to_string())


if __name__=="__main__":
    main()
