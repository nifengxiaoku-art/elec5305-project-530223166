"""Frozen CLAP inference with content-addressed, resumable per-clip caching."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import random
import types
import numpy as np
import soundfile as sf
from .dataset import sha256
from .augmentation import add_noise, noise_seed


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


class FrozenCLAP:
    def __init__(self, config, checkpoint=None, offline=False):
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from msclap import CLAP
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(20261112)
        np.random.seed(20261112)
        random.seed(20261112)
        torch.use_deterministic_algorithms(True)
        self.model = CLAP(model_fp=checkpoint, version=config["version"], use_cuda=False)
        self.model.clap.requires_grad_(False)
        self.model.clap.eval()
        self.config = config
        # Override the instance loader only: avoid torchaudio/torchcodec decoder
        # incompatibilities without monkey-patching third-party global functions.
        def read_audio(instance, audio_path, resample=True):
            waveform, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            tensor = torch.from_numpy(waveform.T.copy())
            if rate != instance.args.sampling_rate:
                if not resample:
                    raise ValueError("Sample-rate mismatch with resampling disabled")
                import torchaudio
                tensor = torchaudio.functional.resample(tensor, rate, instance.args.sampling_rate)
            return tensor, instance.args.sampling_rate
        self.model.read_audio = types.MethodType(read_audio, self.model)
        import msclap
        package_root = Path(msclap.__file__).parent
        implementation = {str(p.relative_to(package_root)): sha256(p)
                          for p in sorted(package_root.rglob("*.py"))}
        self.manifest = {
            "version": config["version"], "checkpoint_sha256": sha256(self.model.model_fp),
            "checkpoint_path": str(self.model.model_fp),
            "checkpoint_snapshot": Path(self.model.model_fp).parent.name,
            "clap_config": self.model.config_as_str,
            "msclap_source_sha256": fingerprint(implementation),
            "extractor_source_sha256": sha256(__file__),
            "augmentation_source_sha256": sha256(Path(__file__).with_name("augmentation.py")),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["msclap", "torch", "torchaudio", "numpy", "soundfile", "transformers"]},
            "preprocessing": "soundfile float32 mono; native 44100 Hz; msclap repeat 5s to 7s; no peak normalization",
            "device": "cpu", "torch_threads": config["torch_threads"],
            "deterministic_algorithms": True,
        }
        print("CLAP loaded:", self.manifest["checkpoint_sha256"], flush=True)

    def text_embeddings(self):
        return self.model.get_text_embeddings(self.config["prompts"]).detach().cpu().numpy()

    def extract(self, rows, cache_root, noisy=False):
        import torch
        condition = {"snr_db": self.config["snr_db"], "noise_seed": self.config["noise_seed"]} if noisy else {"clean": True}
        identity = {**self.manifest, "condition": condition}
        cache_dir = Path(cache_root) / fingerprint(identity)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            write_json(manifest_path, identity)
        embeddings = {}
        metadata = []
        pending = []
        for idx, row in rows.iterrows():
            item = {"filename": row.filename, "category": row.category, "fold": int(row.fold),
                    "src_file": row.src_file, "audio_sha256": sha256(row.path)}
            item["cache_key"] = fingerprint(item)
            path = cache_dir / (item["cache_key"] + ".npy")
            if path.exists():
                vector = np.load(path, allow_pickle=False)
                if vector.shape != (1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError(f"Invalid cached embedding: {path}")
                embeddings[idx] = vector
            else:
                pending.append((idx, row, item, path))
            metadata.append(item)
        print(f"{'noisy' if noisy else 'clean'} cache: {len(embeddings)} reused, {len(pending)} pending", flush=True)
        batch_size = self.config["batch_size"]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            signals = []
            for idx, row, item, path in batch:
                signal, rate = sf.read(row.path, dtype="float32")
                if rate != self.model.args.sampling_rate or signal.ndim != 1:
                    raise ValueError("Expected validated 44.1 kHz mono audio")
                if noisy:
                    seed = noise_seed(row.filename, self.config["noise_seed"])
                    signal, details = add_noise(signal, self.config["snr_db"], seed)
                    details["seed"] = seed
                    write_json(path.with_suffix(".json"), details)
                length = self.model.args.duration * rate
                signal = np.tile(signal, int(np.ceil(length / len(signal))))[:length]
                signals.append(signal)
            with torch.inference_mode():
                tensor = torch.from_numpy(np.stack(signals)).unsqueeze(1)
                vectors = self.model._get_audio_embeddings(tensor).detach().cpu().numpy()
            for (idx, row, item, path), vector in zip(batch, vectors):
                if vector.shape != (1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError("CLAP returned invalid embedding")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, vector, allow_pickle=False)
                temporary.replace(path)
                embeddings[idx] = vector
            print(f"{'noisy' if noisy else 'clean'} encoded {min(start+batch_size, len(pending))}/{len(pending)}", flush=True)
        import pandas as pd
        return embeddings, pd.DataFrame(metadata), cache_dir


def normalized_matrix(embeddings, indices):
    matrix = np.stack([embeddings[i] for i in indices])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
