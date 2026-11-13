"""Deterministic waveform Gaussian noise at a measured global SNR."""
import hashlib
import numpy as np


def noise_seed(filename, base_seed):
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{filename}".encode()).digest()[:8], "little")


def add_noise(waveform, snr_db, seed):
    clean = np.asarray(waveform, dtype=np.float64)
    power = np.mean(clean ** 2)
    if not np.isfinite(clean).all() or power <= 1e-20:
        raise ValueError("Nonfinite or silent waveform cannot define SNR")
    noise = np.random.default_rng(seed).standard_normal(clean.shape)
    noise -= noise.mean()
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    noisy = (clean + noise).astype(np.float32)
    actual = 10 * np.log10(power / np.mean((noisy.astype(np.float64) - clean) ** 2))
    return noisy, {"requested_snr_db": float(snr_db), "measured_snr_db": float(actual),
                   "peak_abs": float(np.max(np.abs(noisy))), "clipped": False}
