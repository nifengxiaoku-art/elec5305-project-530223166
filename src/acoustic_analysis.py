"""Actual prediction-linked acoustic examples and a training-only noise illustration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .augmentation import add_noise, noise_seed
from .analyze_study import figure_save


def time_frequency(signal, rate):
    f, t, z = stft(signal, fs=rate, window="hann", nperseg=1024,
                   noverlap=1024-320, boundary=None, padded=False)
    keep = f <= 8000
    return f[keep], t, np.abs(z[keep])**2


def descriptors(signal, rate):
    f, t, power = time_frequency(signal, rate)
    spectrum = power.sum(1)
    distribution = spectrum / spectrum.sum()
    frame_power = power.sum(0)
    return {"rms":float(np.sqrt(np.mean(signal.astype(float)**2))),
            "centroid_0_8khz_hz":float(np.sum(f*distribution)),
            "rolloff90_0_8khz_hz":float(f[np.searchsorted(np.cumsum(distribution),.9)]),
            "flatness_0_8khz":float(np.exp(np.mean(np.log(spectrum+1e-20)))/np.mean(spectrum)),
            "active_frame_fraction_minus20db":float(np.mean(frame_power > frame_power.max()*.01))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/ESC-50"))
    args = parser.parse_args()
    out = args.run/"acoustics"
    out.mkdir(exist_ok=False)
    pred = pd.read_csv(args.run/"predictions"/"zero_shot.csv")
    fold = pred[pred.fold == 5].sort_values("filename")
    chosen = []
    for category, correct in [("hen",False),("hen",True),("rooster",True)]:
        candidates = fold[(fold.true_label == category) & ((fold.true_label == fold.predicted_label) == correct)]
        if len(candidates):
            chosen.append(candidates.iloc[0])
    if not chosen:
        raise ValueError("No matching observed acoustic cases")
    fig, axes = plt.subplots(len(chosen),2,figsize=(11,2.5*len(chosen)),squeeze=False,constrained_layout=True)
    records = []
    for i,row in enumerate(chosen):
        signal, rate = sf.read(args.data_root/"audio"/row.filename, dtype="float32")
        f,t,power = time_frequency(signal,rate)
        db = 10*np.log10(np.maximum(power,1e-20)/power.max())
        axes[i,0].plot(np.arange(len(signal))/rate,signal,lw=.35)
        axes[i,0].set(xlabel="Time (s)",ylabel="Amplitude",xlim=(0,5),
                      title=f"{row.filename}: true={row.true_label}; zero-shot={row.predicted_label}")
        image=axes[i,1].pcolormesh(t,f/1000,db,vmin=-70,vmax=0,cmap="magma",shading="auto")
        axes[i,1].set(xlabel="Time (s)",ylabel="Frequency (kHz)",ylim=(0,8),
                      title="STFT power: dB relative to this clean clip's maximum")
        fig.colorbar(image,ax=axes[i,1],label="dB")
        records.append({"role":"prediction_example", "filename":row.filename,
                        "true_label":row.true_label,"predicted_label":row.predicted_label,
                        **descriptors(signal,rate)})
    figure_save(fig,out,"prediction_examples")
    selection = pd.read_csv(args.run/"manifests"/"selected_training.csv")
    train = selection[(selection.run_id == "clean_f5_n4_s17") & (selection.category == "hen")].sort_values("filename").iloc[0]
    signal,rate = sf.read(args.data_root/"audio"/train.filename,dtype="float32")
    config = json.loads((args.run/"config.json").read_text())
    noisy,details=add_noise(signal,config["snr_db"],noise_seed(train.filename,config["noise_seed"]))
    ref=time_frequency(signal,rate)[2].max()
    limit=max(np.max(np.abs(signal)),np.max(np.abs(noisy)))*1.05
    fig,axes=plt.subplots(2,2,figsize=(10,5.2),constrained_layout=True)
    for i,(name,wave) in enumerate([("Clean training original",signal),("Training view: 20 dB Gaussian noise",noisy)]):
        f,t,power=time_frequency(wave,rate)
        axes[i,0].plot(np.arange(len(wave))/rate,wave,lw=.35)
        axes[i,0].set(xlim=(0,5),ylim=(-limit,limit),xlabel="Time (s)",ylabel="Amplitude",title=name)
        image=axes[i,1].pcolormesh(t,f/1000,10*np.log10(np.maximum(power,1e-20)/ref),
                                 vmin=-70,vmax=0,cmap="magma",shading="auto")
        axes[i,1].set(ylim=(0,8),xlabel="Time (s)",ylabel="Frequency (kHz)",title=train.filename)
        fig.colorbar(image,ax=axes[i,1],label="dB relative to clean maximum")
        records.append({"role":name,"filename":train.filename,"true_label":train.category,
                        "predicted_label":"not evaluated here",**descriptors(wave,rate),
                        **(details if i else {})})
    figure_save(fig,out,"training_clean_noise")
    pd.DataFrame(records).to_csv(out/"descriptors.csv",index=False)
    (out/"selection_rule.txt").write_text(
        "Exploratory post-evaluation illustrations: lexicographically first Fold-5 hen error, correct hen and correct rooster.\n"
        "Noise illustration: first selected hen training original for clean_f5_n4_s17.\n"
        "No listening-based or causal claim is established by these descriptors.\n"
        "Hann STFT 1024 samples, hop 320 (23.22 ms / 7.26 ms), 0–8 kHz.\n"
        "Noise comparison shares the clean power reference and waveform amplitude limits.\n",encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
