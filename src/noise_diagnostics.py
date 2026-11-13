"""Audit the declared SNR after CLAP repetition and within its analysis band."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from .augmentation import add_noise, noise_seed
from .extract_embeddings import write_json


def snr(clean, error):
    return float(10*np.log10(np.mean(clean.astype(float)**2)/np.mean(error.astype(float)**2)))


def ideal_band_snr(clean, error, rate):
    frequencies=np.fft.rfftfreq(len(clean),1/rate)
    band=(frequencies>=50)&(frequencies<=8000)
    a=np.fft.rfft(clean)[band]
    b=np.fft.rfft(error)[band]
    return float(10*np.log10(np.sum(np.abs(a)**2)/np.sum(np.abs(b)**2)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--data-root",type=Path,default=Path("data/ESC-50"))
    args=parser.parse_args()
    out=args.run/"noise_diagnostics"
    out.mkdir(exist_ok=False)
    config=json.loads((args.run/"config.json").read_text())
    index=pd.read_csv(args.run/"manifests"/"noisy_cache.csv")
    records=[]
    for r in index.itertuples():
        clean,rate=sf.read(args.data_root/"audio"/r.filename,dtype="float32")
        noisy,details=add_noise(clean,config["snr_db"],noise_seed(r.filename,config["noise_seed"]))
        clean7=np.tile(clean,2)[:7*rate]
        noisy7=np.tile(noisy,2)[:7*rate]
        error5=noisy.astype(float)-clean.astype(float)
        error7=noisy7.astype(float)-clean7.astype(float)
        records.append({"filename":r.filename,"category":r.category,
                        "global_snr_5s_db":snr(clean,error5),"global_snr_7s_db":snr(clean7,error7),
                        "ideal_fft_band_50_8000_snr_7s_db":ideal_band_snr(clean7,error7,rate),
                        "peak_exceeds_one":details["peak_abs"]>1,"peak_abs":details["peak_abs"]})
    data=pd.DataFrame(records)
    data.to_csv(out/"snr_input.csv",index=False)
    summary={k:{"mean":float(data[k].mean()),"min":float(data[k].min()),"max":float(data[k].max())}
             for k in ["global_snr_5s_db","global_snr_7s_db","ideal_fft_band_50_8000_snr_7s_db"]}
    summary["views_with_float_peak_above_one"]=int(data.peak_exceeds_one.sum())
    summary["waveforms_checked"]=len(data)
    summary["interpretation"]="20 dB refers to original 5 s global waveform. Repetition changes relative power; ideal FFT band SNR is diagnostic, not exact Mel/CLAP effective SNR. No new inference or parameter selection."
    write_json(out/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
