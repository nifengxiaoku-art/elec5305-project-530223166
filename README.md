# ELEC5305: Animal Sound Classification with Frozen CLAP

Code for zero-shot classification, few-shot label efficiency and matched
Gaussian-noise augmentation on ten ESC-50 animal categories. 

## Setup

Tested with Python 3.12.7 on CPU. From the repository root, in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git data/ESC-50
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Use an existing dataset/environment if present. On macOS/Linux, create a Python
3.12 environment and use `.venv/bin/python` instead. The data loader expects
`data/ESC-50/meta/esc50.csv` and `data/ESC-50/audio/*.wav`.

## Run the complete experiments

```powershell
.venv/Scripts/python.exe -u -m src.reproduce --output runs/experiment_01
```

The command executes zero-shot evaluation, the n=4 pilot, the full label curve,
matched waveform-noise experiments, metric/figure generation, acoustic analysis,
SNR diagnostics, fixed-C sensitivity, error-transition analysis and verification.
Outputs are written under the new directory supplied to `--output`.

The first run downloads model/text assets and extracts frozen features. Later
runs reuse compatible cached features. To run after assets are cached:

```powershell
.venv/Scripts/python.exe -u -m src.reproduce --output runs/experiment_02 --offline
```

Every output directory must be new. Optional arguments are `--data-root PATH`,
`--cache PATH`, `--config PATH` and `--checkpoint PATH`. The CLAP checkpoint
revision and SHA-256 are fixed in the configuration; a mismatched checkpoint
fails before feature extraction. A GPU is not required.

## Short working-code demonstration

```powershell
.venv/Scripts/python.exe -u -m src.demo --output runs/demo_01 --offline
```

Omit `--offline` for first-time model setup. This runs the clean classification
study, generates its results and prints the actual validation scores for C,
the selected C and the corresponding held-out test score. `C` is inverse
regularization strength; highest validation Macro-F1 wins, with smaller-C ties.

## Configuration and generated outputs

`configs/study.json` fixes category order, prompts, model identity, training
budgets, seeds and 20 dB Gaussian-noise strength. Official test folds are
preserved and recording-source conflicts are purged from train/validation.
Three seeds are averaged within each fold before the five-fold mean/sample SD.
n=24 is feasible for only two folds after source purging and is flagged separately.
Primary regularization selection uses additional validation labels.

| Generated location | Contents |
|---|---|
| `metrics/` | Per-run scores, validation C search and dataset checks |
| `predictions/` | Filename, true/predicted category and fold/seed identity |
| `manifests/` | Split membership, training selections and feature identity |
| `analysis/` | Learning curve, aggregate metrics, confusion and paired deltas |
| `acoustics/` | Actual waveform/STFT examples and clean/noisy comparison |
| `noise_diagnostics/` | Five-/seven-second and frequency-band SNR |
| `sensitivity_fixed_c1/`, `interpretation/` | Fixed-C and error-transition analyses |
| `verification.json`, `reproduction.json` | Refit/cache/model checks and completed steps |

Per-recording prediction CSVs and experiment logs are outputs, not files needed
from a previous execution. No data, model weights, cached features or generated
results are versioned. Pinned dependencies/model identity and fixed seeds support
reproduction; floating-point differences may occur on another platform.

## Data and model attribution

ESC-50: Karol J. Piczak, *ESC: Dataset for Environmental Sound Classification*
(2015), [official data/license](https://github.com/karolpiczak/ESC-50).
ESC-50 is CC BY-NC 3.0; this animal subset is not the separately licensed ESC-10.
Keep original clip attributions. The pretrained model is
[Microsoft CLAP](https://github.com/microsoft/CLAP). Neither audio nor weights are
redistributed here. AI tools assisted code development; follow the course's
required acknowledgement and describe actual student contributions accurately.
