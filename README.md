# ELEC5305 Project - Animal Sound Classification Using a Pretrained Audio-Language Model

This project studies animal sound category classification using the ESC-50 dataset and Microsoft CLAP (Contrastive Language-Audio Pretraining).

## Research question
How much labelled animal audio is required to adapt a pretrained audio-language model to animal sound classification, and does noise augmentation provide additional benefit when labelled data are scarce?

## Dataset
The ESC-50 dataset is used, with the following 10 animal sound categories:

- dog
- rooster
- pig
- cow
- frog
- cat
- hen
- insects
- sheep
- crow

The official ESC-50 fold structure is preserved. No new random train/test split is created.

## Model
The main model is Microsoft CLAP. The model remains frozen and is used for:

- text embeddings
- audio embeddings
- zero-shot classification
- feature extraction for few-shot classification

## Experimental pipeline
### 1. CLAP validation
The CLAP package is tested to confirm it loads successfully and can generate text embeddings from fixed category prompts.

### 2. ESC-50 preparation
The ESC-50 metadata is loaded and filtered to the 10 animal classes. The class counts are checked to ensure each class contains 40 recordings, and the official fold assignments are preserved.

### 3. Zero-shot classification
A held-out official ESC-50 fold is used as the test fold. Audio embeddings are extracted with CLAP and compared to the ten text embeddings for the chosen animal categories.

### 4. Few-shot classification
A small linear classifier is trained on CLAP audio embeddings using a small number of labelled examples per class. The investigation includes n = 1, 2, 4, 8, 16, 24.

### 5. Noise augmentation
Noise augmentation is applied to the waveform before CLAP feature extraction. The goal is to test whether augmentation still helps when the amount of labelled data is limited.

## Evaluation metrics
- Accuracy
- Macro-F1
- Per-class recall
- Confusion matrix

## Preliminary results
The zero-shot experiment on ESC-50 fold 5 achieved:

- Accuracy: 0.9125
- Macro-F1: 0.9029

The main confusion was observed for the hen class, while most other animal categories were correctly recognised.

## Repository contents
- src/test_clap.py: verify CLAP loads and generates fixed text embeddings
- src/load_esc50.py: load and filter ESC-50 metadata for the 10 animal categories
- src/zero_shot.py: zero-shot CLAP classification on an official ESC-50 fold
- results/: saved metric and confusion-matrix outputs

## Notes
- Raw ESC-50 audio files and the Python virtual environment are not uploaded to GitHub.
- The project code and result files are intended for version control and reproducibility.
