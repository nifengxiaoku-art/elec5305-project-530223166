# elec5305-project-530223166

ELEC5305 Project - Animal Sound Classification Using a Pretrained Audio-Language Model

## Project Overview

This project investigates animal sound category classification using the ESC-50 dataset.

Instead of training a CNN from scratch, this project uses a pretrained audio-language model, Microsoft CLAP.

The main focus is to investigate how much labelled animal audio is needed to improve classification performance.

Noise augmentation will also be tested to determine whether it is useful when only a small amount of labelled training data is available.

## Research Question

How much labelled animal audio is required to adapt a pretrained audio-language model to animal sound classification, and does noise augmentation provide additional benefit when labelled data are scarce?

## Dataset

The ESC-50 dataset will be used.

This project focuses on the following ten animal sound categories:

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

The official ESC-50 folds will be preserved for training, validation, and testing.

## Proposed Method

The main model is Microsoft CLAP (Contrastive Language-Audio Pretraining).

The project contains three main stages.

### 1. Zero-shot Classification

ESC-50 audio → Pretrained CLAP → Text class prompts → Predicted animal class

No task-specific training is used in this stage.

The zero-shot result will be used as the baseline.

### 2. Few-shot Classification

Audio → Pretrained CLAP → Audio Embedding → Small Classifier → Animal Class

The CLAP model will be frozen.

A simple classifier, such as logistic regression or a linear classifier, will be trained using CLAP audio embeddings.

Different numbers of labelled examples per class will be tested:

n = 1, 2, 4, 8, 16, 24

The aim is to measure how classification performance changes as more labelled data are added.

### 3. Noise Augmentation

Noise augmentation will be applied to the audio waveform before extracting CLAP embeddings.

Two representative low-data conditions will be tested:

- 4 labelled examples per class
- 16 labelled examples per class

The results with and without noise augmentation will be compared.

## Evaluation Metrics

The following metrics will be used:

- Accuracy
- Macro-F1 score
- Per-class recall
- Confusion matrix

The main final result will be a learning curve showing:

Macro-F1 vs. Number of labelled examples per class

## Current Work

The current milestones are:

- Load the ten ESC-50 animal classes
- Preserve the official ESC-50 folds
- Set up Microsoft CLAP
- Run zero-shot classification
- Calculate zero-shot accuracy and Macro-F1
- Generate a confusion matrix
- Extract and save CLAP audio embeddings
- Train a few-shot classifier using 4 labelled examples per class
- Compare zero-shot and few-shot performance

## Optional Baseline

A CNN trained on Mel-spectrograms may be retained as a conventional baseline for comparison.

However, the main focus of this project is the pretrained CLAP representation and labelled-data efficiency.
