# Probing Musical Emotion in Pretrained Music Representations

This project investigates whether continuous musical emotion information can be decoded from pretrained MERT representations using the DEAM dataset.

## Research Questions

1. Can continuous Valence and Arousal be decoded from frozen MERT representations?
2. How does emotion decodability vary across MERT Transformer layers?
3. Do MERT representations provide emotion-related information beyond traditional low-level audio features?
4. To what extent might prediction performance be explained by confounding factors such as tempo and energy?

## Method Overview

DEAM Audio
→ Audio Preprocessing
→ Frozen MERT-v1-95M
→ Layer-wise Hidden Representations
→ Simple Regression Probes
→ Valence / Arousal Prediction

MERT remains frozen in the main experiments.

## Planned Experiments

The following experiments are planned and have not yet been completed:

- Data and representation setup
- Basic Valence/Arousal probing
- Layer-wise probing
- Traditional audio-feature baseline
- Confound and error analysis
- Optional robustness/ablation experiments

## Evaluation

- MAE
- R²
- Pearson correlation (r)

Valence and Arousal will be evaluated separately.

## Repository Structure

- `configs/`: Experiment configuration files
- `notebooks/`: Exploratory and analysis notebooks
- `src/`: Reusable project source code
- `scripts/`: Command-line workflow scripts
- `data/`: Raw, processed, and metadata files
- `outputs/`: Embeddings, results, and figures
- `docs/`: Project documentation

Raw DEAM audio, large processed data, model weights, and cached MERT embeddings are intentionally excluded from version control.

## Environment

This project uses Python 3.10 with CUDA-enabled PyTorch and has been tested locally on an NVIDIA RTX 4060 Laptop GPU. The environment specification is provided in `environment.yml`.

## Status

Work in progress.

Current stage:
Repository and reproducibility setup.

## Scope

The first version of this project includes:

- No MERT fine-tuning
- No foundation-model training
- No music generation
- No large-scale human study
