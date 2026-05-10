# Text Classification with Transformer Models (BERT & DistilBERT)

## Overview

Fine-tuning pre-trained transformer models for sentiment classification on the Yelp Reviews dataset using Hugging Face Transformers and the Trainer API.

## Technologies Used

- **Python** — core language
- **Hugging Face Transformers** — BERT, DistilBERT models and Trainer API
- **Hugging Face Datasets** — Yelp Review Full dataset
- **PyTorch** — model training backend
- **evaluate** — accuracy metrics
- **Google Colab (GPU T4)** — training environment

## Setup

### Install dependencies

```bash
pip install transformers==4.44.0 datasets evaluate accelerate
```

### Run in Google Colab

1. Open the notebook in Google Colab
2. Set runtime to **GPU (T4 recommended)**
3. Run all cells in order

## Project Structure

The notebook includes the following experiments:

### Experiment 1 — Baseline BERT
- `bert-base-cased` with default hyperparameters
- 2000 training samples, 500 test samples

### Experiment 2 — DistilBERT Comparison
- `distilbert-base-cased` for speed vs accuracy tradeoff
- ~2× faster than BERT with comparable accuracy

### Experiment 3 — Hyperparameter Tuning
- Varying epochs, batch size, and dataset size
- Analysing the effect on model accuracy

## Models Compared

| Model | Training Data | Epochs | Training Time | Accuracy |
|---|---|---|---|---|
| BERT (bert-base-cased) | 2000 samples | 3 | ~10.7 min | 59.2% |
| DistilBERT (distilbert-base-cased) | 2000 samples | 3 | ~5.4 min | — |

## Dataset

- **Yelp Review Full** — 650,000 reviews with 5-star ratings (1–5)
- Subset used: 2,000 training samples, 500 test samples (for faster experimentation)

## Key Concepts Demonstrated

- **Tokenization** — how BERT converts text to `input_ids` and `attention_mask`
- **Padding & Truncation** — handling variable-length sequences
- **Fine-tuning** — adapting pre-trained weights to a downstream classification task
- **Transfer Learning** — leveraging models pre-trained on large corpora
- **Hyperparameter experimentation** — epochs, batch size, learning rate, dataset size
