# SA-DNN-LFG-Intrusion-Detection

## Overview
This is an implementation and extension of the SA-DNN+LFG intrusion detection model with SHAP/LIME
explainability features in IoT environments presented in the paper:
[Interpretable intrusion detection for IoT environments using a self-attention-based explainable AI framework](https://www.nature.com/articles/s41598-025-23750-0).

## Datasets

### UNSW-NB15
General-purpose network intrusion benchmark, using the pre-partitioned
Kaggle train/test release (`UNSW_NB15_training-set.csv` +
`testing-set.csv`, the same resource the paper links): 257,673 records
total, 43 features, 10 classes (1 benign + 9 attack categories). This is
severely imbalanced — Normal/Generic/Exploits alone are ~76% of the data,
and Worms is only 174 rows (~0.07%) — but no resampling is applied here,
matching the paper's stated approach for this dataset.

### BoT-IoT
IoT-specific botnet attack traffic (~72M records, 46 features, 5 classes:
Normal, DoS, DDoS, Reconnaissance, Theft). Severely imbalanced (Theft
and Normal are on the order of a few thousand raw rows, while DoS/DDoS are
tens of millions), resampled to 10k samples per class.

### N-BaIoT
Mirai & BASHLITE botnet traffic (~7M records, 115 features, 11 classes:
1 benign + 10 attack types). Also severely imbalanced, so resampled to 100k samples per
class.

## Project Structure
SA-DNN-LFG-Intrusion-Detection/
│
├── data/
│   ├── Bot-IoT/
│   ├── N-BaIoT/
│   └── UNSW-NB15/
│
├── Bot-IoT_model.py
├── N-BaIoT_model.py
├── UNSW-NB15_model.py
├── sa_dnn_lfg.py           # shared SA-DNN+LFG model architecture
├── explainability.py       # shared SHAP + LIME explainability
├── baselines.py            # shared Logistic Regression/SVM/RF/LSTM/CNN/BiLSTM+Attention comparison baselines
├── Bot-IoT_dataprep.py
├── N-BaIoT_dataprep.py
├── UNSW-NB15_dataprep.py
├── requirements.txt        # model dependencies
└── README.md

Install dependencies with `pip install -r requirements.txt`.

## Data Preparation

`Bot-IoT_dataprep.py` implements data preprocessing, data sampling, and feature selection for the Bot-IoT model.


`N-BaIoT_dataprep.py` implements data preprocessing, data sampling, and feature selection for the N-BaIoT model.

`UNSW-NB15_dataprep.py` implements data preprocessing, data sampling, and feature selection for the UNSW-NB15 model.


## Model Architecture

`sa_dnn_lfg.py` implements the shared SA-DNN+LFG architecture: three dense
layers (128→64→32, ReLU, dropout 0.3) feeding a 4-head self-attention block
(16-dim key/query/value), followed by a sigmoid-gated Learnable Feature
Gating layer, batch normalization, and a softmax classification head.
Weights use Xavier/Glorot initialization, trained with Adam (lr=5e-4),
batch size 64, up to 50 epochs with early stopping — matching the paper's
reported hyperparameters (Table 2).

## Running the Models

Bot-IoT and N-BaIoT require a pre-sampled CSV (produced by their
`*_datasample.py` script); UNSW-NB15 is pointed at its data folder directly.

```bash
# UNSW-NB15
python3 UNSW-NB15_model.py --dataset data/UNSW-NB15

# Bot-IoT
python3 Bot-IoT_datasample.py --dataset data/Bot-IoT # regenerate data/Bot-IoT_sampled.csv if stale
python3 Bot-IoT_model.py --dataset data/Bot-IoT_sampled.csv

# N-BaIoT
python3 N-BaIoT_datasample.py --dataset data/N-BaIoT  # regenerate data/N-BaIoT_sampled.csv if stale
python3 N-BaIoT_model.py --dataset data/N-BaIoT_sampled.csv
```

Each script trains SA-DNN+LFG, then by default also trains an SA-DNN (no
LFG) ablation, the baseline models, and runs SHAP/LIME explainability.
Useful flags (all three scripts):

- `--no-baselines` — skip the SA-DNN (no LFG) ablation and the Logistic
  Regression/SVM/RF/LSTM/CNN/BiLSTM+Attention baselines
- `--no-explain` — skip SHAP/LIME explainability
- `--explain-dir <path>` — override the explainability output directory
  (default `explainability_output/<dataset>/`)
- `--baseline-dir <path>` — override the baseline comparison CSV directory
  (default `baseline_results/`)

## Results

Each model script prints a classification report/confusion matrix for
SA-DNN+LFG and, by default, also trains an SA-DNN (no LFG) ablation plus
Logistic Regression, SVM, Random Forest, LSTM, CNN, and BiLSTM+Attention
baselines on the same preprocessed data for comparison
(accuracy/precision/recall/F1-macro, training time, inference latency). The
no-LFG ablation row isolates the contribution of the Learnable Feature
Gating layer, matching the paper's own SA-DNN vs. SA-DNN+LFG comparison.
The comparison table is saved to
`baseline_results/<dataset>_baseline_comparison.csv`. Skip this step with
`--no-baselines`.

## Explainability

`explainability.py` runs SHAP (global feature attribution via a
model-agnostic `shap.Explainer`, saved as summary bar/beeswarm plots and
per-instance waterfall plots) and LIME (local per-instance explanations,
saved as HTML + PNG) against the trained SA-DNN+LFG model. It is invoked
automatically by every `*_model.py` script and writes to
`explainability_output/<dataset>/` by default (override with
`--explain-dir`, or skip with `--no-explain`).

## References

Interpretable intrusion detection for IoT environments using a
self-attention-based explainable AI framework.
https://doi.org/10.1038/s41598-025-23750-0
