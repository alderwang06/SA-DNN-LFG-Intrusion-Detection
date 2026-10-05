# SA-DNN-LFG-Intrusion-Detection

## Overview
This is an implementation of the interpretable intrusion detection model in IoT
environments presented in the paper:
[Interpretable intrusion detection for IoT environments using a self-attention-based explainable AI framework](https://www.nature.com/articles/s41598-025-23750-0).

## Datasets

### UNSW-NB15
General-purpose network intrusion benchmark (~2.54M records, 49 features,
10 classes: 1 benign + 9 attack categories). Already reasonably balanced
across its major attack categories, so no resampling is applied.

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
├── sa_dnn_lfg.py              # shared SA-DNN+LFG model architecture
├── explainability.py          # shared SHAP + LIME explainability
├── baselines.py                # shared Logistic Regression/SVM/RF/LSTM/CNN/BiLSTM+Attention comparison baselines
├── Bot-IoT_model.py
├── N-BaIoT_model.py
├── UNSW-NB15_model.py
└── README.md

Install dependencies with `pip install -r requirements.txt`.

## Data Preparation

For Bot-IoT and N-BaIoT, `Bot-IoT_datasample.py` / `N-BaIoT_datasample.py`
first reduce the raw per-file dataset to a fixed number of samples per class
(10k for Bot-IoT, 100k for N-BaIoT — N-BaIoT's target is confirmed directly
from the paper text; BoT-IoT's isn't stated unambiguously in the paper, so
10k was chosen to keep oversampling of its rarest class, Theft at ~1.6k raw
rows, to a defensible ~6x rather than stretching a handful of real samples
much further): a streaming reservoir sample keeps a uniform random subset of
up to the target size per class as all source files are read (so majority
classes are randomly undersampled without bias toward whichever file happens
to be read first), and any class whose raw population falls short of the
target is randomly oversampled (with replacement) back up to it — matching
the paper's combined oversample-minority/undersample-majority approach,
applied before the data is split.

Each `*_model.py` script then loads its (already class-balanced, for Bot-IoT
and N-BaIoT) dataset, splits it 70/15/15 (train/val/test, stratified),
label-/one-hot-encodes targets, scales numeric features with `StandardScaler`,
and reduces Bot-IoT/N-BaIoT to their top features (correlation-based for
Bot-IoT, Random Forest importance for N-BaIoT); UNSW-NB15 keeps all original
features and is not resampled, matching the paper.

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
