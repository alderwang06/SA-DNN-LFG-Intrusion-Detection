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
total, 43 features, 10 classes (1 benign + 9 attack categories). Severely
imbalanced — Normal/Generic/Exploits alone are ~76% of the raw data, and
Worms is only 174 rows (~0.07%). A large share of the raw rows are also
exact duplicates (~37%, concentrated almost entirely in the `Generic` and
`DoS` classes) and are dropped before splitting. No oversampling/undersampling
is applied, matching the paper's stated approach for this dataset.

### BoT-IoT
IoT-specific botnet attack traffic (~72M raw records, 5 classes: Normal,
DoS, DDoS, Reconnaissance, Theft). Severely imbalanced in the raw data
(Theft ~1.6k rows, Normal ~9.5k rows, vs. tens of millions for DoS/DDoS).
Resampled to 10k samples per class — the paper doesn't state an exact
BoT-IoT target, so 10k was chosen to keep oversampling of the rarest class
(Theft) to a defensible ~6x rather than stretching a handful of real
samples much further. Reduced to the top 20 features via correlation-based
selection (CFS).

### N-BaIoT
Mirai & BASHLITE botnet traffic, 11 classes (1 benign + 10 attack types),
115 statistical features (no protocol/port/IP fields — see Limitations).
Restricted to a single device, Danmini Doorbell (device 1), matching the
paper's own methodology ("this research...specifically utilized data from
the Danmini Doorbell"), rather than pooling traffic across all 9 N-BaIoT
devices. Resampled to 100k samples per class — this target is stated
directly in the paper. Reduced to the top 20 features via Random Forest
importance.

## Project Structure
SA-DNN-LFG-Intrusion-Detection/
│
├── data/
│   ├── Bot-IoT/
│   ├── N-BaIoT/
│   └── UNSW-NB15/
│
├── dataprep/
│   ├── Bot-IoT_dataprep.py     # sampling, preprocessing, CFS feature selection
│   ├── N-BaIoT_dataprep.py     # sampling, preprocessing, RF feature selection
│   └── UNSW-NB15_dataprep.py   # dedup, preprocessing (no sampling/selection)
│
├── model/
│   ├── Bot-IoT_model.py
│   ├── N-BaIoT_model.py
│   ├── UNSW-NB15_model.py
│   ├── sa_dnn_lfg.py           # shared SA-DNN+LFG model architecture
│   ├── explainability.py       # shared SHAP + LIME explainability
│   └── baselines.py            # shared Logistic Regression/SVM/RF/LSTM/CNN/BiLSTM+Attention comparison baselines
│
├── other/
│   ├── datacheck.py            # standalone class-count inspection utility
│   └── dataset.py
│
├── requirements.txt
└── README.md

Install dependencies with `pip install -r requirements.txt`.

## Data Preparation

Each `dataprep/*_dataprep.py` script is fully self-contained: it owns
sampling, preprocessing, and feature selection, and writes a single CSV
with a `split` column (`train`/`val`/`test`) already assigned. The
corresponding `model/*_model.py` script only scales that CSV and trains —
it does no encoding, cleaning, or feature selection itself.

**Split-before-oversample.** For Bot-IoT and N-BaIoT, a streaming reservoir
sample keeps a uniform random subset of up to the per-class target as all
source files are read (undersampling majority classes without bias toward
whichever file is read first). Exact-duplicate rows are then dropped, the
remaining pool is split 70/15/15 *before* any oversampling, and only then
is each split independently oversampled (with replacement) up to its
proportional target. This order matters: oversampling before the split (the
original approach) let duplicated rows land in both train and test, letting
models like Random Forest "memorize" test rows they'd also seen in
training. Splitting first means a duplicated row can only ever land back in
the split it came from. UNSW-NB15 gets the same duplicate-drop treatment
but no resampling, since the paper applies none to this dataset.

**Preprocessing**, fit only on the train split so val/test never leak into
encoders/scalers/selectors:
- Bot-IoT: hex-to-decimal port conversion (`sport`/`dport`), raw IP
  addresses (`saddr`/`daddr`) dropped as non-generalizable identifiers,
  `proto`/`state`/`flgs` label-encoded, NaN/inf replaced with 0.
- N-BaIoT: NaN/inf replaced with 0 (no categorical or port/IP fields exist
  in this dataset's feature set).
- UNSW-NB15: `id` dropped, `proto`/`service`/`state` label-encoded.

**Feature selection**, also fit only on the train split:
- Bot-IoT: correlation-based selection (CFS) — ranks features by their
  strongest per-class correlation, drops any feature too correlated
  (>0.95) with an already-kept one, and excludes zero-variance columns
  from candidacy (several raw Bot-IoT columns, e.g. `soui`/`doui`/`dco`,
  are entirely empty in this dataset release and would otherwise fill
  selected slots with pure noise). Keeps the top 20.
- N-BaIoT: Random Forest importance ranking, keeps the top 20.
- UNSW-NB15: no selection — uses the full feature set, matching the paper.

Scaling (`StandardScaler`, fit on train only) happens in `model/*_model.py`,
after loading the already-split, already-preprocessed CSV.

### Regenerating the sampled/split CSVs

```bash
python3 dataprep/UNSW-NB15_dataprep.py --dataset data/UNSW-NB15 --output data/UNSW-NB15_sampled.csv
python3 dataprep/Bot-IoT_dataprep.py --dataset data/Bot-IoT --output data/Bot-IoT_sampled.csv
python3 dataprep/N-BaIoT_dataprep.py --dataset data/N-BaIoT --output data/N-BaIoT_sampled.csv
```

`N-BaIoT_dataprep.py` also accepts `--device <id>` (default `1` = Danmini
Doorbell) to sample from a different device, or to pool multiple devices by
running it multiple times against different `--output` paths.

## Model Architecture

`model/sa_dnn_lfg.py` implements the shared SA-DNN+LFG architecture: three dense
layers (128→64→32, ReLU, dropout 0.3) feeding a 4-head self-attention block
(16-dim key/query/value), a feed-forward + layer-norm block, a sigmoid-gated
Learnable Feature Gating layer, batch normalization, and a softmax
classification head. Weights use Xavier/Glorot initialization, trained with
Adam (lr=5e-4), batch size 64, up to 50 epochs with early stopping
(patience=5) — matching the paper's reported hyperparameters (Table 2).
`keras.utils.set_random_seed()` is called immediately before each model is
built, so the SA-DNN-LFG vs. SA-DNN (no-LFG) comparison starts from an
identical, reproducible random state rather than one model consuming
TensorFlow's global RNG stream before the other's turn.

## Running the Models

Each `model/*_model.py` script loads the CSV produced by its matching
`dataprep/*_dataprep.py` script (see Data Preparation above for how to
regenerate them if stale).

```bash
# UNSW-NB15
python3 model/UNSW-NB15_model.py --dataset data/UNSW-NB15_sampled.csv

# Bot-IoT
python3 model/Bot-IoT_model.py --dataset data/Bot-IoT_sampled.csv

# N-BaIoT
python3 model/N-BaIoT_model.py --dataset data/N-BaIoT_sampled.csv
```

Each script trains SA-DNN+LFG, then by default also trains an SA-DNN (no
LFG) ablation, the baseline models, and runs SHAP/LIME explainability for
both. Useful flags (all three scripts unless noted):

- `--no-baselines` — skip the SA-DNN (no LFG) ablation and the Logistic
  Regression/SVM/RF/LSTM/CNN/BiLSTM+Attention baselines
- `--no-explain` — skip SHAP/LIME explainability
- `--explain-dir <path>` — override the explainability output directory
  (default `explainability_output/<dataset>/`)
- `--baseline-dir <path>` — override the baseline comparison output directory
  (default `baseline_results/`)
- `--leak-check-only` (Bot-IoT and N-BaIoT only) — run only the train/test
  duplicate-row diagnostic (exact-match rate, and Random Forest accuracy
  split between rows seen vs. not seen in training) and exit, skipping all
  model training

## Results

Each model script prints a classification report for SA-DNN+LFG and, by
default, also trains an SA-DNN (no LFG) ablation plus Logistic Regression,
SVM, Random Forest, LSTM, CNN, and BiLSTM+Attention baselines on the same
preprocessed data for comparison (accuracy/precision/recall/F1-macro,
training time, inference latency). The no-LFG ablation isolates the
contribution of the Learnable Feature Gating layer, matching the paper's
own SA-DNN vs. SA-DNN+LFG comparison. Skip this step with `--no-baselines`.
Saved to `baseline_results/`:

- `<dataset>_baseline_comparison.csv` / `.png` — the full comparison table,
  as data and as a rendered image
- `<dataset>_confusion_matrix_SA-DNN-LFG.png` /
  `<dataset>_confusion_matrix_SA-DNN_no-LFG.png` — `ConfusionMatrixDisplay`
  heatmaps for both models
- `<dataset>_training_curves.png` — train/validation accuracy per epoch,
  four lines (SA-DNN-LFG train/val, SA-DNN no-LFG train/val) on one plot

## Explainability

`model/explainability.py` runs SHAP and LIME against **both** the SA-DNN+LFG
model and the SA-DNN (no-LFG) ablation (skip with `--no-explain`), writing
to `explainability_output/<dataset>/SA-DNN-LFG/` and
`explainability_output/<dataset>/SA-DNN_no-LFG/` respectively. For each
model:

- **SHAP**: one beeswarm plot per class (`shap_beeswarm_<class>.png`,
  showing the per-feature SHAP value distribution for that class) plus one
  overall waterfall plot (`shap_waterfall_overall.png`) for a single
  representative prediction.
- **LIME**: one feature-importance plot per class (`lime_<class>.png`,
  using a real test instance of that class) plus one overall plot
  (`lime_overall.png`) for a single representative prediction.

## Limitations

- **N-BaIoT's feature set has no protocol field.** All 115 features are
  statistical aggregates (packet size/rate/jitter) computed over decaying
  time windows — there is no column recording whether traffic was TCP or
  UDP. This makes `gafgyt_tcp` and `gafgyt_udp` close to indistinguishable
  from the available features alone (SA-DNN+LFG collapses `gafgyt_tcp`
  entirely into `gafgyt_udp` in practice), independent of model choice.
- **N-BaIoT's decaying-window features are highly autocorrelated in time.**
  Consecutive rows from a sustained attack session can be near-identical
  even after deduplication and feature selection, which a random (not
  time-ordered) train/val/test split doesn't fully account for. This is a
  known limitation of the current split strategy, not yet addressed.

## References

Interpretable intrusion detection for IoT environments using a
self-attention-based explainable AI framework.
https://doi.org/10.1038/s41598-025-23750-0
