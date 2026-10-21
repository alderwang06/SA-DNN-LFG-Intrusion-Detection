import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RANDOM = 42
# Paper's exact BoT-IoT per-class resampling target is unverifiable (its "normal
# traffic dominates" oversampling example doesn't match BoT-IoT's real class
# counts, where Normal/Theft are the rare classes). 10k keeps oversampling of
# the rarest class (Theft, ~1.6k raw rows) at a defensible ~6x rather than ~46x.
TARGET = 10_000  # 10k samples per class
TOP_K_FEATURES = 20  # correlation-based feature selection target
REDUNDANCY_THRESHOLD = 0.95  # drop features this correlated with an already-kept feature
LABEL_COLS = ['attack', 'category']
CATEGORY_COLS = ['proto', 'state', 'flgs']

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the raw Bot-IoT dataset folder"
)
parser.add_argument(
    "--output",
    default="data/Bot-IoT_sampled.csv",
    help="Path to write the sampled CSV"
)

args = parser.parse_args()
DATASET_FOLDER = args.dataset

# saddr/daddr are raw IP identifiers (not generalizable statistical
# features); smac/dmac/soui/doui/sco/dco are 100% NaN in this dataset.
# Dropped early so neither group gets carried through resampling for no
# reason.
DROP_COLS = [
    'pkSeqID', 'seq', 'subcategory', 'stime', 'ltime',
    'saddr', 'daddr', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco',
]
CLASSES = ['Reconnaissance', 'Normal', 'DoS', 'Theft', 'DDoS']
KEY_COL = '_reservoir_key'

rng = np.random.default_rng(RANDOM)


def parse_ports(series):
    """sport/dport are read as strings and occasionally hex ('0x50'); this
    parses them numeric like every other continuous column. Vectorized over
    the whole column instead of a per-row .apply() -- across ~73M Bot-IoT
    rows, a Python-level function call per row (x2, for sport and dport) is
    the actual bottleneck in this script, not the number of files read."""
    s = series.astype(str).str.strip()
    is_hex = s.str.lower().str.startswith('0x')

    result = pd.to_numeric(s.where(~is_hex), errors='coerce')
    if is_hex.any():
        result.loc[is_hex] = s.loc[is_hex].apply(lambda v: int(v, 16))

    return result.astype('float64')

# Data Sampling: reservoir sampling keeps a uniform random sample of up to
# TARGET rows per class as the full dataset streams past, regardless of file
# order, so classes with more than TARGET raw rows are randomly undersampled
# without the greedy first-N-rows-encountered bias of a single-chunk sample.
reservoirs = {c: None for c in CLASSES}
seen = {c: 0 for c in CLASSES}

for filename in sorted(os.listdir(DATASET_FOLDER)):
    if not filename.endswith('.csv'):
        continue

    filepath = os.path.join(DATASET_FOLDER, filename)
    print("Loading:", filename)

    for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.drop(columns=DROP_COLS, errors='ignore')
        chunk['sport'] = parse_ports(chunk['sport'])
        chunk['dport'] = parse_ports(chunk['dport'])
        # remove NaN
        numeric_cols = chunk.select_dtypes(include=np.number).columns
        chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # groupby once instead of re-scanning the whole chunk with a fresh
        # boolean mask per class (5x over the same 100k-row chunk).
        for category, class_data in chunk.groupby('category'):
            if category not in CLASSES:
                continue

            seen[category] += len(class_data)
            class_data = class_data.copy()
            class_data[KEY_COL] = rng.random(len(class_data))

            combined = class_data if reservoirs[category] is None else pd.concat(
                [reservoirs[category], class_data], ignore_index=True
            )
            if len(combined) > TARGET:
                combined = combined.nlargest(TARGET, KEY_COL)
            reservoirs[category] = combined

    print("Rows seen so far:", seen)

# Split each class's deduplicated pool into train/val/test *before* any
# resampling, then under/oversample independently within each split. A
# resampled row can then only ever land back in the split it came from, so
# no row can appear in both train and test (the previous oversample-then-
# split order allowed that, which let models like Random Forest "memorize"
# test rows they'd also seen, exact duplicate, in training).
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SPLIT_TARGETS = {
    'train': round(TARGET * SPLIT_RATIOS['train']),
    'val': round(TARGET * SPLIT_RATIOS['val']),
}
SPLIT_TARGETS['test'] = TARGET - SPLIT_TARGETS['train'] - SPLIT_TARGETS['val']

raw_splits_by_category = {}
for category in CLASSES:
    if reservoirs[category] is None:
        print(f"Warning: no rows found for category {category}, skipping")
        continue

    pool = reservoirs[category].drop(columns=[KEY_COL])
    # Drop exact-duplicate raw rows (e.g. repetitive benign heartbeat traffic)
    # before splitting, so a duplicate that already existed in the raw data
    # can't get scattered across train/val/test by chance.
    pool = pool.drop_duplicates().reset_index(drop=True)
    pool = pool.sample(frac=1, random_state=RANDOM).reset_index(drop=True)

    n_train = int(round(len(pool) * SPLIT_RATIOS['train']))
    n_val = int(round(len(pool) * SPLIT_RATIOS['val']))
    raw_splits_by_category[category] = {
        'train': pool.iloc[:n_train],
        'val': pool.iloc[n_train:n_train + n_val],
        'test': pool.iloc[n_train + n_val:],
    }

# Within each split, classes above target are randomly undersampled and
# classes below target are oversampled with replacement, up to target. A
# resampled row can then only ever land back in the split it came from, so
# no row can appear in both train and test.
parts = {}
for split, target in SPLIT_TARGETS.items():
    split_df = pd.concat(
        [raw_splits_by_category[c][split].assign(category=c) for c in raw_splits_by_category],
        ignore_index=True,
    )
    counts = split_df['category'].value_counts()

    resampled = []
    for c, group in split_df.groupby('category'):
        n = counts[c]
        if n != target:
            print(f"{split}/{c}: {n} raw rows, resampling to {target}")
            group = group.sample(n=target, replace=n < target, random_state=RANDOM)
        resampled.append(group)

    parts[split] = pd.concat(resampled, ignore_index=True)

data = pd.concat(
    [df.assign(split=split) for split, df in parts.items()],
    ignore_index=True,
)

print("\nSampled dataset:")
print(data.shape)
print(data['category'].value_counts())

train_mask = data['split'] == 'train'

for col in CATEGORY_COLS:
    le = LabelEncoder()
    data[col] = data[col].astype(str)

    le.fit(data.loc[train_mask, col])
    train_classes = set(le.classes_)
    data[col] = data[col].apply(lambda x: x if x in train_classes else '__unseen__')

    if '__unseen__' not in le.classes_:
        le.classes_ = np.append(le.classes_, '__unseen__')

    data[col] = le.transform(data[col])

data = data.replace([np.inf, -np.inf], np.nan)
na_counts = data.isna().sum()
print("\nNaN counts before feature selection:")
print(na_counts[na_counts > 0])
data = data.fillna(0)

# Correlation-based Feature Selection (CFS), fit only on the train split
def select_features_by_correlation(X, y, top_k, redundancy_threshold):
    # Zero-variance (constant, e.g. all-NaN-then-filled-to-0) columns have
    # undefined correlation with anything -- corr() gives NaN, and
    # "NaN > redundancy_threshold" is always False, so a constant column
    # can never be flagged as redundant and silently fills a selected slot
    # despite carrying zero information. Drop them from candidacy entirely.
    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols) > 0:
        print(f"Dropping {len(constant_cols)} zero-variance columns from CFS candidacy: {constant_cols.tolist()}")
        X = X.drop(columns=constant_cols)

    corr_matrix = X.corr().abs()

    # y is label-encoded (arbitrary integer per class), which has no ordinal
    # meaning across a 5-class target. Correlating directly against it would
    # implicitly treat "class 3 > class 1" as meaningful. Instead, correlate
    # each feature against a one-hot indicator per class and keep the
    # strongest per-class association as that feature's relevance score.
    y_onehot = pd.get_dummies(pd.Series(y, index=X.index))
    per_class_corr = pd.DataFrame({cls: X.corrwith(y_onehot[cls]) for cls in y_onehot.columns})
    target_corr = per_class_corr.abs().max(axis=1).fillna(0)
    ranked = target_corr.sort_values(ascending=False).index.tolist()

    selected = []
    for feature in ranked:
        if any(corr_matrix.loc[feature, kept] > redundancy_threshold for kept in selected):
            continue
        selected.append(feature)
        if len(selected) >= top_k:
            break

    return selected

train_features = data.loc[train_mask].drop(columns=LABEL_COLS + ['split'])
train_label_enc = LabelEncoder().fit_transform(data.loc[train_mask, 'category'])

selected_features = select_features_by_correlation(
    train_features, train_label_enc, TOP_K_FEATURES, REDUNDANCY_THRESHOLD
)
print(f"\nSelected {len(selected_features)} features via correlation-based selection:")
print(selected_features)

data = data[selected_features + LABEL_COLS + ['split']]

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
