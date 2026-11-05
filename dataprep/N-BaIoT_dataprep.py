import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RANDOM = 42
TARGET = 50_000
TOP_K_FEATURES = 20  # Random Forest-based feature selection

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the raw N-BaIoT dataset folder"
)
parser.add_argument(
    "--output",
    default="data/N-BaIoT_sampled.csv",
    help="Path to write the sampled CSV"
)
parser.add_argument(
    "--device",
    default="1",
    help="N-BaIoT device ID to sample from (default 1 = Danmini_Doorbell, "
         "matching the paper, which trains/evaluates on this single device "
         "rather than pooling traffic across all 9 devices)"
)

args = parser.parse_args()
DATASET_FOLDER = args.dataset
DEVICE = args.device

CLASSES = [
    'benign',
    'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp',
    'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain'
]

LABEL_PATTERN = re.compile(r'^(\d+)\.(.+)\.csv$')
KEY_COL = '_reservoir_key'

def parse_device_and_label(filename):
    match = LABEL_PATTERN.match(filename)
    if not match:
        return None, None
    return match.group(1), match.group(2).replace('.', '_')

rng = np.random.default_rng(RANDOM)

# keep uniform random sample per label
reservoirs = {c: None for c in CLASSES}
seen = {c: 0 for c in CLASSES}

for filename in sorted(os.listdir(DATASET_FOLDER)):
    if not filename.endswith('.csv'):
        continue

    device_id, label = parse_device_and_label(filename)
    if device_id != DEVICE or label not in CLASSES:
        continue

    filepath = os.path.join(DATASET_FOLDER, filename)
    print("Loading:", filename, "-> label:", label)

    for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk['label'] = label

        seen[label] += len(chunk)
        chunk[KEY_COL] = rng.random(len(chunk))

        combined = chunk if reservoirs[label] is None else pd.concat(
            [reservoirs[label], chunk], ignore_index=True
        )
        if len(combined) > TARGET:
            combined = combined.nlargest(TARGET, KEY_COL)
        reservoirs[label] = combined

    print("Rows seen so far:", seen)

# oversample/undersample after split
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
TRAIN_TARGET = round(TARGET * SPLIT_RATIOS['train'])

parts = {split: [] for split in SPLIT_RATIOS}
for label in CLASSES:
    if reservoirs[label] is None:
        print(f"Warning: no rows found for label {label}, skipping")
        continue

    pool = reservoirs[label].drop(columns=[KEY_COL])
    # drop exact duplicates
    pool = pool.drop_duplicates().reset_index(drop=True)
    pool = pool.sample(frac=1, random_state=RANDOM).reset_index(drop=True)

    n_train = int(round(len(pool) * SPLIT_RATIOS['train']))
    n_val = int(round(len(pool) * SPLIT_RATIOS['val']))
    raw_splits = {
        'train': pool.iloc[:n_train],
        'val': pool.iloc[n_train:n_train + n_val],
        'test': pool.iloc[n_train + n_val:],
    }

    for split, split_df in raw_splits.items():
        if split == 'train' and len(split_df) != TRAIN_TARGET:
            action = 'undersampling' if len(split_df) > TRAIN_TARGET else 'oversampling (with replacement)'
            print(f"{label}/train: {len(split_df)} raw rows, {action} to {TRAIN_TARGET}")
            split_df = split_df.sample(
                n=TRAIN_TARGET, replace=len(split_df) < TRAIN_TARGET, random_state=RANDOM,
            )
        parts[split].append(split_df)

data = pd.concat(
    [df.assign(split=split) for split, dfs in parts.items() for df in dfs],
    ignore_index=True,
)

print("\nSampled dataset:")
print(data.shape)
print(data.groupby('split')['label'].value_counts())

# remove NaN/inf before feature selection
data = data.replace([np.inf, -np.inf], np.nan)
na_counts = data.isna().sum()
print("\nNaN counts before feature selection:")
print(na_counts[na_counts > 0])
data = data.fillna(0)

# Random forest-based feature selection
feature_cols = [c for c in data.columns if c not in ('label', 'split')]
train_rows = data[data['split'] == 'train']

rf_selector = RandomForestClassifier(n_estimators=200, random_state=RANDOM, n_jobs=-1)
rf_selector.fit(train_rows[feature_cols], train_rows['label'])

importances = pd.Series(rf_selector.feature_importances_, index=feature_cols).sort_values(ascending=False)
selected_features = importances.head(TOP_K_FEATURES).index.tolist()

print(f"\nSelected {len(selected_features)} features via Random Forest importance:")
print(selected_features)

data = data[selected_features + ['label', 'split']]

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
