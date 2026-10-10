import os
import argparse
import numpy as np
import pandas as pd

RANDOM = 42
# Paper's exact BoT-IoT per-class resampling target is unverifiable (its "normal
# traffic dominates" oversampling example doesn't match BoT-IoT's real class
# counts, where Normal/Theft are the rare classes). 10k keeps oversampling of
# the rarest class (Theft, ~1.6k raw rows) at a defensible ~6x rather than ~46x.
TARGET = 10_000  # 10k samples per class

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

DROP_COLS = ['pkSeqID', 'seq', 'subcategory', 'stime', 'ltime']
CLASSES = ['Reconnaissance', 'Normal', 'DoS', 'Theft', 'DDoS']
KEY_COL = '_reservoir_key'

rng = np.random.default_rng(RANDOM)

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

        for category in CLASSES:
            class_data = chunk[chunk['category'] == category]
            if len(class_data) == 0:
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
# oversampling, then oversample (with replacement, up to the paper's target)
# independently within each split. A duplicated row can then only ever land
# back in the split it was duplicated from, so no row can appear in both
# train and test (the previous oversample-then-split order allowed that,
# which let models like Random Forest "memorize" test rows they'd also
# seen, exact duplicate, in training).
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SPLIT_TARGETS = {
    'train': round(TARGET * SPLIT_RATIOS['train']),
    'val': round(TARGET * SPLIT_RATIOS['val']),
}
SPLIT_TARGETS['test'] = TARGET - SPLIT_TARGETS['train'] - SPLIT_TARGETS['val']

parts = {split: [] for split in SPLIT_RATIOS}
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
    raw_splits = {
        'train': pool.iloc[:n_train],
        'val': pool.iloc[n_train:n_train + n_val],
        'test': pool.iloc[n_train + n_val:],
    }

    for split, split_df in raw_splits.items():
        target = SPLIT_TARGETS[split]
        if len(split_df) != target:
            print(f"{category}/{split}: {len(split_df)} raw rows, resampling to {target}")
            split_df = split_df.sample(n=target, replace=len(split_df) < target, random_state=RANDOM)
        parts[split].append(split_df)

data = pd.concat(
    [df.assign(split=split) for split, dfs in parts.items() for df in dfs],
    ignore_index=True,
)

print("\nSampled dataset:")
print(data.shape)
print(data['category'].value_counts())

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
