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

# Any class whose raw population is below TARGET gets randomly oversampled
# (with replacement) up to TARGET, matching classes that were undersampled.
# This is the paper's implementation, might need to change because it oversamples
# the Theft class (1.6k) too much (10k)
parts = []
for category in CLASSES:
    if reservoirs[category] is None:
        print(f"Warning: no rows found for category {category}, skipping")
        continue

    pool = reservoirs[category].drop(columns=[KEY_COL])

    if len(pool) < TARGET:
        print(f"{category}: only {len(pool)} raw rows, oversampling to {TARGET}")
        pool = pool.sample(n=TARGET, replace=True, random_state=RANDOM)

    parts.append(pool)

data = pd.concat(parts, ignore_index=True)

print("\nSampled dataset:")
print(data.shape)
print(data['category'].value_counts())

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
