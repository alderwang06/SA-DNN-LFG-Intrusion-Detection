import os
import re
import argparse
import numpy as np
import pandas as pd

RANDOM = 42
TARGET = 100_000  # 100k samples per class

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

args = parser.parse_args()
DATASET_FOLDER = args.dataset

CLASSES = [
    'benign',
    'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp',
    'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain'
]

LABEL_PATTERN = re.compile(r'^\d+\.(.+)\.csv$')
KEY_COL = '_reservoir_key'

def parse_label(filename):
    match = LABEL_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1).replace('.', '_')

rng = np.random.default_rng(RANDOM)

# Data Sampling: each label is spread across multiple per-device files
# (e.g. 1.mirai.udp.csv ... 9.mirai.udp.csv). Reservoir sampling streams
# through every device's file for a label and keeps a uniform random sample
# of up to TARGET rows, so the result isn't biased toward whichever device
# happens to be listed first (and stops short of TARGET, unsampled, once
# the earliest-listed devices alone satisfy it).
reservoirs = {c: None for c in CLASSES}
seen = {c: 0 for c in CLASSES}

for filename in sorted(os.listdir(DATASET_FOLDER)):
    if not filename.endswith('.csv'):
        continue

    label = parse_label(filename)
    if label not in CLASSES:
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

# Any class whose raw population is below TARGET gets randomly oversampled
# (with replacement) up to TARGET, matching classes that were undersampled.
parts = []
for label in CLASSES:
    if reservoirs[label] is None:
        print(f"Warning: no rows found for label {label}, skipping")
        continue

    pool = reservoirs[label].drop(columns=[KEY_COL])

    if len(pool) < TARGET:
        print(f"{label}: only {len(pool)} raw rows, oversampling to {TARGET}")
        pool = pool.sample(n=TARGET, replace=True, random_state=RANDOM)

    parts.append(pool)

data = pd.concat(parts, ignore_index=True)

print("\nSampled dataset:")
print(data.shape)
print(data['label'].value_counts())

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
