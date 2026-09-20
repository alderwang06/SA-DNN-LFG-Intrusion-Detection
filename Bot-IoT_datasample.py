import os
import argparse
import pandas as pd

RANDOM = 42
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

# Data Sampling (Need to change undersampling technique)
samples = {c: [] for c in CLASSES}
counts = {c: 0 for c in CLASSES}

for filename in os.listdir(DATASET_FOLDER):
    if not filename.endswith('.csv'):
        continue

    filepath = os.path.join(DATASET_FOLDER, filename)
    print("Loading:", filename)

    for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.drop(columns=DROP_COLS, errors='ignore')

        for category in CLASSES:
            if counts[category] >= TARGET:
                continue

            class_data = chunk[
                chunk['category'] == category
            ]

            if len(class_data) == 0:
                continue

            remaining = TARGET - counts[category]

            if len(class_data) > remaining:
                class_data = class_data.sample(n=remaining, random_state=RANDOM)

            samples[category].append(class_data)
            counts[category] += len(class_data)

    print("Current counts:", counts)

    # Stop reading files once every class hits 10k
    if all(counts[c] >= TARGET for c in CLASSES):
        break

data = pd.concat(  # Combine all samples
    [pd.concat(samples[c], ignore_index=True)
    for c in CLASSES],
    ignore_index=True
)

print("\nSampled dataset:")
print(data.shape)
print(data['category'].value_counts())

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")
