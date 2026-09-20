import os
import re
import argparse
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

def parse_label(filename):
    match = LABEL_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1).replace('.', '_')

# Data Sampling
samples = {c: [] for c in CLASSES}
counts = {c: 0 for c in CLASSES}

for filename in os.listdir(DATASET_FOLDER):
    if not filename.endswith('.csv'):
        continue

    label = parse_label(filename)
    if label not in CLASSES:
        continue

    if counts[label] >= TARGET:
        continue

    filepath = os.path.join(DATASET_FOLDER, filename)
    print("Loading:", filename, "-> label:", label)

    for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
        if counts[label] >= TARGET:
            break

        chunk.columns = chunk.columns.str.strip()
        chunk['label'] = label

        remaining = TARGET - counts[label]

        if len(chunk) > remaining:
            chunk = chunk.sample(n=remaining, random_state=RANDOM)

        samples[label].append(chunk)
        counts[label] += len(chunk)

    print("Current counts:", counts)

data = pd.concat(  # Combine all samples
    [pd.concat(samples[c], ignore_index=True)
    for c in CLASSES if samples[c]],
    ignore_index=True
)

print("\nSampled dataset:")
print(data.shape)
print(data['label'].value_counts())

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

data.to_csv(args.output, index=False)
print(f"\nSaved sampled dataset to {args.output}")