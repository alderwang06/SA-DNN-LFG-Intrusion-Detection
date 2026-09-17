# Count labels for each dataset

import os
import pandas as pd
import argparse
from collections import Counter

UNSW_DATASET = 'data/UNSW-NB15/'
BOT_DATASET = 'data/BoT-IoT/'
NBAIOT_DATASET = 'data/N-BaIoT'


# Label Count
class_counts = Counter()

print("UNSW-NB15 Dataset")
for filename in os.listdir(UNSW_DATASET):
    if 'set' in filename:
        filepath = os.path.join(UNSW_DATASET, filename)

        print("     Counting:", filename)

        for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
            class_counts.update(chunk['attack_cat'].value_counts().to_dict())
print("Class counts:")
for label, count in class_counts.items():
    print("     ", label, count)

print("\nN-BaIoT Dataset")
for filename in os.listdir(NBAIOT_DATASET):
    if filename.endswith(".csv") & filename[0].isdigit():
        print("     Counting:", filename)

        # Get class from filename
        parts = filename.replace(".csv", "").split(".")

        if parts[1] == "benign":
            label = "Benign"
        else:
            label = f"{parts[1]}-{parts[2]}"

        # Count rows
        df = pd.read_csv(
            os.path.join(NBAIOT_DATASET, filename),
            low_memory=False
        )

        class_counts[label] += len(df)

print("Class counts:")
for label, count in class_counts.items():
    print(f"    {label}: {count:,}")

print("\nBoT-IoT Dataset")
for filename in os.listdir(BOT_DATASET):
    if filename.endswith('.csv'):
        filepath = os.path.join(BOT_DATASET, filename)

        print("     Counting:", filename)

        for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
            class_counts.update(chunk['category'].value_counts().to_dict())
print("Class counts:")
for label, count in class_counts.items():
    print("     ", label, count)



