import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM = 42
TARGET = 30_000


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the raw UNSW-NB15 dataset folder"
)
parser.add_argument(
    "--output",
    default="data/UNSW-NB15_sampled.csv",
    help="Path to write the split CSV"
)

args = parser.parse_args()
DATASET_FOLDER = args.dataset
DROP_COLS = ['id']

data1 = pd.read_csv(os.path.join(DATASET_FOLDER, 'UNSW_NB15_training-set.csv'))
data2 = pd.read_csv(os.path.join(DATASET_FOLDER, 'UNSW_NB15_testing-set.csv'))
data = pd.concat([data1, data2], ignore_index=True)
data = data.drop(columns=DROP_COLS, errors='ignore')

print("Loaded dataset:")
print(data.shape)
print(data['attack_cat'].value_counts())

# drop exact duplicat
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)
print(f"\nDropped {before - len(data)} duplicate rows")

train_data, temp_data = train_test_split(
    data, test_size=0.3, stratify=data['attack_cat'], random_state=RANDOM
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_data['attack_cat'], random_state=RANDOM
)

# only train is resampled to TARGET; val/test keep the real distribution
counts = train_data['attack_cat'].value_counts()
resampled = []
for cls, group in train_data.groupby('attack_cat'):
    n = counts[cls]
    if n != TARGET:
        action = 'undersampling' if n > TARGET else 'oversampling (with replacement)'
        print(f"train/{cls}: {n} raw rows, {action} to {TARGET}")
        group = group.sample(n=TARGET, replace=n < TARGET, random_state=RANDOM)
    resampled.append(group)
train_data = pd.concat(resampled, ignore_index=True)

train_data = train_data.assign(split='train')
val_data = val_data.assign(split='val')
test_data = test_data.assign(split='test')

result = pd.concat([train_data, val_data, test_data], ignore_index=True)

print("\nSplit dataset:")
print(result.shape)
print(result.groupby('split')['attack_cat'].value_counts())

# Encode categorical features, fit only on the train split so val/test
# never influence the encoding.
CATEGORY_COLS = ['proto', 'service', 'state']
train_mask = result['split'] == 'train'

for col in CATEGORY_COLS:
    le = LabelEncoder()
    result[col] = result[col].astype(str)

    le.fit(result.loc[train_mask, col])
    train_classes = set(le.classes_)
    result[col] = result[col].apply(lambda x: x if x in train_classes else '__unseen__')

    if '__unseen__' not in le.classes_:
        le.classes_ = np.append(le.classes_, '__unseen__')

    result[col] = le.transform(result[col])

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

result.to_csv(args.output, index=False)
print(f"\nSaved split dataset to {args.output}")
