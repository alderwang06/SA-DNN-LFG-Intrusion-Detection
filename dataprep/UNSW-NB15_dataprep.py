import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC

RANDOM = 42
CEILING = 10_000 # Max samples
FLOOR = 1_000
CATEGORY_COLS = ['proto', 'service', 'state']

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

DROP_COLS = ['id']  # row identifier, not a generalizable statistical feature

data1 = pd.read_csv(os.path.join(DATASET_FOLDER, 'UNSW_NB15_training-set.csv'))
data2 = pd.read_csv(os.path.join(DATASET_FOLDER, 'UNSW_NB15_testing-set.csv'))
data = pd.concat([data1, data2], ignore_index=True)
data = data.drop(columns=DROP_COLS, errors='ignore')

print("Loaded dataset:")
print(data.shape)
print(data['attack_cat'].value_counts())

# Drop exact-duplicate raw rows so a duplicate that already exists in the
# raw data can't get scattered across train/val/test by chance.
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)
print(f"\nDropped {before - len(data)} duplicate rows")

# Stratified 70/15/15 train/val/test split, done *before* any resampling so
# a duplicated/synthetic row can only ever land back in the split it came
# from (see CEILING/FLOOR comment above for why Bot-IoT/N-BaIoT's leakage
# fix matters here too).
train_data, temp_data = train_test_split(
    data, test_size=0.3, stratify=data['attack_cat'], random_state=RANDOM
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_data['attack_cat'], random_state=RANDOM
)

SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
raw_splits = {'train': train_data, 'val': val_data, 'test': test_data}

parts = {}
for split, split_df in raw_splits.items():
    ratio = SPLIT_RATIOS[split]
    ceiling = round(CEILING * ratio)
    floor = round(FLOOR * ratio)
    counts = split_df['attack_cat'].value_counts()

    undersample_targets = {cls for cls, n in counts.items() if n > ceiling}
    if undersample_targets:
        kept = []
        for cls, group in split_df.groupby('attack_cat'):
            if cls in undersample_targets:
                group = group.sample(n=ceiling, random_state=RANDOM)
            kept.append(group)
        split_df = pd.concat(kept, ignore_index=True)
        counts = split_df['attack_cat'].value_counts()

    oversample_targets = {cls: floor for cls, n in counts.items() if n < floor}
    if oversample_targets:
        print(f"{split}: SMOTE-NC oversampling {oversample_targets}")
        y = split_df['attack_cat']
        # 'label' is fully determined by attack_cat (0 for Normal, 1
        # otherwise), so it's dropped here and re-derived after SMOTE rather
        # than asking SMOTE-NC to keep two redundant targets consistent on
        # synthetic rows.
        X = split_df.drop(columns=['attack_cat', 'label'])
        categorical_idx = [X.columns.get_loc(c) for c in CATEGORY_COLS]
        min_minority = min(counts[cls] for cls in oversample_targets)
        k_neighbors = max(1, min(5, min_minority - 1))

        smote = SMOTENC(
            categorical_features=categorical_idx,
            sampling_strategy=oversample_targets,
            k_neighbors=k_neighbors,
            random_state=RANDOM,
        )
        X_res, attack_cat_res = smote.fit_resample(X, y)

        split_df = X_res
        split_df['attack_cat'] = attack_cat_res
        split_df['label'] = (attack_cat_res != 'Normal').astype(int)
    else:
        split_df = split_df.reset_index(drop=True)

    parts[split] = split_df.assign(split=split)

result = pd.concat(parts.values(), ignore_index=True)

print("\nSplit dataset:")
print(result.shape)
print(result.groupby('split')['attack_cat'].value_counts())

# Encode categorical features, fit only on the train split so val/test
# never influence the encoding.
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
