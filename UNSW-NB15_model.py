import os
import tensorflow as tf
import pandas as pd
import argparse
from collections import Counter
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM = 42

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the dataset folder"
)

args = parser.parse_args()
DATASET_FOLDER = args.dataset

# Load Data
DATASET_FOLDER = 'data/UNSW-NB15'
data1 = pd.read_csv(DATASET_FOLDER + '/UNSW_NB15_training-set.csv')
data2 = pd.read_csv(DATASET_FOLDER + '/UNSW_NB15_testing-set.csv')
data = pd.concat([data1, data2], ignore_index=True)

ID_COLS = ['id']
LABEL_COLS = ['label', 'attack_cat']
CATEGORY_COLS = ['proto', 'service', 'state']

# Data Split
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['attack_cat'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['attack_cat'], random_state=RANDOM)

train_features = train_data.drop(columns=LABEL_COLS + ID_COLS)
val_features = val_data.drop(columns=LABEL_COLS + ID_COLS)
test_features = test_data.drop(columns=LABEL_COLS + ID_COLS)

train_label = train_data[LABEL_COLS]
val_label = val_data[LABEL_COLS]
test_label = test_data[LABEL_COLS]

# Data Preprocessing
numeric = train_features.drop(columns=CATEGORY_COLS).columns

scaler = StandardScaler()
train_features[numeric] = scaler.fit_transform(train_features[numeric])
val_features[numeric] = scaler.transform(val_features[numeric])
test_features[numeric] = scaler.transform(test_features[numeric])
# should i exclude binary features?

print(f'Train: {train_features.shape}  Val: {val_features.shape}  Test: {test_features.shape}')

# Feature Extraction
