import os
import tensorflow as tf
import pandas as pd
import argparse
from collections import Counter
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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
dataset = []

for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith('.csv'):
        print("Loading:", filename)
        datafile = pd.read_csv(os.path.join(DATASET_FOLDER + filename), low_memory=False)
        print("Loaded:", datafile.shape)
        dataset.append(datafile)

data = pd.concat(dataset, ignore_index=True)

print("Final Shape: " + data.shape)
print(data['category'].value_counts())

DROP_COLS = ['pkSeqID', 'seq', 'subcategory'] # include irrelevant features
LABEL_COLS = ['attack', 'category']

data = data.drop(DROP_COLS)

# Data Sampling
ros = RandomOverSampler(random_state=RANDOM)
rus = RandomUnderSampler(random_state=RANDOM)

data = ros.fit_resample(rus.fit_resample(data))

# Data Split
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['category'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=data['category'], random_state=RANDOM)

train_features = train_data.drop(columns=DROP_COLS)
val_features = val_data.drop(columns=DROP_COLS)
test_features = test_data.drop(columns=DROP_COLS)

train_label = train_data[LABEL_COLS]
val_label = val_data[LABEL_COLS]
test_label = test_data[LABEL_COLS]


# Data Preprocessing
scaler = StandardScaler()


# Feature Extraction

