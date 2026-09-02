import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

RANDOM = 42

# Load Data
DATASET_FOLDER = 'data/BoT-IoT/'
dataset = []

for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith('.csv'):
        datafile = pd.read_csv(DATASET_FOLDER + filename)
        dataset.append(datafile)

data = pd.concat(dataset, ignore_index=True)

print(data.shape)
print(data['category'].value_counts())

DROP_COLS = ['pkSeqID', 'seq', 'subcategory'] # include irrelevant features
LABEL_COLS = ['attack', 'category']

data = data.drop(DROP_COLS)

# Data Sampling
ros = RandomOverSampler(random_state=RANDOM)
rus = RandomUnderSampler(random_state=RANDOM)

data = ros.fit_resample(rus.fit_resample(data))



# Data Split
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['attack_cat'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=data['attack_cat'], random_state=RANDOM)

train_features = train_data.drop(columns=DROP_COLS)
val_features = val_data.drop(columns=DROP_COLS)
test_features = test_data.drop(columns=DROP_COLS)

train_label = train_data[LABEL_COLS]
val_label = val_data[LABEL_COLS]
test_label = test_data[LABEL_COLS]


# Data Preprocessing
scaler = StandardScaler()


# Feature Extraction

