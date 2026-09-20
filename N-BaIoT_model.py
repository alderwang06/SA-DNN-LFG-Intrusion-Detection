import argparse
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix

import sa_dnn_lfg

RANDOM = 42
TARGET = 100_000  # 100k samples per class

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the dataset folder"
)

args = parser.parse_args()
data = pd.read_csv(args.dataset, low_memory=False)

print("\nLoaded dataset:")
print(data.shape)
print(data['label'].value_counts())

# Data Split 70/15/15
train_data, temp_data = train_test_split(data, test_size=0.30, stratify=data['label'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.50, stratify=temp_data['label'], random_state=RANDOM)

print("\nBefore oversampling:")
print("Train:")
print(train_data['label'].value_counts())

print("\nValidation:")
print(val_data['label'].value_counts())

print("\nTest:")
print(test_data['label'].value_counts())

train_features = train_data.drop(columns=['label'])
val_features = val_data.drop(columns=['label'])
test_features = test_data.drop(columns=['label'])

train_label = train_data['label']
val_label = val_data['label']
test_label = test_data['label']

# Oversampling (brings any under-target train classes up to TARGET)
ros = RandomOverSampler(
    sampling_strategy={c: TARGET for c in train_label.unique()},
    random_state=RANDOM
)

train_features, train_label = ros.fit_resample(train_features, train_label)

print("\nPost-resample TRAIN class counts:")
print(train_label.value_counts())

numeric = train_features.select_dtypes(include=np.number).columns

print("NaN counts before scaling:")
print(train_features.isna().sum()[train_features.isna().sum() > 0])

train_features = train_features.fillna(0)
val_features = val_features.fillna(0)
test_features = test_features.fillna(0)

# Data Scaling
scaler = StandardScaler()
train_features[numeric] = scaler.fit_transform(train_features[numeric])
val_features[numeric] = scaler.transform(val_features[numeric])
test_features[numeric] = scaler.transform(test_features[numeric])

print(f'Train: {train_features.shape}  Val: {val_features.shape}  Test: {test_features.shape}')

# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(train_label)

train_label_enc = label_encoder.transform(train_label)
val_label_enc = label_encoder.transform(val_label)
test_label_enc = label_encoder.transform(test_label)

num_classes = len(label_encoder.classes_)

train_label_onehot = keras.utils.to_categorical(train_label_enc, num_classes=num_classes)
val_label_onehot = keras.utils.to_categorical(val_label_enc, num_classes=num_classes)
test_label_onehot = keras.utils.to_categorical(test_label_enc, num_classes=num_classes)

input_dim = train_features.shape[1]

# Train Model
model = sa_dnn_lfg.build_model(input_dim, num_classes)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(
    train_features.values,
    train_label_onehot,
    validation_data=(val_features.values, val_label_onehot),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_features.values, test_label_onehot)
print(f'Test Loss: {test_loss:.4f}  Test Accuracy: {test_accuracy:.4f}')

y_pred = model.predict(test_features.values)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_label_onehot, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
print(confusion_matrix(y_true_classes, y_pred_classes))