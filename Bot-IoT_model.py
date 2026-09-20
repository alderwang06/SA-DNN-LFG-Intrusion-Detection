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
TARGET = 10_000 # 10k samples per class
TOP_K_FEATURES = 20  # correlation-based feature selection target
REDUNDANCY_THRESHOLD = 0.95  # drop features this correlated with an already-kept feature

# Get sampled dataset path (produced by Bot-IoT_datasample.py)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the sampled Bot-IoT CSV file"
)

args = parser.parse_args()
DATASET_PATH = args.dataset

LABEL_COLS = ['attack', 'category']
CATEGORY_COLS = ['proto', 'state', 'flgs']
CLASSES = ['Reconnaissance', 'Normal', 'DoS', 'Theft', 'DDoS']

data = pd.read_csv(DATASET_PATH, low_memory=False)

# Data Split 70/15/15
train_data, temp_data = train_test_split(data, test_size=0.30, stratify=data['category'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.50, stratify=temp_data['category'], random_state=RANDOM)

print("\nBefore oversampling:")
print("Train:")
print(train_data['category'].value_counts())

print("\nValidation:")
print(val_data['category'].value_counts())

print("\nTest:")
print(test_data['category'].value_counts())

print("\nSampled dataset:")
print(data.shape)
print(data['category'].value_counts())

train_features = train_data.drop(columns=LABEL_COLS)
val_features = val_data.drop(columns=LABEL_COLS)
test_features = test_data.drop(columns=LABEL_COLS)

train_label = train_data['category']
val_label = val_data['category']
test_label = test_data['category']

# Oversampling
ros = RandomOverSampler(
    sampling_strategy={
        'Reconnaissance': TARGET,
        'Normal': TARGET,
        'DoS': TARGET,
        'Theft': TARGET,
        'DDoS': TARGET
    },
    random_state=RANDOM
)

train_features, train_label = ros.fit_resample(train_features, train_label)

print("\nPost-resample TRAIN class counts:")
print(train_label.value_counts())

# Label Encoding (done early so it's available for correlation-based feature selection)
label_encoder = LabelEncoder()
label_encoder.fit(train_label)

train_label_enc = label_encoder.transform(train_label)
val_label_enc = label_encoder.transform(val_label)
test_label_enc = label_encoder.transform(test_label)

num_classes = len(label_encoder.classes_)

# Data Preprocessing
encoders = {} # Encode categorical features
for col in CATEGORY_COLS:
    le = LabelEncoder()
    train_features[col] = train_features[col].astype(str)
    val_features[col] = val_features[col].astype(str)
    test_features[col] = test_features[col].astype(str)

    le.fit(train_features[col])
    train_classes = set(le.classes_)

    val_features[col] = val_features[col].apply(lambda x: x if x in train_classes else '__unseen__')
    test_features[col] = test_features[col].apply(lambda x: x if x in train_classes else '__unseen__')

    if '__unseen__' not in le.classes_:
        le.classes_ = np.append(le.classes_, '__unseen__')

    train_features[col] = le.transform(train_features[col])
    val_features[col] = le.transform(val_features[col])
    test_features[col] = le.transform(test_features[col])

    encoders[col] = le

# Data Cleaning
def parse_port(x):
    if pd.isna(x):
        return np.nan
    try:
        x = str(x).strip()
        if x.lower().startswith("0x"):
            return int(x, 16)
        return int(float(x))
    except (ValueError, TypeError):
        return np.nan

for df in [train_features, val_features, test_features]:
    df['sport'] = df['sport'].apply(parse_port)
    df['dport'] = df['dport'].apply(parse_port)

# Raw IP addresses are identifiers, not generalizable statistical features
train_features = train_features.drop(columns=['saddr', 'daddr'])
val_features = val_features.drop(columns=['saddr', 'daddr'])
test_features = test_features.drop(columns=['saddr', 'daddr'])

numeric = train_features.select_dtypes(include=np.number).columns

print("NaN counts before scaling:")
print(train_features.isna().sum()[train_features.isna().sum() > 0])

train_features = train_features.fillna(0)
val_features = val_features.fillna(0)
test_features = test_features.fillna(0)

# Correlation-based Feature Selection
def select_features_by_correlation(X, y, top_k, redundancy_threshold):
    corr_matrix = X.corr().abs()
    target_corr = X.apply(lambda col: col.corr(pd.Series(y, index=X.index))).abs().fillna(0)
    ranked = target_corr.sort_values(ascending=False).index.tolist()

    selected = []
    for feature in ranked:
        if any(corr_matrix.loc[feature, kept] > redundancy_threshold for kept in selected):
            continue
        selected.append(feature)
        if len(selected) >= top_k:
            break

    return selected

selected_features = select_features_by_correlation(
    train_features, train_label_enc, TOP_K_FEATURES, REDUNDANCY_THRESHOLD
)
print(f"\nSelected {len(selected_features)} features via correlation-based selection:")
print(selected_features)

train_features = train_features[selected_features]
val_features = val_features[selected_features]
test_features = test_features[selected_features]

numeric = train_features.columns

# Data Scaling
scaler = StandardScaler()
train_features[numeric] = scaler.fit_transform(train_features[numeric])
val_features[numeric] = scaler.transform(val_features[numeric])
test_features[numeric] = scaler.transform(test_features[numeric])

print(f'Train: {train_features.shape}  Val: {val_features.shape}  Test: {test_features.shape}')

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