import os
import argparse
import ipaddress
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix


RANDOM = 42
TARGET = 10_000 # 10k samples per class

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the dataset folder"
)

args = parser.parse_args()
DATASET_FOLDER = args.dataset

DROP_COLS = ['pkSeqID', 'seq', 'subcategory', 'stime', 'ltime']
LABEL_COLS = ['attack', 'category']
CATEGORY_COLS = ['proto', 'state', 'flgs']
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

data = pd.concat( # Combine all samples
    [pd.concat(samples[c], ignore_index=True)
    for c in CLASSES],
    ignore_index=True
)

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

def ip_features(x, prefix):
    try:
        ip = ipaddress.ip_address(str(x).strip())

        return pd.Series({
            f'{prefix}_private': int(ip.is_private),
            f'{prefix}_loopback': int(ip.is_loopback),
            f'{prefix}_multicast': int(ip.is_multicast),
            f'{prefix}_first_octet': int(str(ip).split('.')[0]),
        })

    except (ValueError, TypeError):
        return pd.Series({
            f'{prefix}_private': np.nan,
            f'{prefix}_loopback': np.nan,
            f'{prefix}_multicast': np.nan,
            f'{prefix}_first_octet': np.nan,
        })

def add_ip_features(df):
    saddr_features = df['saddr'].apply(
        lambda x: ip_features(x, 'saddr')
    )
    daddr_features = df['daddr'].apply(
        lambda x: ip_features(x, 'daddr')
    )
    df = pd.concat(
        [
            df.drop(columns=['saddr', 'daddr']),
            saddr_features,
            daddr_features
        ],
        axis=1
    )
    return df

for df in [train_features, val_features, test_features]:
    df['sport'] = df['sport'].apply(parse_port)
    df['dport'] = df['dport'].apply(parse_port)

train_features = add_ip_features(train_features)
val_features = add_ip_features(val_features)
test_features = add_ip_features(test_features)

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

# SA-DNN Model Architecture
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


inputs = keras.Input(shape=(input_dim,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)

attn_input = keras.layers.Reshape((32, 1))(x)
attn_output = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, value_dim=16, output_shape=16)(attn_input, attn_input, attn_input)
attn_output = keras.layers.Flatten()(attn_output)

ff = keras.layers.Dense(64, activation='relu')(attn_output)
ff = keras.layers.LayerNormalization()(ff)

# Learnable Feature Gating
gate = keras.layers.Dense(64, activation='sigmoid')(ff)
gated = keras.layers.Multiply()([gate, ff])

bn = keras.layers.BatchNormalization()(gated)

clf = keras.layers.Dense(16, activation='relu')(bn)
outputs = keras.layers.Dense(num_classes, activation='softmax')(clf)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
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