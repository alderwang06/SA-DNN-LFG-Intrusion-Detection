import pandas as pd
import argparse
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
data1 = pd.read_csv(DATASET_FOLDER + '/UNSW_NB15_training-set.csv')
data2 = pd.read_csv(DATASET_FOLDER + '/UNSW_NB15_testing-set.csv')
data = pd.concat([data1, data2], ignore_index=True)

ID_COLS = ['id']
LABEL_COLS = ['label', 'attack_cat']
CATEGORY_COLS = ['proto', 'service', 'state']

# Data Split 70/15/15
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['attack_cat'], random_state=RANDOM)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['attack_cat'], random_state=RANDOM)

train_features = train_data.drop(columns=LABEL_COLS + ID_COLS)
val_features = val_data.drop(columns=LABEL_COLS + ID_COLS)
test_features = test_data.drop(columns=LABEL_COLS + ID_COLS)

train_label = train_data[LABEL_COLS]
val_label = val_data[LABEL_COLS]
test_label = test_data[LABEL_COLS]

# Data Preprocessing
encoders = {} # Encode categorical features into vectors
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
        le.classes_ = pd.np.append(le.classes_, '__unseen__') if hasattr(pd, 'np') else __import__('numpy').append(le.classes_, '__unseen__')

    train_features[col] = le.transform(train_features[col])
    val_features[col] = le.transform(val_features[col])
    test_features[col] = le.transform(test_features[col])

    encoders[col] = le

numeric = train_features.drop(columns=CATEGORY_COLS).columns

scaler = StandardScaler() # Scale numerical features
train_features[numeric] = scaler.fit_transform(train_features[numeric])
val_features[numeric] = scaler.transform(val_features[numeric])
test_features[numeric] = scaler.transform(test_features[numeric])

print(f'Train: {train_features.shape}  Val: {val_features.shape}  Test: {test_features.shape}')

# SA-DNN Model Architecture
label_encoder = LabelEncoder()
label_encoder.fit(train_label['attack_cat'])

train_label_enc = label_encoder.transform(train_label['attack_cat'])
val_label_enc = label_encoder.transform(val_label['attack_cat'])
test_label_enc = label_encoder.transform(test_label['attack_cat'])

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