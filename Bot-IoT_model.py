import time

import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import sa_dnn_lfg
import baselines
import explainability

RANDOM = 42
TOP_K_FEATURES = 20  # correlation-based feature selection target
REDUNDANCY_THRESHOLD = 0.95  # drop features this correlated with an already-kept feature
DATASET_NAME = "Bot-IoT"

# Get sampled dataset path (produced by Bot-IoT_datasample.py)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the sampled Bot-IoT CSV file"
)
parser.add_argument(
    "--no-baselines", action="store_true",
    help="Skip training LogisticRegression/SVM/RF/LSTM/CNN/BiLSTM+Attention comparison baselines"
)
parser.add_argument(
    "--no-explain", action="store_true",
    help="Skip SHAP/LIME explainability"
)
parser.add_argument(
    "--explain-dir", default=f"explainability_output/{DATASET_NAME}",
    help="Directory to write SHAP/LIME plots to"
)
parser.add_argument(
    "--baseline-dir", default="baseline_results",
    help="Directory to write baseline comparison CSVs to"
)

args = parser.parse_args()
DATASET_PATH = args.dataset

LABEL_COLS = ['attack', 'category']
CATEGORY_COLS = ['proto', 'state', 'flgs']
CLASSES = ['Reconnaissance', 'Normal', 'DoS', 'Theft', 'DDoS']

data = pd.read_csv(DATASET_PATH, low_memory=False)

# Data Split 70/15/15: Bot-IoT_datasample.py already assigns each row to a
# split (before oversampling, so no row can appear in more than one split).
train_data = data[data['split'] == 'train'].drop(columns=['split'])
val_data = data[data['split'] == 'val'].drop(columns=['split'])
test_data = data[data['split'] == 'test'].drop(columns=['split'])

print("\nClass counts after split:")
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

# Classes are already balanced per class by Bot-IoT_datasample.py before
# this split, so no further oversampling is needed here.

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

for df in [train_features, val_features, test_features]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("NaN counts before scaling:")
print(train_features.isna().sum()[train_features.isna().sum() > 0])

train_features = train_features.fillna(0)
val_features = val_features.fillna(0)
test_features = test_features.fillna(0)

# Correlation-based Feature Selection
def select_features_by_correlation(X, y, top_k, redundancy_threshold):
    corr_matrix = X.corr().abs()

    # y is label-encoded (arbitrary integer per class), which has no ordinal
    # meaning across a 5-class target. Correlating directly against it would
    # implicitly treat "class 3 > class 1" as meaningful. Instead, correlate
    # each feature against a one-hot indicator per class and keep the
    # strongest per-class association as that feature's relevance score.
    y_onehot = pd.get_dummies(pd.Series(y, index=X.index))
    per_class_corr = pd.DataFrame({cls: X.corrwith(y_onehot[cls]) for cls in y_onehot.columns})
    target_corr = per_class_corr.abs().max(axis=1).fillna(0)
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
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

train_start = time.time()
history = model.fit(
    train_features.values,
    train_label_onehot,
    validation_data=(val_features.values, val_label_onehot),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)
train_time_s = time.time() - train_start

test_loss, test_accuracy = model.evaluate(test_features.values, test_label_onehot)
print(f'Test Loss: {test_loss:.4f}  Test Accuracy: {test_accuracy:.4f}')

inference_start = time.time()
y_pred = model.predict(test_features.values)
inference_ms_per_sample = (time.time() - inference_start) / len(test_features) * 1000
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_label_onehot, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
print(confusion_matrix(y_true_classes, y_pred_classes))

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true_classes, y_pred_classes, average='macro', zero_division=0
)
sa_dnn_lfg_metrics = {
    'accuracy': test_accuracy,
    'precision_macro': precision_macro,
    'recall_macro': recall_macro,
    'f1_macro': f1_macro,
    'train_time_s': train_time_s,
    'inference_ms_per_sample': inference_ms_per_sample,
}

# Baseline Model Comparison (Logistic Regression, SVM, Random Forest, LSTM, CNN, BiLSTM+Attention)
# + SA-DNN (no LFG) ablation, matching the paper's LFG contribution analysis
if not args.no_baselines:
    no_lfg_model = sa_dnn_lfg.build_model_no_lfg(input_dim, num_classes)
    no_lfg_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    no_lfg_train_start = time.time()
    no_lfg_model.fit(
        train_features.values, train_label_onehot,
        validation_data=(val_features.values, val_label_onehot),
        epochs=50, batch_size=64, callbacks=no_lfg_callbacks,
    )
    no_lfg_train_time_s = time.time() - no_lfg_train_start

    no_lfg_test_loss, no_lfg_test_accuracy = no_lfg_model.evaluate(test_features.values, test_label_onehot)
    print(f'[SA-DNN no-LFG ablation] Test Loss: {no_lfg_test_loss:.4f}  Test Accuracy: {no_lfg_test_accuracy:.4f}')

    no_lfg_inference_start = time.time()
    no_lfg_y_pred_classes = np.argmax(no_lfg_model.predict(test_features.values), axis=1)
    no_lfg_inference_ms_per_sample = (time.time() - no_lfg_inference_start) / len(test_features) * 1000

    no_lfg_precision_macro, no_lfg_recall_macro, no_lfg_f1_macro, _ = precision_recall_fscore_support(
        y_true_classes, no_lfg_y_pred_classes, average='macro', zero_division=0
    )
    no_lfg_metrics = {
        'accuracy': no_lfg_test_accuracy,
        'precision_macro': no_lfg_precision_macro,
        'recall_macro': no_lfg_recall_macro,
        'f1_macro': no_lfg_f1_macro,
        'train_time_s': no_lfg_train_time_s,
        'inference_ms_per_sample': no_lfg_inference_ms_per_sample,
    }

    baselines.run_comparison(
        dataset_name=DATASET_NAME,
        X_train=train_features.values, y_train_enc=train_label_enc, y_train_onehot=train_label_onehot,
        X_val=val_features.values, y_val_onehot=val_label_onehot,
        X_test=test_features.values, y_test_enc=test_label_enc,
        num_classes=num_classes,
        sa_dnn_lfg_metrics=sa_dnn_lfg_metrics,
        no_lfg_metrics=no_lfg_metrics,
        output_dir=args.baseline_dir,
    )

# Explainability (SHAP + LIME)
if not args.no_explain:
    explainability.explain_model(
        model=model,
        X_train=train_features.values,
        X_test=test_features.values,
        feature_names=selected_features,
        class_names=list(label_encoder.classes_),
        output_dir=args.explain_dir,
    )