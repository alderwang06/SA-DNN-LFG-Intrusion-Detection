import os
import time
import numpy as np
import pandas as pd
import argparse
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import sa_dnn_lfg
import baselines
import explainability

RANDOM = 42
DATASET_NAME = "UNSW-NB15"

# Get Dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the split UNSW-NB15 CSV file (produced by UNSW-NB15_dataprep.py)"
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

LABEL_COLS = ['label', 'attack_cat']

# Get Preprocessed Data
data = pd.read_csv(DATASET_PATH, low_memory=False)
train_data = data[data['split'] == 'train'].drop(columns=['split'])
val_data = data[data['split'] == 'val'].drop(columns=['split'])
test_data = data[data['split'] == 'test'].drop(columns=['split'])

train_features = train_data.drop(columns=LABEL_COLS)
val_features = val_data.drop(columns=LABEL_COLS)
test_features = test_data.drop(columns=LABEL_COLS)

train_label = train_data[LABEL_COLS]
val_label = val_data[LABEL_COLS]
test_label = test_data[LABEL_COLS]

numeric = train_features.columns  # already encoded in UNSW-NB15_dataprep.py

scaler = StandardScaler() # Scale numerical features
train_features[numeric] = scaler.fit_transform(train_features[numeric])
val_features[numeric] = scaler.transform(val_features[numeric])
test_features[numeric] = scaler.transform(test_features[numeric])

print(f'Train: {train_features.shape}  Val: {val_features.shape}  Test: {test_features.shape}')

print(train_features.columns.tolist())
print(len(train_features.columns))

# Label Encoding
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

# Train Model
keras.utils.set_random_seed(RANDOM)  # isolate LFG vs no-LFG comparison from TF's global random state
model = sa_dnn_lfg.build_model(input_dim, num_classes)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
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

os.makedirs(args.baseline_dir, exist_ok=True)
baselines.save_confusion_matrix(
    y_true_classes, y_pred_classes, label_encoder.classes_,
    f"{DATASET_NAME} — SA-DNN-LFG Confusion Matrix",
    os.path.join(args.baseline_dir, f"{DATASET_NAME}_confusion_matrix_SA-DNN-LFG.png"),
)

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
    keras.utils.set_random_seed(RANDOM)
    no_lfg_model = sa_dnn_lfg.build_model_no_lfg(input_dim, num_classes)
    no_lfg_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    no_lfg_train_start = time.time()
    no_lfg_history = no_lfg_model.fit(
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

    baselines.save_confusion_matrix(
        y_true_classes, no_lfg_y_pred_classes, label_encoder.classes_,
        f"{DATASET_NAME} — SA-DNN (no LFG) Confusion Matrix",
        os.path.join(args.baseline_dir, f"{DATASET_NAME}_confusion_matrix_SA-DNN_no-LFG.png"),
    )

    baselines.save_training_curves(
        history, no_lfg_history, DATASET_NAME,
        os.path.join(args.baseline_dir, f"{DATASET_NAME}_training_curves.png"),
    )

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

    # Explainability (SHAP + LIME) for the no-LFG ablation
    if not args.no_explain:
        explainability.explain_model(
            model=no_lfg_model,
            X_train=train_features.values,
            X_test=test_features.values,
            y_test=test_label_enc,
            feature_names=train_features.columns.tolist(),
            class_names=list(label_encoder.classes_),
            output_dir=os.path.join(args.explain_dir, "SA-DNN_no-LFG"),
        )

# Explainability (SHAP + LIME) for the proposed SA-DNN-LFG model
if not args.no_explain:
    explainability.explain_model(
        model=model,
        X_train=train_features.values,
        X_test=test_features.values,
        y_test=test_label_enc,
        feature_names=train_features.columns.tolist(),
        class_names=list(label_encoder.classes_),
        output_dir=os.path.join(args.explain_dir, "SA-DNN-LFG"),
    )