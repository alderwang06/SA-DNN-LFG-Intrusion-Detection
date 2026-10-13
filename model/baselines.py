# Baseline models for comparison against SA-DNN-LFG, reused across all three datasets
# Logistic Regression, SVM, Random Forest, LSTM, CNN, and BiLSTM+Attention

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay

RANDOM = 42
SVM_MAX_TRAIN_SAMPLES = 20_000  # SVC is O(n^2)-O(n^3); cap so it stays tractable on large datasets

COMPARISON_COLUMNS = [
    'model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
    'train_time_s', 'inference_ms_per_sample',
]


def build_lstm(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Reshape((input_dim, 1))(inputs)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Reshape((input_dim, 1))(inputs)
    x = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_bilstm_attention(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Reshape((input_dim, 1))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = keras.layers.GlobalAveragePooling1D()(attn)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def save_confusion_matrix(y_true, y_pred, class_names, title, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(class_names)), max(5, 0.6 * len(class_names))))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(class_names)), display_labels=class_names,
        cmap='Blues', xticks_rotation='vertical', ax=ax, colorbar=True,
    )
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")
    return disp.confusion_matrix


def save_training_curves(history, no_lfg_history, dataset_name, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs_lfg = range(1, len(history.history['accuracy']) + 1)
    epochs_no_lfg = range(1, len(no_lfg_history.history['accuracy']) + 1)

    ax.plot(epochs_lfg, history.history['accuracy'], label='SA-DNN-LFG train', color='#1f77b4', linestyle='-')
    ax.plot(epochs_lfg, history.history['val_accuracy'], label='SA-DNN-LFG val', color='#1f77b4', linestyle='--')
    ax.plot(epochs_no_lfg, no_lfg_history.history['accuracy'], label='SA-DNN (no LFG) train', color='#ff7f0e', linestyle='-')
    ax.plot(epochs_no_lfg, no_lfg_history.history['val_accuracy'], label='SA-DNN (no LFG) val', color='#ff7f0e', linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{dataset_name} — Training/Validation Accuracy', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {path}")


def _save_comparison_table_image(df, dataset_name, path):
    display_df = df.copy()
    for col in display_df.columns:
        if col != 'model':
            display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(1.7 * len(display_df.columns), 0.45 * (len(display_df) + 1)))
    ax.axis('off')
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#dddddd')

    ax.set_title(f"{dataset_name} — Model Comparison", fontweight='bold', pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison table image to {path}")


def compute_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
    }


def _subsample(X, y, max_samples, random_state=RANDOM):
    if len(X) <= max_samples:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def _train_sklearn_baseline(model, name, X_train, y_train, X_test, y_test, max_samples=None):
    if max_samples is not None:
        before = len(X_train)
        X_train, y_train = _subsample(X_train, y_train, max_samples)
        if len(X_train) < before:
            print(f"  [{name}] subsampled training set {before} -> {len(X_train)} rows for tractability")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    inference_ms_per_sample = (time.time() - start) / len(X_test) * 1000

    metrics = compute_classification_metrics(y_test, y_pred)
    metrics.update({'model': name, 'train_time_s': train_time, 'inference_ms_per_sample': inference_ms_per_sample})
    return metrics


def _train_keras_baseline(build_fn, name, input_dim, num_classes,
                           X_train, y_train_onehot, X_val, y_val_onehot,
                           X_test, y_test_enc, epochs, batch_size):
    keras.utils.set_random_seed(RANDOM)  # isolate each baseline from TF's global random state
    model = build_fn(input_dim, num_classes)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    start = time.time()
    model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - start

    start = time.time()
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    inference_ms_per_sample = (time.time() - start) / len(X_test) * 1000

    metrics = compute_classification_metrics(y_test_enc, y_pred)
    metrics.update({'model': name, 'train_time_s': train_time, 'inference_ms_per_sample': inference_ms_per_sample})
    return metrics


def run_comparison(
    dataset_name,
    X_train, y_train_enc, y_train_onehot,
    X_val, y_val_onehot,
    X_test, y_test_enc,
    num_classes,
    sa_dnn_lfg_metrics,
    no_lfg_metrics=None,
    epochs=30,
    batch_size=64,
    output_dir='baseline_results',
):
    os.makedirs(output_dir, exist_ok=True)
    input_dim = X_train.shape[1]

    results = [dict(sa_dnn_lfg_metrics, model='SA-DNN-LFG (proposed)')]
    if no_lfg_metrics is not None:
        results.append(dict(no_lfg_metrics, model='SA-DNN (no LFG, ablation)'))

    print(f"\n=== Baseline comparison: {dataset_name} ===")

    print("Training Logistic Regression...")
    results.append(_train_sklearn_baseline(
        LogisticRegression(max_iter=1000, random_state=RANDOM), 'Logistic Regression',
        X_train, y_train_enc, X_test, y_test_enc,
    ))

    print("Training SVM...")
    results.append(_train_sklearn_baseline(
        SVC(kernel='rbf', random_state=RANDOM), 'SVM',
        X_train, y_train_enc, X_test, y_test_enc, max_samples=SVM_MAX_TRAIN_SAMPLES,
    ))

    print("Training Random Forest...")
    results.append(_train_sklearn_baseline(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM, n_jobs=-1), 'Random Forest',
        X_train, y_train_enc, X_test, y_test_enc,
    ))

    for name, build_fn in [
        ('LSTM', build_lstm),
        ('CNN', build_cnn),
        ('BiLSTM+Attention', build_bilstm_attention),
    ]:
        print(f"Training {name}...")
        results.append(_train_keras_baseline(
            build_fn, name, input_dim, num_classes,
            X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_enc,
            epochs=epochs, batch_size=batch_size,
        ))

    df = pd.DataFrame(results)[COMPARISON_COLUMNS]
    print("\nComparison table:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    csv_path = os.path.join(output_dir, f"{dataset_name}_baseline_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")

    png_path = os.path.join(output_dir, f"{dataset_name}_baseline_comparison.png")
    _save_comparison_table_image(df, dataset_name, png_path)

    return df
