# SHAP + LIME explainability for SA-DNN models

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

RANDOM = 42


def explain_model(
    model,
    X_train,
    X_test,
    feature_names,
    class_names,
    output_dir,
    background_size=100,
    shap_sample_size=200,
    lime_num_instances=5,
    lime_num_features=10,
    random_state=RANDOM,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    def predict_fn(x):
        return model.predict(np.asarray(x), verbose=0)

    shap_values = _run_shap(
        predict_fn, X_train, X_test, feature_names, class_names,
        output_dir, background_size, shap_sample_size, rng,
    )
    _run_lime(
        predict_fn, X_train, X_test, feature_names, class_names,
        output_dir, lime_num_instances, lime_num_features, rng, random_state,
    )
    return shap_values

# SHAP used for global feature attributes
def _run_shap(predict_fn, X_train, X_test, feature_names, class_names,
              output_dir, background_size, shap_sample_size, rng):
    print(f"\n[SHAP] Building explainer (background={background_size} samples)...")
    background = shap.sample(X_train, min(background_size, len(X_train)), random_state=0)

    n_explain = min(shap_sample_size, len(X_test))
    explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[explain_idx]

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    print(f"[SHAP] Computing SHAP values for {n_explain} test instances...")
    shap_values = explainer(X_explain)

    # Global importance: mean(|SHAP value|) per feature, summed over classes
    plt.figure()
    shap.summary_plot(
        shap_values, X_explain, feature_names=feature_names,
        class_names=class_names, plot_type="bar", show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=150)
    plt.close()

    # Beeswarm for the class with the largest mean |SHAP| (most "active" class)
    class_impact = np.abs(shap_values.values).mean(axis=(0, 1))
    top_class = int(np.argmax(class_impact))
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, top_class], X_explain, feature_names=feature_names, show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{class_names[top_class]}.png"), dpi=150)
    plt.close()

    # Local waterfall explanations for a few individual predictions
    preds = np.argmax(predict_fn(X_explain), axis=1)
    for i in range(min(3, len(X_explain))):
        pred_class = int(preds[i])
        plt.figure()
        shap.plots.waterfall(shap_values[i, :, pred_class], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_instance{i}_{class_names[pred_class]}.png"), dpi=150)
        plt.close()

    print(f"[SHAP] Saved plots to {output_dir}")
    return shap_values

# LIME used for local instance explainability
def _run_lime(predict_fn, X_train, X_test, feature_names, class_names,
              output_dir, lime_num_instances, lime_num_features, rng, random_state):
    print(f"\n[LIME] Building explainer...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )

    n_instances = min(lime_num_instances, len(X_test))
    lime_idx = rng.choice(len(X_test), size=n_instances, replace=False)

    print(f"[LIME] Explaining {n_instances} individual instances...")
    for i, idx in enumerate(lime_idx):
        exp = lime_explainer.explain_instance(
            X_test[idx], predict_fn, num_features=lime_num_features, top_labels=1,
        )
        top_label = exp.top_labels[0]

        exp.save_to_file(os.path.join(output_dir, f"lime_instance{i}_{class_names[top_label]}.html"))

        fig = exp.as_pyplot_figure(label=top_label)
        fig.savefig(
            os.path.join(output_dir, f"lime_instance{i}_{class_names[top_label]}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"[LIME] Saved explanations to {output_dir}")
