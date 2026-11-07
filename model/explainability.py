# SHAP + LIME explainability for SA-DNN models

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

RANDOM = 42

def explain_model(
    model,
    X_train,
    X_test,
    y_test,
    feature_names,
    class_names,
    output_dir,
    background_size=100,
    shap_sample_size=200,
    lime_num_features=10,
    random_state=RANDOM,
    overview_only=False,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    def predict_fn(x):
        return model.predict(np.asarray(x), verbose=0)

    shap_values = _run_shap(
        model, predict_fn, X_train, X_test, feature_names, class_names,
        output_dir, background_size, shap_sample_size, rng, overview_only,
    )
    _run_lime(
        predict_fn, X_train, X_test, y_test, feature_names, class_names,
        output_dir, lime_num_features, random_state, overview_only,
    )
    return shap_values

# SHAP used for global feature attributes
def _run_shap(model, predict_fn, X_train, X_test, feature_names, class_names,
              output_dir, background_size, shap_sample_size, rng, overview_only):
    print(f"\n[SHAP] Building explainer (background={background_size} samples)...")
    background = shap.sample(X_train, min(background_size, len(X_train)), random_state=0)

    n_explain = min(shap_sample_size, len(X_test))
    explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[explain_idx]

    explainer = shap.GradientExplainer(model, background)
    print(f"[SHAP] Computing SHAP values for {n_explain} test instances...")
    shap_values = explainer.shap_values(X_explain)
    check_shap_agreement(predict_fn, shap_values, X_explain, background, feature_names)

    # One beeswarm per class, showing per-feature SHAP value distribution for that class
    if not overview_only:
        for class_idx, class_name in enumerate(class_names):
            plt.figure()
            shap.summary_plot(
                shap_values[:, :, class_idx], X_explain, feature_names=feature_names,
                max_display=10, show=False,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{class_name}.png"), dpi=150)
            plt.close()

    # One overall waterfall explanation, for a single representative instance
    pred_class = int(np.argmax(predict_fn(X_explain[0:1]), axis=1)[0])
    expected_value = predict_fn(background).mean(axis=0)[pred_class]
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values[0, :, pred_class], feature_names=feature_names, show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall_overall.png"), dpi=150)
    plt.close()

    importances = np.abs(shap_values).mean(axis=(0, 2))
    _plot_feature_importance_bar(
        importances, feature_names, "Mean |SHAP value| per feature",
        os.path.join(output_dir, "shap_feature_importance_bar.png"), top_k=10,
    )

    print(f"[SHAP] Saved plots to {output_dir}")
    return shap_values


def _plot_feature_importance_bar(importances, feature_names, title, path, top_k=20):
    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    values = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(order))))
    ax.barh(names, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[SHAP] Saved feature importance bar chart to {path}")

# LIME used for local instance explainability
def _run_lime(predict_fn, X_train, X_test, y_test, feature_names, class_names,
              output_dir, lime_num_features, random_state, overview_only):
    print(f"\n[LIME] Building explainer...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )

    # One explanation per class, using a real test instance of that class
    if not overview_only:
        print(f"[LIME] Explaining one representative instance per class...")
        for class_idx, class_name in enumerate(class_names):
            matches = np.flatnonzero(y_test == class_idx)
            if len(matches) == 0:
                print(f"[LIME] No test instances found for class {class_name}, skipping")
                continue
            idx = matches[0]
            exp = lime_explainer.explain_instance(
                X_test[idx], predict_fn, num_features=lime_num_features, labels=(class_idx,),
            )
            fig = exp.as_pyplot_figure(label=class_idx)
            fig.savefig(
                os.path.join(output_dir, f"lime_{class_name}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

    # One overall explanation, for a single representative instance (its top predicted label)
    exp = lime_explainer.explain_instance(
        X_test[0], predict_fn, num_features=lime_num_features, top_labels=1,
    )
    top_label = exp.top_labels[0]
    fig = exp.as_pyplot_figure(label=top_label)
    fig.savefig(
        os.path.join(output_dir, "lime_overall.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[LIME] Saved explanations to {output_dir}")

def check_shap_agreement(predict_fn, shap_values, X_explain, background,
                         feature_names, k=10, n_check=20):
    F = X_explain.shape[1]
    ref = shap.Explainer(predict_fn, background[:50], algorithm="permutation")
    ref_vals = ref(X_explain[:n_check], max_evals=2 * F + 1).values  # (n, F, C)

    grad_imp = np.abs(shap_values[:n_check]).mean(axis=(0, 2))
    ref_imp = np.abs(ref_vals).mean(axis=(0, 2))

    top_grad = np.argsort(grad_imp)[::-1][:k]
    top_ref = np.argsort(ref_imp)[::-1][:k]
    overlap = len(set(top_grad) & set(top_ref))
    rho = spearmanr(grad_imp, ref_imp).correlation

    print(f"[SHAP check] top-{k} overlap: {overlap}/{k}, Spearman rho: {rho:.3f}")
    print("  Gradient top:   ", [feature_names[i] for i in top_grad])
    print("  Permutation top:", [feature_names[i] for i in top_ref])
