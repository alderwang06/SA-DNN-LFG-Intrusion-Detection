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
    y_test,
    feature_names,
    class_names,
    output_dir,
    background_size=100,
    shap_sample_size=200,
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
        predict_fn, X_train, X_test, y_test, feature_names, class_names,
        output_dir, lime_num_features, random_state,
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

    # One beeswarm per class, showing per-feature SHAP value distribution for that class
    for class_idx, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(
            shap_values[:, :, class_idx], X_explain, feature_names=feature_names, show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{class_name}.png"), dpi=150)
        plt.close()

    # One overall waterfall explanation, for a single representative instance
    pred_class = int(np.argmax(predict_fn(X_explain[0:1]), axis=1)[0])
    plt.figure()
    shap.plots.waterfall(shap_values[0, :, pred_class], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall_overall.png"), dpi=150)
    plt.close()

    print(f"[SHAP] Saved plots to {output_dir}")
    return shap_values

# LIME used for local instance explainability
def _run_lime(predict_fn, X_train, X_test, y_test, feature_names, class_names,
              output_dir, lime_num_features, random_state):
    print(f"\n[LIME] Building explainer...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )

    # One explanation per class, using a real test instance of that class
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
