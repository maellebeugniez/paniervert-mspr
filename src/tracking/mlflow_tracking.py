from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mining.classification import (  # noqa: E402
    GOLD_DIR,
    MODELS_DIR,
    prepare_features_and_target,
    resolve_input_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_shap_summary(model: object, X_test: pd.DataFrame, output_path: Path) -> None:
    """Calcule les contributions SHAP sur un échantillon et exporte le graphique résumé."""
    final_step = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(final_step, "feature_importances_"):
        return

    shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(final_step)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        arrays = [np.asarray(v) for v in shap_values]
        shap_matrix = np.mean(np.abs(np.stack(arrays, axis=0)), axis=0)
    else:
        values = np.asarray(shap_values)
        n_samples, n_features = shap_sample.shape
        if values.ndim == 2:
            shap_matrix = values
        elif values.ndim == 3:
            if values.shape[:2] == (n_samples, n_features):
                shap_matrix = np.mean(np.abs(values), axis=-1)
            elif values.shape[1:] == (n_samples, n_features):
                shap_matrix = np.mean(np.abs(values), axis=0)
            else:
                shap_matrix = np.mean(np.abs(values), axis=-1)
        else:
            shap_matrix = np.mean(np.abs(values), axis=0)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_matrix, shap_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(target_col: str = "segment") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = resolve_input_dataset()
    if input_path is None:
        raise FileNotFoundError("Fichier segments.parquet ou user_features.parquet introuvable.")

    df = pd.read_parquet(input_path)
    X, y, feature_cols, _ = prepare_features_and_target(df, target_col=target_col)

    class_counts = y.value_counts()
    can_stratify = len(class_counts) >= 2 and int(class_counts.min()) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        ),
    }

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("paniervert_classification")

    best_name = ""
    best_model = None
    best_f1 = -1.0

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("dataset", str(input_path.relative_to(PROJECT_ROOT)))
            mlflow.log_param("target_column", target_col)
            mlflow.log_param("features", ",".join(feature_cols))

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1_weighted = float(f1_score(y_test, preds, average="weighted", zero_division=0))
            f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
            })
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=X_train.head(100),
                registered_model_name="paniervert_segment_classifier",
            )

            if f1_weighted > best_f1:
                best_f1 = f1_weighted
                best_name = name
                best_model = model

    if best_model is None:
        raise RuntimeError("Aucun modèle n'a pu être entraîné.")

    shap_path = MODELS_DIR / "shap_summary.png"
    cm_path = MODELS_DIR / "confusion_matrix.png"
    report_path = MODELS_DIR / "classification_report.json"

    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.set_tag("model_role", "best")
        mlflow.set_tag("dataset", str(input_path.relative_to(PROJECT_ROOT)))
        mlflow.log_param("model_name", best_name)
        mlflow.log_param("target_column", target_col)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_metric("f1_weighted", best_f1)
        model_info = mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=X_train.head(100),
        )

        QUALITY_GATE_F1 = 0.95
        if best_f1 >= QUALITY_GATE_F1:
            client = MlflowClient()
            client.set_registered_model_alias(
                "paniervert_segment_classifier",
                "champion",
                str(model_info.registered_model_version),
            )
            mlflow.set_tag("quality_gate", "passed")
            logging.info("Modèle promu champion : version %s", model_info.registered_model_version)
        else:
            mlflow.set_tag("quality_gate", "failed")
            logging.warning("Quality gate échouée : F1 %.4f < %.4f", best_f1, QUALITY_GATE_F1)

        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))
        if report_path.exists():
            mlflow.log_artifact(str(report_path))
        if best_name == "random_forest":
            save_shap_summary(best_model, X_test, shap_path)
            if shap_path.exists():
                mlflow.log_artifact(str(shap_path))
                logging.info("Graphique SHAP sauvegardé : %s", shap_path)

    logging.info("Meilleur modèle : %s (F1 pondéré = %.4f)", best_name, best_f1)
    return PROJECT_ROOT / "mlruns"


if __name__ == "__main__":
    main()
