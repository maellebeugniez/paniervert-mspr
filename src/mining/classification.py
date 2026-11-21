from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"

EXCLUDED_FEATURE_COLS = {
    "user_id",
    "order_id",
    "product_id",
    "segment",
    "segment_nom",
    "cluster",
    "pca_1",
    "pca_2",
}


def resolve_input_dataset() -> Path | None:
    """Localise la table enrichie (segments Gold en priorité, sinon user_features)."""
    candidates = [
        GOLD_DIR / "segments.parquet",
        PROCESSED_DIR / "user_features.parquet",
        GOLD_DIR / "user_features.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "segment",
) -> tuple[pd.DataFrame, pd.Series, list[str], LabelEncoder | None]:
    """Prépare X (variables explicatives numériques sans ID/PCA) et y (cible)."""
    work = df.copy()

    if target_col in work.columns:
        y_raw = work[target_col]
    elif "organic_ratio" in work.columns:
        med = pd.to_numeric(work["organic_ratio"], errors="coerce").median()
        med = 0.0 if pd.isna(med) else float(med)
        y_raw = (pd.to_numeric(work["organic_ratio"], errors="coerce").fillna(0.0) >= med).astype(int)
        target_col = "high_organic_target"
        logging.info("Colonne 'segment' absente : cible binaire '%s' créée (seuil=%.4f).", target_col, med)
    else:
        raise ValueError("Aucune cible exploitable ('segment' ou 'organic_ratio') trouvée dans les données.")

    feature_cols = [
        col
        for col in work.select_dtypes(include=[np.number]).columns
        if col not in EXCLUDED_FEATURE_COLS and col != target_col
    ]
    if not feature_cols:
        raise ValueError("Aucune variable numérique explicative disponible pour la classification.")

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    le: LabelEncoder | None = None
    if y_raw.dtype == object or str(y_raw.dtype).startswith("string") or str(y_raw.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=work.index)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)

    return X, y, feature_cols, le


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Sauvegarde la matrice de confusion du meilleur modèle."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax)

    ax.set_title("Matrice de confusion — Classification Panier Vert")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j]):,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(target_col: str = "segment") -> Path | None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_dataset()
    if input_path is None:
        logging.error("Aucun fichier d'entrée trouvé (segments.parquet ou user_features.parquet).")
        return None

    logging.info("Chargement des données depuis %s...", input_path)
    df = pd.read_parquet(input_path)
    if len(df) < 10:
        logging.warning("Jeu de données trop petit (%d lignes) pour entraîner un classifieur.", len(df))
        return None

    X, y, feature_cols, label_encoder = prepare_features_and_target(df, target_col=target_col)
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
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob" if len(class_counts) > 2 else "binary:logistic",
            eval_metric="mlogloss" if len(class_counts) > 2 else "logloss",
            n_jobs=-1,
            random_state=42,
        )

    best_name = ""
    best_model = None
    best_f1 = -1.0
    models_summary: dict[str, dict] = {}

    print("\n=== ÉVALUATION DES MODÈLES DE CLASSIFICATION ===")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1_w = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        f1_m = float(f1_score(y_test, preds, average="macro", zero_division=0))

        models_summary[name] = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }
        print(f"{name:20s} | accuracy={acc:.4f} | f1_weighted={f1_w:.4f} | f1_macro={f1_m:.4f}")

        if f1_w > best_f1:
            best_f1 = f1_w
            best_name = name
            best_model = model

    assert best_model is not None
    logging.info("Meilleur modèle retenu : %s (f1_weighted=%.4f)", best_name, best_f1)

    full_preds = best_model.predict(X)
    full_proba = best_model.predict_proba(X).max(axis=1) if hasattr(best_model, "predict_proba") else None

    out_df = pd.DataFrame(index=df.index)
    if "user_id" in df.columns:
        out_df["user_id"] = df["user_id"]
    out_df["predicted_class"] = full_preds
    if label_encoder is not None:
        out_df["predicted_label"] = label_encoder.inverse_transform(full_preds)
    if full_proba is not None:
        out_df["prediction_confidence"] = np.round(full_proba.astype(np.float64), 4)

    preds_path = GOLD_DIR / "classification_predictions.parquet"
    out_df.to_parquet(preds_path, index=False)
    logging.info("Prédictions sauvegardées : %s", preds_path)

    test_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, test_preds)
    unique_labels = sorted(np.unique(np.concatenate([y_test.to_numpy(), test_preds])))
    if label_encoder is not None:
        display_names = [str(label_encoder.inverse_transform([int(lbl)])[0]) for lbl in unique_labels]
    else:
        display_names = [f"Classe {int(lbl)}" for lbl in unique_labels]

    cm_path = MODELS_DIR / "confusion_matrix.png"
    save_confusion_matrix_plot(cm, display_names, cm_path)
    logging.info("Matrice de confusion sauvegardée : %s", cm_path)

    feature_importances: dict[str, float] = {}
    final_step = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(final_step, "feature_importances_"):
        feature_importances = {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(feature_cols, final_step.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    report_payload = {
        "source_file": str(input_path.relative_to(PROJECT_ROOT)),
        "target_column": target_col,
        "features_used": feature_cols,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "feature_importances": feature_importances,
        "models": models_summary,
    }
    report_path = MODELS_DIR / "classification_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Rapport de classification sauvegardé : %s", report_path)

    # Sauvegarde du meilleur modèle
    model_path = MODELS_DIR / "champion.joblib"
    dump({
        "model": best_model,
        "features": feature_cols,
        "label_encoder": label_encoder,
    }, model_path)
    logging.info("Modèle champion sauvegardé : %s", model_path)

    return preds_path


if __name__ == "__main__":
    main()