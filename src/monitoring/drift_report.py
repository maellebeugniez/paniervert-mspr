from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> dict:
    input_path = PROCESSED_DIR / "gold" / "user_features.parquet"
    if not input_path.exists():
        input_path = PROCESSED_DIR / "user_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError("Fichier user_features introuvable")

    df = pd.read_parquet(input_path)
    numeric = df.select_dtypes(include="number").columns.tolist()
    exclude = {"user_id", "segment"}
    features = [c for c in numeric if c not in exclude]

    midpoint = len(df) // 2
    reference = df.iloc[:midpoint][features]
    current = df.iloc[midpoint:][features]

    if reference.empty or current.empty:
        raise ValueError("Échantillons de référence et courant trop petits")

    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        html_path = MODELS_DIR / "evidently_drift_report.html"
        json_path = MODELS_DIR / "evidently_drift_report.json"
        report.save_html(str(html_path))
        report.save_json(str(json_path))
        status = "generated"
    except Exception as exc:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        html_path = MODELS_DIR / "evidently_drift_report.html"
        json_path = MODELS_DIR / "evidently_drift_report.json"
        html_path.write_text(
            "<html><body><h1>Rapport Evidently</h1><p>Génération indisponible :</p><pre>"
            + str(exc)
            + "</pre></body></html>",
            encoding="utf-8",
        )
        status = "fallback"

    payload = {
        "status": status,
        "source_file": str(input_path.relative_to(PROJECT_ROOT)),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": features,
        "html_report": str(html_path.relative_to(PROJECT_ROOT)),
        "json_report": str(json_path.relative_to(PROJECT_ROOT)),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
