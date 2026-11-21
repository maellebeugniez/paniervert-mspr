from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> dict:
    input_path = PROCESSED_DIR / "gold" / "user_features.parquet"
    if not input_path.exists():
        input_path = PROCESSED_DIR / "user_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError("Fichier user_features introuvable")

    df = pd.read_parquet(input_path)
    hypotheses: list[dict] = []

    if "organic_ratio" in df.columns and "avg_basket_size" in df.columns:
        median = float(df["organic_ratio"].median())
        low_bio = df[df["organic_ratio"] < median]
        high_bio = df[df["organic_ratio"] >= median]
        low_mean = float(low_bio["avg_basket_size"].mean())
        high_mean = float(high_bio["avg_basket_size"].mean())
        hypotheses.append({
            "id": "H1",
            "hypothesis": "Les clients à proportion bio élevée ont un panier moyen plus élevé.",
            "metric": "avg_basket_size",
            "low_group_mean": round(low_mean, 3),
            "high_group_mean": round(high_mean, 3),
            "status": "confirmed" if high_mean > low_mean else "not_confirmed",
        })

    if "total_orders" in df.columns and "reorder_rate" in df.columns:
        median = float(df["total_orders"].median())
        low = df[df["total_orders"] < median]
        high = df[df["total_orders"] >= median]
        low_reorder = float(low["reorder_rate"].mean())
        high_reorder = float(high["reorder_rate"].mean())
        hypotheses.append({
            "id": "H2",
            "hypothesis": "Les clients fréquents ont un taux de réachat supérieur.",
            "metric": "reorder_rate",
            "low_group_mean": round(low_reorder, 4),
            "high_group_mean": round(high_reorder, 4),
            "status": "confirmed" if high_reorder > low_reorder else "not_confirmed",
        })

    if "total_orders" in df.columns and "avg_basket_size" in df.columns:
        correlation = float(df[["total_orders", "avg_basket_size"]].corr().iloc[0, 1])
        hypotheses.append({
            "id": "H3",
            "hypothesis": "La fréquence d’achat et la taille du panier sont liées.",
            "metric": "correlation(total_orders, avg_basket_size)",
            "value": round(correlation, 4),
            "status": "confirmed" if correlation > 0.10 else "not_confirmed",
        })

    report = {
        "source_file": str(input_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "hypotheses": hypotheses,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "eda_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
