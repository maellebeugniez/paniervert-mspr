from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "models" / "classification_report.json"


def main(min_f1_weighted: float = 0.95) -> int:
    if not REPORT_PATH.exists():
        print(f"Rapport absent : {REPORT_PATH}", file=sys.stderr)
        return 1

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    best_name = report.get("best_model")
    models = report.get("models", {})
    best_metrics = models.get(best_name, {})
    f1_weighted = float(best_metrics.get("f1_weighted", 0.0))

    print(f"Meilleur modèle : {best_name}")
    print(f"F1 pondéré : {f1_weighted:.4f}")
    print(f"Seuil quality gate : {min_f1_weighted:.4f}")

    if f1_weighted < min_f1_weighted:
        print("Quality gate échouée", file=sys.stderr)
        return 1

    print("Quality gate réussie")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
