from __future__ import annotations

from pathlib import Path

import bentoml
import joblib
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "champion.joblib"


class SegmentInput(BaseModel):
    total_orders: float = Field(ge=0)
    avg_basket_size: float = Field(ge=0)
    days_since_prior_order_mean: float = Field(ge=0)
    reorder_rate: float = Field(ge=0, le=1)
    organic_ratio: float = Field(ge=0, le=1)


@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 30})
class SegmentClassifier:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle champion introuvable : {MODEL_PATH}. Lance src/mining/classification.py."
            )
        payload = joblib.load(MODEL_PATH)
        self.model = payload["model"]
        self.features = payload["features"]
        self.label_encoder = payload["label_encoder"]

    @bentoml.api
    def predict(self, request: SegmentInput) -> dict:
        row = request.model_dump()
        X = pd.DataFrame([{feature: row[feature] for feature in self.features}])
        predicted = int(self.model.predict(X)[0])

        label = str(predicted)
        if self.label_encoder is not None:
            label = str(self.label_encoder.inverse_transform([predicted])[0])

        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = round(float(self.model.predict_proba(X).max()), 4)

        return {
            "predicted_segment": predicted,
            "predicted_label": label,
            "prediction_confidence": confidence,
        }
