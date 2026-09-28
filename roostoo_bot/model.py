from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


class DirectionModel:
    """XGBoost inference with a deterministic momentum fallback."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.log = logging.getLogger(__name__)
        if self.model_path.exists():
            try:
                from xgboost import XGBClassifier

                model = XGBClassifier()
                model.load_model(self.model_path)
                self.model = model
                self.log.info("Loaded XGBoost model from %s", self.model_path)
            except Exception:
                self.log.exception("Could not load XGBoost model; using fallback")

    @property
    def is_ml(self) -> bool:
        return self.model is not None

    def predict_probability(self, features: pd.DataFrame) -> float:
        row = features[FEATURE_COLUMNS].dropna().tail(1)
        if row.empty:
            return 0.5
        if self.model is not None:
            return float(self.model.predict_proba(row)[0, 1])
        latest = row.iloc[0]
        raw = (
            35 * latest["ema_ratio"]
            + 20 * latest["sma_ratio"]
            + 4 * latest["return_6h"]
            + 2 * latest["trend_24h"]
            - 0.4 * latest["volatility"]
        )
        return float(1 / (1 + np.exp(-np.clip(raw, -8, 8))))

