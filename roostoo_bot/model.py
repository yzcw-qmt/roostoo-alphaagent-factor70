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


class PairDirectionModel:
    """Load separate XGBoost models for each traded asset.

    A shared model is retained as a fallback so dry-run remains usable before
    the final rolling models have been trained.
    """

    def __init__(self, paths: dict[str, str | Path], fallback: str | Path | None = None) -> None:
        self.models = {pair: DirectionModel(path) for pair, path in paths.items()}
        self.fallback = DirectionModel(fallback) if fallback else None

    @property
    def is_ml(self) -> bool:
        return any(model.is_ml for model in self.models.values()) or bool(self.fallback and self.fallback.is_ml)

    def predict_probability_for_pair(self, pair: str, features: pd.DataFrame) -> float:
        model = self.models.get(pair)
        if model is not None and model.is_ml:
            return model.predict_probability(features)
        if self.fallback is not None:
            return self.fallback.predict_probability(features)
        return 0.5
