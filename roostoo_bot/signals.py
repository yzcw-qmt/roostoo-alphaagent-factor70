from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DirectionModel


@dataclass(frozen=True)
class Signal:
    pair: str
    model_probability: float
    technical_score: float
    strength: float
    action: str
    price: float

    @property
    def confidence(self) -> float:
        return float(np.clip(abs(self.strength - 0.5) / 0.5, 0.0, 1.0))


def technical_score(features: pd.DataFrame) -> float:
    row = features.dropna().tail(1)
    if row.empty:
        return 0.5
    x = row.iloc[0]
    rsi_reversal = float(np.clip(1.0 - x["rsi"], 0, 1))
    macd = float(1 / (1 + np.exp(-120 * x["ema_ratio"])))
    bollinger = float(np.clip(1.0 - x["donchian_position"], 0, 1))
    direction = np.sign(x["ema_ratio"])
    volume_confirmation = float(np.clip(0.5 + direction * (x["volume_ratio"] - 1) * 0.25, 0, 1))
    trend = float(np.mean([x["trend_4h"] > 0, x["trend_24h"] > 0]))
    return float(0.20 * rsi_reversal + 0.25 * macd + 0.15 * bollinger + 0.20 * volume_confirmation + 0.20 * trend)


class SignalGenerator:
    def __init__(
        self,
        model: DirectionModel,
        long_threshold: float = 0.65,
        short_threshold: float = 0.35,
        model_weight: float = 0.60,
    ) -> None:
        self.model = model
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.model_weight = model_weight

    def generate(self, pair: str, candles: pd.DataFrame, features: pd.DataFrame) -> Signal:
        probability = self.model.predict_probability(features)
        technical = technical_score(features)
        strength = self.model_weight * probability + (1 - self.model_weight) * technical
        action = "LONG" if strength > self.long_threshold else "SHORT" if strength < self.short_threshold else "HOLD"
        return Signal(pair, probability, technical, float(strength), action, float(candles["close"].iloc[-1]))
