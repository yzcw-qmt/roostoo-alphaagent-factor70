from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSignal:
    pair: str
    factor_value: float
    rank: int
    selected: bool
    price: float

    @property
    def action(self) -> str:
        return "LONG" if self.selected else "HOLD"


def _closed_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily OHLCV data and remove an unfinished current candle."""
    result = frame.copy().sort_index()
    required = {"high", "low", "close"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Missing factor columns: {sorted(missing)}")
    result = result.dropna(subset=list(required))
    if len(result) > 1:
        # Binance/Roostoo timestamps are UTC. The newest daily candle may
        # still be forming; ranking on it would leak intraday information.
        now = pd.Timestamp.now(tz="UTC")
        if result.index[-1] + pd.Timedelta(days=1) > now:
            result = result.iloc[:-1]
    return result


def alphaagent_70(frame: pd.DataFrame) -> float:
    """AlphaAgent factor 70: net 20-day move / absolute 20-day move."""
    data = _closed_daily(frame).tail(21)
    if len(data) < 21:
        return float("nan")
    delta = data["close"].astype(float).diff().iloc[1:]
    denominator = float(delta.abs().sum())
    if denominator == 0:
        return 0.0
    return float(delta.sum() / denominator)


class CrossSectionalFactorStrategy:
    """Daily top-N long-only cross-sectional strategy."""

    def __init__(self, name: str = "alphaagent_70", top_n: int = 5) -> None:
        if name != "alphaagent_70":
            raise ValueError(f"Unsupported factor strategy: {name}")
        self.name = name
        self.top_n = top_n
        self.function = alphaagent_70

    @staticmethod
    def is_crypto_pair(pair: str) -> bool:
        """Exclude Roostoo stock proxies, whose symbols use a ``B`` suffix."""
        base = pair.split("/", 1)[0].upper()
        return not base.endswith("B")

    def rank(self, candles: Mapping[str, pd.DataFrame]) -> list[FactorSignal]:
        values: list[tuple[str, float, float]] = []
        for pair, frame in candles.items():
            if not self.is_crypto_pair(pair):
                continue
            value = self.function(frame)
            if np.isfinite(value):
                price = float(frame["close"].iloc[-1])
                values.append((pair, value, price))
        # Stable lexical tie-break is deliberate for the binary factor.
        values.sort(key=lambda item: (-item[1], item[0]))
        selected = {pair for pair, _, _ in values[: self.top_n]}
        return [
            FactorSignal(pair, value, index + 1, pair in selected, price)
            for index, (pair, value, price) in enumerate(values)
        ]
