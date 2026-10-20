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
    daily_gain: float = 0.0
    blocked: str = ""

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


def alphaagent_70_live(frame: pd.DataFrame, price: float) -> float:
    """Factor 70 with the live price standing in for the latest close.

    Replacing the last point of the 21-close window keeps the live score on
    the same time window as :func:`alphaagent_70`. At the daily rebalance
    boundary, passing the latest completed close reproduces the stored daily
    baseline exactly; during the day, only that endpoint moves.
    """
    data = _closed_daily(frame).tail(21)
    if len(data) < 21 or not np.isfinite(price):
        return float("nan")
    closes = list(data["close"].astype(float))
    closes[-1] = float(price)
    delta = pd.Series(closes).diff().iloc[1:]
    denominator = float(delta.abs().sum())
    if denominator == 0:
        return 0.0
    return float(delta.sum() / denominator)


class CrossSectionalFactorStrategy:
    """Daily top-N long-only cross-sectional strategy.

    Three filters sit on top of the raw ranking:

    ``min_score``
        A pair is only eligible when its factor is at or above this value.
        Rank alone is not a signal: on quiet days the third-ranked name can
        still be weak, and buying it loses money.

    ``max_daily_gain``
        A pair whose latest closed daily candle gained more than this is
        excluded. A single parabolic day inflates Factor 70 (the numerator
        jumps while the path length grows less), so an un-filtered ranking
        chases exactly the names that mean-revert hardest afterwards.

    Both filters only change *eligibility*; the score itself is unchanged so
    it stays comparable across days for the exit rule in the engine.
    """

    def __init__(
        self,
        name: str = "alphaagent_70",
        top_n: int = 3,
        min_score: float = 0.0,
        max_daily_gain: float = 0.0,
    ) -> None:
        if name != "alphaagent_70":
            raise ValueError(f"Unsupported factor strategy: {name}")
        self.name = name
        self.top_n = top_n
        self.min_score = float(min_score)
        self.max_daily_gain = float(max_daily_gain)
        self.function = alphaagent_70

    def rank(self, candles: Mapping[str, pd.DataFrame]) -> list[FactorSignal]:
        values: list[tuple[str, float, float, float]] = []
        for pair, frame in candles.items():
            value = self.function(frame)
            if not np.isfinite(value):
                continue
            closed = _closed_daily(frame)
            closes = closed["close"].astype(float)
            price = float(closes.iloc[-1])
            gain = float(closes.iloc[-1] / closes.iloc[-2] - 1.0) if len(closes) > 1 else 0.0
            values.append((pair, value, price, gain))
        # Stable lexical tie-break is deliberate for the binary factor.
        values.sort(key=lambda item: (-item[1], item[0]))

        blocked: dict[str, str] = {}
        selected: list[str] = []
        for pair, value, _, gain in values:
            reason = ""
            if self.max_daily_gain > 0 and gain > self.max_daily_gain:
                reason = "daily_gain_spike"
            elif value < self.min_score:
                reason = "below_min_score"
            blocked[pair] = reason
            if not reason and len(selected) < self.top_n:
                selected.append(pair)

        chosen = set(selected)
        return [
            FactorSignal(pair, value, index + 1, pair in chosen, price, gain, blocked[pair])
            for index, (pair, value, price, gain) in enumerate(values)
        ]
