from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DirectionModel


@dataclass(frozen=True)
class Extremum:
    point: int
    price: float
    confirmed_at: int
    kind: str


def _atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = candles["close"].shift(1)
    true_range = pd.concat(
        [candles["high"] - candles["low"],
         (candles["high"] - previous).abs(),
         (candles["low"] - previous).abs()], axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _online_extrema(
    candles: pd.DataFrame,
    atr: pd.Series,
    diff: pd.Series,
    fixed_n: int = 30,
    k: float = 2.0,
    risk_pct: float = 0.03,
    dynamic_n: bool = False,
) -> tuple[list[Extremum], list[Extremum]]:
    """Confirm extrema causally; no future bars or repainting are used."""
    volatility = atr / candles["close"].replace(0, np.nan)
    highs_price = candles["high"].to_numpy(dtype=float)
    lows_price = candles["low"].to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    volatility_values = volatility.to_numpy(dtype=float)
    highs: list[Extremum] = []
    lows: list[Extremum] = []
    state = "HIGH"
    current_high = -np.inf
    s_high: int | None = None
    current_low = np.inf
    s_low: int | None = None

    for t in range(len(candles)):
        if dynamic_n and volatility_values[t] == volatility_values[t] and t >= 100:
            history = pd.Series(volatility_values[t - 99:t + 1]).dropna()
            q20, q80 = history.quantile(0.20), history.quantile(0.80)
            n = 15 if volatility.iloc[t] > q80 else 30 if volatility.iloc[t] < q20 else 20
        else:
            n = fixed_n
        if state == "HIGH":
            if s_high is None:
                current_high, s_high = highs_price[t], t
            elif highs_price[t] >= current_high:
                current_high, s_high = highs_price[t], t
            if s_high is not None and t - s_high >= n:
                low_since = float(np.min(lows_price[s_high:t + 1]))
                threshold = max(k * atr_values[t], risk_pct * current_high)
                if current_high - low_since >= threshold:
                    highs.append(Extremum(s_high, current_high, t, "HIGH"))
                    state = "LOW"
                    start = s_high + 1
                    if start <= t:
                        segment = lows_price[start:t + 1]
                        s_low = start + int(segment.argmin())
                        current_low = lows_price[s_low]
                    else:
                        s_low, current_low = None, np.inf
        else:
            if s_low is None:
                current_low, s_low = lows_price[t], t
            elif lows_price[t] <= current_low:
                current_low, s_low = lows_price[t], t
            if s_low is not None and t - s_low >= n:
                high_since = float(np.max(highs_price[s_low:t + 1]))
                threshold = max(k * atr_values[t], risk_pct * current_low)
                if high_since - current_low >= threshold:
                    lows.append(Extremum(s_low, current_low, t, "LOW"))
                    state = "HIGH"
                    start = s_low + 1
                    if start <= t:
                        segment = highs_price[start:t + 1]
                        s_high = start + int(segment.argmax())
                        current_high = highs_price[s_high]
                    else:
                        s_high, current_high = None, -np.inf
    return highs, lows


@dataclass(frozen=True)
class MaMacdSignal:
    pair: str
    model_probability: float
    technical_score: float
    strength: float
    action: str
    price: float
    exit_long: bool = False
    exit_short: bool = False
    extremum_type: str = ""
    extremum_point: int | None = None
    extremum_confirmed_at: int | None = None
    score_components: dict[str, float] | None = None

    @property
    def confidence(self) -> float:
        return float(np.clip(abs(self.strength - 0.5) / 0.5, 0.0, 1.0))


class MaMacdSignalGenerator:
    """Scored MA10/MA60 + MACD strategy.

    Every rule is causal and contributes evidence instead of acting as a
    mandatory gate.  ``strength`` is a directional score in [0, 1]: 0.5 is
    neutral, higher values favour a long and lower values favour a short.
    """

    def __init__(
        self, atr_period: int = 14, n: int = 20, k: float = 1.5,
        risk_pct: float = 0.02, dynamic_n: bool = False,
        long_threshold: float = 0.62, short_threshold: float = 0.38,
    ) -> None:
        self.atr_period = atr_period
        self.n = n
        self.k = k
        self.risk_pct = risk_pct
        self.dynamic_n = dynamic_n
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate(self, pair: str, candles: pd.DataFrame, features: pd.DataFrame | None = None) -> MaMacdSignal:
        frame = candles.copy().sort_index()
        close = frame["close"].astype(float)
        ma10, ma60 = close.rolling(10).mean(), close.rolling(60).mean()
        ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        atr = _atr(frame, self.atr_period)
        highs, lows = _online_extrema(frame, atr, diff, self.n, self.k, self.risk_pct, self.dynamic_n)
        t = len(frame) - 1
        if any(pd.isna(x) for x in (ma10.iloc[t], ma60.iloc[t], diff.iloc[t], dea.iloc[t])):
            return MaMacdSignal(pair, 0.5, 0.5, 0.5, "HOLD", float(close.iloc[t]), score_components={})
        cross_up = bool(ma10.iloc[t] > ma60.iloc[t] and ma10.iloc[t - 1] <= ma60.iloc[t - 1])
        cross_down = bool(ma10.iloc[t] < ma60.iloc[t] and ma10.iloc[t - 1] >= ma60.iloc[t - 1])
        low_confirmed = next((x for x in reversed(lows) if x.confirmed_at == t), None)
        high_confirmed = next((x for x in reversed(highs) if x.confirmed_at == t), None)
        low_prior = diff.iloc[max(0, low_confirmed.point - self.n):low_confirmed.point] if low_confirmed else pd.Series(dtype=float)
        high_prior = diff.iloc[max(0, high_confirmed.point - self.n):high_confirmed.point] if high_confirmed else pd.Series(dtype=float)
        low_diff_extreme = low_confirmed is not None and not low_prior.empty and diff.iloc[low_confirmed.point] <= low_prior.min()
        high_diff_extreme = high_confirmed is not None and not high_prior.empty and diff.iloc[high_confirmed.point] >= high_prior.max()
        # Symmetric evidence score.  The previous all-or-nothing conjunction
        # made a valid signal rare; here each confirmed rule adds evidence.
        components = {
            "ma_trend": 1.0 if ma10.iloc[t] > ma60.iloc[t] else -1.0,
            "ma_cross": 1.0 if cross_up else -1.0 if cross_down else 0.0,
            "macd_zero_axis": 1.0 if diff.iloc[t] > 0 and dea.iloc[t] > 0 else -1.0 if diff.iloc[t] < 0 and dea.iloc[t] < 0 else 0.0,
            "diff_dea": 1.0 if diff.iloc[t] > dea.iloc[t] else -1.0 if diff.iloc[t] < dea.iloc[t] else 0.0,
            "diff_low_extremum": 1.0 if low_diff_extreme else 0.0,
            "diff_high_extremum": -1.0 if high_diff_extreme else 0.0,
        }
        weights = {"ma_trend": 0.24, "ma_cross": 0.20, "macd_zero_axis": 0.22,
                   "diff_dea": 0.16, "diff_low_extremum": 0.09, "diff_high_extremum": 0.09}
        signed_evidence = sum(components[name] * weight for name, weight in weights.items())
        strength = float(np.clip(0.5 + 0.5 * signed_evidence, 0.0, 1.0))
        action = "LONG" if strength >= self.long_threshold else "SHORT" if strength <= self.short_threshold else "HOLD"
        return MaMacdSignal(
            pair, 0.5, strength, strength, action, float(close.iloc[t]),
            bool(diff.iloc[t] <= 0), bool(diff.iloc[t] >= 0),
            "LOW" if low_confirmed else "HIGH" if high_confirmed else "",
            low_confirmed.point if low_confirmed else high_confirmed.point if high_confirmed else None,
            t if low_confirmed or high_confirmed else None,
            components,
        )

    def generate_series(self, pair: str, candles: pd.DataFrame) -> list[MaMacdSignal]:
        """Generate a causal signal for every bar in one pass.

        This is used by long historical backtests.  Extrema are confirmed
        once on the complete chronological frame, but each signal only reads
        extrema whose ``confirmed_at`` is the current bar, so no future data
        is exposed to the signal at time t.
        """
        frame = candles.copy().sort_index()
        close = frame["close"].astype(float)
        ma10, ma60 = close.rolling(10).mean(), close.rolling(60).mean()
        diff = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        dea = diff.ewm(span=9, adjust=False).mean()
        atr = _atr(frame, self.atr_period)
        highs, lows = _online_extrema(frame, atr, diff, self.n, self.k, self.risk_pct, self.dynamic_n)
        highs_at = {x.confirmed_at: x for x in highs}
        lows_at = {x.confirmed_at: x for x in lows}
        signals: list[MaMacdSignal] = []
        for t in range(len(frame)):
            price = float(close.iloc[t])
            if t < 60 or any(pd.isna(x) for x in (ma10.iloc[t], ma60.iloc[t], diff.iloc[t], dea.iloc[t])):
                signals.append(MaMacdSignal(pair, 0.5, 0.5, 0.5, "HOLD", price, score_components={}))
                continue
            cross_up = bool(ma10.iloc[t] > ma60.iloc[t] and ma10.iloc[t - 1] <= ma60.iloc[t - 1])
            cross_down = bool(ma10.iloc[t] < ma60.iloc[t] and ma10.iloc[t - 1] >= ma60.iloc[t - 1])
            low = lows_at.get(t)
            high = highs_at.get(t)
            low_prior = diff.iloc[max(0, low.point - self.n):low.point] if low else pd.Series(dtype=float)
            high_prior = diff.iloc[max(0, high.point - self.n):high.point] if high else pd.Series(dtype=float)
            low_extreme = low is not None and not low_prior.empty and diff.iloc[low.point] <= low_prior.min()
            high_extreme = high is not None and not high_prior.empty and diff.iloc[high.point] >= high_prior.max()
            components = {
                "ma_trend": 1.0 if ma10.iloc[t] > ma60.iloc[t] else -1.0,
                "ma_cross": 1.0 if cross_up else -1.0 if cross_down else 0.0,
                "macd_zero_axis": 1.0 if diff.iloc[t] > 0 and dea.iloc[t] > 0 else -1.0 if diff.iloc[t] < 0 and dea.iloc[t] < 0 else 0.0,
                "diff_dea": 1.0 if diff.iloc[t] > dea.iloc[t] else -1.0 if diff.iloc[t] < dea.iloc[t] else 0.0,
                "diff_low_extremum": 1.0 if low_extreme else 0.0,
                "diff_high_extremum": -1.0 if high_extreme else 0.0,
            }
            weights = {"ma_trend": 0.24, "ma_cross": 0.20, "macd_zero_axis": 0.22,
                       "diff_dea": 0.16, "diff_low_extremum": 0.09, "diff_high_extremum": 0.09}
            score = float(np.clip(0.5 + 0.5 * sum(components[k] * weights[k] for k in weights), 0.0, 1.0))
            action = "LONG" if score >= self.long_threshold else "SHORT" if score <= self.short_threshold else "HOLD"
            signals.append(MaMacdSignal(
                pair, 0.5, score, score, action, price,
                bool(diff.iloc[t] <= 0), bool(diff.iloc[t] >= 0),
                "LOW" if low else "HIGH" if high else "",
                low.point if low else high.point if high else None,
                t if low or high else None, components,
            ))
        return signals


class MaMacdXGBSignalGenerator:
    """Blend the causal MA/MACD signal with an XGBoost probability.

    The technical score remains the primary signal and keeps all of the
    original causal exit metadata. XGBoost only adjusts direction strength;
    if no model can be loaded, ``DirectionModel`` exposes a deterministic
    fallback and ``is_ml`` stays false so callers can report that fact.
    """

    def __init__(self, technical: MaMacdSignalGenerator, model: DirectionModel, model_weight: float = 0.60) -> None:
        self.technical = technical
        self.model = model
        self.model_weight = float(np.clip(model_weight, 0.0, 1.0))

    @property
    def is_ml(self) -> bool:
        return self.model.is_ml

    def generate(self, pair: str, candles: pd.DataFrame, features: pd.DataFrame | None = None) -> MaMacdSignal:
        technical = self.technical.generate(pair, candles, features)
        probability = self.model.predict_probability(features) if features is not None else 0.5
        strength = float(
            self.model_weight * probability
            + (1.0 - self.model_weight) * technical.technical_score
        )
        action = (
            "LONG" if strength >= self.technical.long_threshold
            else "SHORT" if strength <= self.technical.short_threshold
            else "HOLD"
        )
        return MaMacdSignal(
            pair=technical.pair,
            model_probability=probability,
            technical_score=technical.technical_score,
            strength=strength,
            action=action,
            price=technical.price,
            exit_long=technical.exit_long,
            exit_short=technical.exit_short,
            extremum_type=technical.extremum_type,
            extremum_point=technical.extremum_point,
            extremum_confirmed_at=technical.extremum_confirmed_at,
            score_components=technical.score_components,
        )
