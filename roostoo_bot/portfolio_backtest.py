from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentResult:
    start: str
    end: str
    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    composite_score: float
    turnover: float
    rebalance_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def random_non_overlapping_segments(
    start: pd.Timestamp,
    end: pd.Timestamp,
    segment_days: int = 15,
    count: int = 16,
    seed: int = 42,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    rng = np.random.default_rng(seed)
    offset_days = int(rng.integers(0, segment_days))
    segment_delta = timedelta(days=int(segment_days))
    first = start.normalize() + timedelta(days=offset_days)
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first
    while cursor + segment_delta <= end:
        candidates.append((cursor, cursor + segment_delta))
        cursor += segment_delta
    if not candidates:
        raise ValueError("Date range is shorter than one segment")
    selected = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return sorted((candidates[int(index)] for index in selected), key=lambda item: item[0])


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def build_portfolio_weights(closes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create point-in-time weights; every signal uses data available at time t only."""
    closes = closes.sort_index().ffill()
    returns = closes.pct_change()
    log_prices = np.log(closes)
    volatility = returns.rolling(72).std().replace(0, np.nan)
    momentum_24 = closes.pct_change(24)
    momentum_72 = closes.pct_change(72)
    trend_strength = momentum_72 / (volatility * math.sqrt(72))
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    btc = "BTC/USD"
    eth = "ETH/USD"
    diagnostics: dict[str, float | str] = {}
    if btc not in closes:
        raise ValueError("BTC/USD is required for regime and cross-asset signals")

    btc_z = _rolling_zscore(log_prices[btc], 72)
    btc_signal = pd.Series(0.0, index=closes.index)

    # BTC mean reversion uses only its rolling distribution. There are no
    # hard-coded absolute price levels, so the rule adapts across regimes.
    btc_signal = btc_signal.mask(btc_z > 2.0, -1.0).mask(btc_z < -2.0, 1.0)

    # Strong market regime: take one direction; do not force a long-short hedge.
    strong_up = trend_strength[btc] > 2.25
    strong_down = trend_strength[btc] < -2.25
    for pair in closes.columns:
        own_trend = np.sign(momentum_24[pair]).fillna(0)
        directional = pd.Series(0.0, index=closes.index)
        directional = directional.mask(strong_up & (own_trend >= 0), 1.0)
        directional = directional.mask(strong_down & (own_trend <= 0), -1.0)
        weights[pair] = 0.15 * directional
    weights[btc] = weights[btc].where(strong_up | strong_down, 0.20 * btc_signal)

    if eth in closes:
        correlation = returns[btc].rolling(168).corr(returns[eth])
        same_trend = np.sign(momentum_24[btc]) == np.sign(momentum_24[eth])
        relative = log_prices[eth] - log_prices[btc]
        spread_z = _rolling_zscore(relative, 168)
        hedge_regime = (correlation > 0.70) & same_trend & ~(strong_up | strong_down) & (spread_z.abs() > 2.0)
        # ETH rich versus BTC: long BTC / short ETH; reverse when ETH is cheap.
        eth_rich = spread_z > 0
        weights.loc[hedge_regime & eth_rich, btc] = 0.25
        weights.loc[hedge_regime & eth_rich, eth] = -0.25
        weights.loc[hedge_regime & ~eth_rich, btc] = -0.25
        weights.loc[hedge_regime & ~eth_rich, eth] = 0.25

        # Lead-lag: yesterday's/previous hour's leader return can tilt the follower.
        btc_leads_eth = returns[btc].shift(1).rolling(168).corr(returns[eth])
        eth_leads_btc = returns[eth].shift(1).rolling(168).corr(returns[btc])
        calm = ~(strong_up | strong_down | hedge_regime)
        btc_impulse = returns[btc].rolling(3).sum() / (volatility[btc] * math.sqrt(3))
        eth_impulse = returns[eth].rolling(3).sum() / (volatility[eth] * math.sqrt(3))
        btc_lead_signal = calm & (btc_leads_eth > 0.20) & (btc_impulse.abs() > 1.5)
        eth_lead_signal = calm & (eth_leads_btc > 0.20) & (eth_impulse.abs() > 1.5)
        weights.loc[btc_lead_signal, eth] += 0.15 * np.sign(btc_impulse.loc[btc_lead_signal])
        weights.loc[eth_lead_signal, btc] += 0.15 * np.sign(eth_impulse.loc[eth_lead_signal])
        diagnostics = {
            "btc_eth_hourly_correlation": float(returns[btc].corr(returns[eth])),
            "btc_leads_eth_correlation": float(returns[btc].shift(1).corr(returns[eth])),
            "eth_leads_btc_correlation": float(returns[eth].shift(1).corr(returns[btc])),
        }

    # Give non-BTC assets a small independent trend allocation when no market regime is active.
    calm_market = ~(strong_up | strong_down)
    for pair in closes.columns:
        if pair not in {btc, eth}:
            independent = np.sign(trend_strength[pair]).where(trend_strength[pair].abs() > 2.0, 0)
            weights.loc[calm_market, pair] = 0.15 * independent.loc[calm_market]

    # Portfolio limits: 30% per asset, <=90% gross and <=60% net exposure.
    weights = weights.clip(-0.30, 0.30).fillna(0.0)
    weights = weights.mask(weights.abs() < 0.12, 0.0)
    gross = weights.abs().sum(axis=1)
    gross_scale = (0.90 / gross).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(gross_scale, axis=0)
    net = weights.sum(axis=1).abs()
    net_scale = (0.60 / net).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(net_scale, axis=0)
    # Rebalance only every twelve hours and ignore small portfolio changes.
    # Exits still pass the buffer because closing an established strong position
    # changes gross exposure by at least 15%.
    rebalance_mask = pd.Series(weights.index.hour % 12 == 0, index=weights.index)
    candidates = weights.where(rebalance_mask, np.nan)
    buffered = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    current = pd.Series(0.0, index=weights.columns)
    for timestamp in weights.index:
        candidate = candidates.loc[timestamp]
        if candidate.notna().all() and float((candidate - current).abs().sum()) >= 0.15:
            current = candidate.copy()
        buffered.loc[timestamp] = current
    return buffered, diagnostics


def simulate_segment(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_equity: float = 100000.0,
    fee_rate: float = 0.001,
) -> SegmentResult:
    segment_closes = closes.loc[(closes.index >= start) & (closes.index < end)]
    segment_weights = weights.reindex(segment_closes.index).fillna(0.0)
    if len(segment_closes) < 24:
        raise ValueError(f"Not enough observations in segment starting {start}")
    asset_returns = segment_closes.pct_change().fillna(0.0)
    held = segment_weights.shift(1).fillna(0.0)
    gross_return = (held * asset_returns).sum(axis=1)
    previous = segment_weights.shift(1).fillna(0.0)
    turnover_series = (segment_weights - previous).abs().sum(axis=1)
    net_return = gross_return - turnover_series * fee_rate
    equity = initial_equity * (1 + net_return).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    downside = net_return[net_return < 0]
    periods_per_year = 24 * 365
    std = float(net_return.std())
    downside_std = float(downside.std()) if len(downside) else 0.0
    sharpe = float(net_return.mean() / std * math.sqrt(periods_per_year)) if std else 0.0
    sortino = float(net_return.mean() / downside_std * math.sqrt(periods_per_year)) if downside_std else 0.0
    total_return = float(equity.iloc[-1] / initial_equity - 1)
    years = len(net_return) / periods_per_year
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    return SegmentResult(
        start=str(start),
        end=str(end),
        initial_equity=initial_equity,
        final_equity=float(equity.iloc[-1]),
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        composite_score=float(composite),
        turnover=float(turnover_series.sum()),
        rebalance_count=int((turnover_series > 1e-12).sum()),
    )
