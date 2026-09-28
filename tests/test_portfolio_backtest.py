import numpy as np
import pandas as pd

from roostoo_bot.portfolio_backtest import (
    build_portfolio_weights,
    random_non_overlapping_segments,
    simulate_segment,
)


def sample_closes(periods: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    market = rng.normal(0.0001, 0.006, periods)
    return pd.DataFrame(
        {
            "BTC/USD": 80000 * np.exp(np.cumsum(market)),
            "ETH/USD": 3000 * np.exp(np.cumsum(0.8 * market + rng.normal(0, 0.003, periods))),
            "BNB/USD": 600 * np.exp(np.cumsum(0.4 * market + rng.normal(0, 0.004, periods))),
            "TON/USD": 3 * np.exp(np.cumsum(0.5 * market + rng.normal(0, 0.006, periods))),
            "TRX/USD": 0.3 * np.exp(np.cumsum(0.3 * market + rng.normal(0, 0.003, periods))),
        },
        index=index,
    )


def test_random_segments_are_reproducible_and_non_overlapping() -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    first = random_non_overlapping_segments(start, end, count=16, seed=42)
    second = random_non_overlapping_segments(start, end, count=16, seed=42)
    assert first == second
    assert len(first) == 16
    for (_, left_end), (right_start, _) in zip(first, first[1:]):
        assert left_end <= right_start


def test_weights_obey_portfolio_limits() -> None:
    weights, diagnostics = build_portfolio_weights(sample_closes())
    assert (weights.abs() <= 0.300001).all().all()
    assert (weights.abs().sum(axis=1) <= 0.900001).all()
    assert (weights.sum(axis=1).abs() <= 0.600001).all()
    assert "btc_eth_hourly_correlation" in diagnostics


def test_segment_resets_initial_capital() -> None:
    closes = sample_closes()
    weights, _ = build_portfolio_weights(closes)
    result = simulate_segment(
        closes,
        weights,
        pd.Timestamp("2025-01-20", tz="UTC"),
        pd.Timestamp("2025-02-04", tz="UTC"),
    )
    assert result.initial_equity == 100000
    assert result.final_equity > 0
    assert result.max_drawdown <= 0
