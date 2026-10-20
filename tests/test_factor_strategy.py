import numpy as np
import pandas as pd

from roostoo_bot.factor_strategy import CrossSectionalFactorStrategy, alphaagent_70


def daily_frame(periods: int = 90, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), periods=periods, freq="D")
    close = np.linspace(start, start * 1.2, periods)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index)


def test_factors_require_closed_history() -> None:
    frame = daily_frame()
    assert -1.0 <= alphaagent_70(frame) <= 1.0


def test_cross_sectional_strategy_selects_top_n_long_only() -> None:
    frames = {f"C{i}/USD": daily_frame(start=100 + i) for i in range(6)}
    frames["NVDAB/USD"] = daily_frame(start=1000)
    signals = CrossSectionalFactorStrategy("alphaagent_70", top_n=5).rank(frames)
    assert len(signals) == 7
    assert sum(signal.selected for signal in signals) == 5
    assert all(signal.action in {"LONG", "HOLD"} for signal in signals)


def spike_frame(start: float = 100.0, jump: float = 1.0) -> pd.DataFrame:
    """Steady uptrend with one parabolic final day."""
    periods = 90
    index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), periods=periods, freq="D"
    )
    close = np.linspace(start, start * 1.2, periods - 1)
    close = np.append(close, close[-1] * (1 + jump))
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index)


def test_daily_gain_spike_is_excluded_and_slot_goes_to_next_ranked() -> None:
    frames = {f"C{i}/USD": daily_frame(start=100 + i) for i in range(6)}
    frames["SPIKE/USD"] = spike_frame(jump=1.0)
    strategy = CrossSectionalFactorStrategy("alphaagent_70", top_n=3, max_daily_gain=0.30)
    signals = strategy.rank(frames)
    by_pair = {signal.pair: signal for signal in signals}
    # A one-way move scores a perfect 1.0, yet the spike day makes it unsafe.
    assert by_pair["SPIKE/USD"].factor_value == 1.0
    assert by_pair["SPIKE/USD"].selected is False
    assert by_pair["SPIKE/USD"].blocked == "daily_gain_spike"
    assert sum(signal.selected for signal in signals) == 3


def test_min_score_blocks_weak_names_instead_of_forcing_top_n() -> None:
    # A flat series has a near-zero factor: it must not be bought just to
    # fill a slot.
    index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), periods=90, freq="D"
    )
    flat = pd.Series(100.0, index=index)
    frames = {
        "A/USD": daily_frame(start=100, periods=90),
        "B/USD": pd.DataFrame({"open": flat, "high": flat + 1, "low": flat - 1, "close": flat}),
    }
    signals = CrossSectionalFactorStrategy("alphaagent_70", top_n=3, min_score=0.15).rank(frames)
    by_pair = {signal.pair: signal for signal in signals}
    assert by_pair["A/USD"].selected is True
    assert by_pair["B/USD"].selected is False
    assert by_pair["B/USD"].blocked == "below_min_score"


def test_empty_basket_when_nothing_clears_the_threshold() -> None:
    index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), periods=90, freq="D"
    )
    flat = pd.Series(100.0, index=index)
    frames = {"B/USD": pd.DataFrame({"open": flat, "high": flat + 1, "low": flat - 1, "close": flat})}
    signals = CrossSectionalFactorStrategy("alphaagent_70", top_n=3, min_score=0.15).rank(frames)
    assert sum(signal.selected for signal in signals) == 0
