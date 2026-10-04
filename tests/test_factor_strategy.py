import numpy as np
import pandas as pd

from roostoo_bot.factor_strategy import CrossSectionalFactorStrategy, gplearn_1852


def daily_frame(periods: int = 90, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), periods=periods, freq="D")
    close = np.linspace(start, start * 1.2, periods)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index)


def test_factors_require_closed_history() -> None:
    frame = daily_frame()
    assert gplearn_1852(frame) in {0.0, 1.0}


def test_cross_sectional_strategy_selects_top_n_long_only() -> None:
    frames = {f"C{i}/USD": daily_frame(start=100 + i) for i in range(6)}
    signals = CrossSectionalFactorStrategy("gplearn_1852", top_n=5).rank(frames)
    assert len(signals) == 6
    assert sum(signal.selected for signal in signals) == 5
    assert all(signal.action in {"LONG", "HOLD"} for signal in signals)
