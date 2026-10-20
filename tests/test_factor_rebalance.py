from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from roostoo_bot.config import Settings
from roostoo_bot.engine import TradingEngine
from roostoo_bot.factor_strategy import FactorSignal, alphaagent_70, alphaagent_70_live


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def buy_limit(self, pair, quantity, price):
        self.calls.append(("buy", pair, quantity, price))

    def sell_limit(self, pair, quantity, price):
        self.calls.append(("sell", pair, quantity, price))


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    settings = replace(
        Settings(),
        state_path=tmp_path / "state.json",
        log_dir=tmp_path / "logs",
        dry_run=True,
    )
    eng = TradingEngine(settings)
    eng.executor = FakeExecutor()
    # Pin the clock to a rebalance window so the schedule guard passes.
    fixed = datetime(2026, 10, 20, 0, 5, tzinfo=timezone.utc)
    monkeypatch.setattr("roostoo_bot.engine.datetime", _FixedClock(fixed))
    return eng


class _FixedClock(datetime):
    _fixed: datetime

    def __new__(cls, fixed):
        return datetime.__new__(cls, fixed.year, fixed.month, fixed.day, fixed.hour, fixed.minute, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 20, 0, 5, tzinfo=timezone.utc).astimezone(tz) if tz else datetime(2026, 10, 20, 0, 5)


RULES = {
    "AAA/USD": {"CanTrade": True, "AmountPrecision": 6, "PricePrecision": 2, "MiniOrder": 1.0},
    "BBB/USD": {"CanTrade": True, "AmountPrecision": 6, "PricePrecision": 2, "MiniOrder": 1.0},
    "OLD/USD": {"CanTrade": True, "AmountPrecision": 6, "PricePrecision": 2, "MiniOrder": 1.0},
}
TICKERS = {
    "AAA/USD": {"LastPrice": 100.0, "MaxBid": 99.9, "MinAsk": 100.1},
    "BBB/USD": {"LastPrice": 50.0, "MaxBid": 49.9, "MinAsk": 50.1},
    "OLD/USD": {"LastPrice": 10.0, "MaxBid": 9.99, "MinAsk": 10.01},
}


def signal(pair, value, rank, selected, blocked=""):
    return FactorSignal(pair, value, rank, selected, TICKERS[pair]["LastPrice"], 0.0, blocked)


def test_empty_basket_liquidates_everything(engine):
    wallet = {
        "USD": {"Free": 0.0, "Lock": 0.0},
        "OLD": {"Free": 10.0, "Lock": 0.0},
    }
    signals = [signal("OLD/USD", -0.5, 1, False, "below_min_score")]
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    kinds = [call[0] for call in engine.executor.calls]
    assert kinds == ["sell"]
    assert engine.bot_state["factor_rebalance"]["selected"] == []
    # A pending Maker sell means the day is not finished yet.
    assert engine.bot_state["factor_rebalance"]["retry"] is True


def test_blocked_spike_is_not_bought_and_slot_stays_cash(engine):
    wallet = {"USD": {"Free": 100_000.0, "Lock": 0.0}}
    signals = [
        signal("AAA/USD", -0.2, 1, False, "daily_gain_spike"),
        signal("BBB/USD", -0.3, 2, False, "below_min_score"),
        signal("OLD/USD", -0.4, 3, False, "below_min_score"),
    ]
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    assert engine.executor.calls == []
    assert engine.bot_state["factor_rebalance"]["selected"] == []
    assert engine.bot_state["factor_rebalance"]["retry"] is False


def test_underfunded_target_retries_instead_of_marking_done(engine):
    # 100 USD equity split over 3 names is far below the 1.0 minimum once we
    # cap each buy by the free balance, so the buys cannot all be placed.
    wallet = {"USD": {"Free": 2.5, "Lock": 0.0}}
    signals = [
        signal("AAA/USD", 0.9, 1, True),
        signal("BBB/USD", 0.8, 2, True),
        signal("OLD/USD", 0.7, 3, True),
    ]
    engine.bot_state["factor_rebalance"] = {"date": "2026-10-19"}
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    state = engine.bot_state["factor_rebalance"]
    assert state["retry"] is True
    assert state["date"] == "2026-10-19"


def test_exit_rule_drops_held_name_and_refills_from_pool(engine):
    engine.settings = replace(engine.settings, factor_exit_rel_drop=0.45)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": 0.9}}
    wallet = {
        "USD": {"Free": 0.0, "Lock": 0.0},
        "AAA": {"Free": 100.0, "Lock": 0.0},
    }
    signals = [
        signal("AAA/USD", 0.4, 1, True),
        signal("BBB/USD", 0.5, 2, True),
    ]
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    kinds = [call[:2] for call in engine.executor.calls]
    assert ("sell", "AAA/USD") in kinds


def test_forced_exit_is_not_refilled_by_the_pool(engine):
    # AAA collapses outright and is also still the highest raw score in the
    # pool.  The refill loop must not hand the slot back to it.
    engine.settings = replace(engine.settings, factor_exit_rel_drop=0.45)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": 0.9}}
    wallet = {
        "USD": {"Free": 0.0, "Lock": 0.0},
        "AAA": {"Free": 100.0, "Lock": 0.0},
    }
    signals = [signal("AAA/USD", 0.4, 1, True)]
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    assert engine.bot_state["factor_rebalance"]["selected"] == []
    assert [call[:2] for call in engine.executor.calls] == [("sell", "AAA/USD")]


def test_holding_without_previous_score_falls_back_to_entry_filters(engine):
    # After a state reset there is no baseline to compare against. A name we
    # would not buy today must not be carried on faith.
    engine.settings = replace(engine.settings, factor_exit_rel_drop=0.45)
    wallet = {
        "USD": {"Free": 0.0, "Lock": 0.0},
        "OLD": {"Free": 10.0, "Lock": 0.0},
    }
    signals = [signal("OLD/USD", -0.5, 1, False, "below_min_score")]
    engine._factor_rebalance(signals, wallet, RULES, TICKERS, 100_000.0, set())
    assert [call[:2] for call in engine.executor.calls] == [("sell", "OLD/USD")]


def test_holding_that_still_passes_filters_survives_a_low_rank(engine):
    # Sticky book: slipping down the ranking is not by itself a sell signal.
    engine.settings = replace(engine.settings, factor_exit_rel_drop=0.45)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": 0.4}}
    wallet = {
        "USD": {"Free": 0.0, "Lock": 0.0},
        "AAA": {"Free": 100.0, "Lock": 0.0},
    }
    signals = [
        signal("BBB/USD", 0.9, 1, True),
        signal("AAA/USD", 0.35, 2, False),
    ]
    targets, forced = engine._factor_targets(signals, engine.bot_state["factor_rebalance"], {"AAA/USD"})
    assert forced == set()
    assert [item.pair for item in targets] == ["AAA/USD", "BBB/USD"]


def _rising_frame(days: int = 30, step: float = 1.0, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=days, freq="1D", tz="UTC")
    closes = [start + step * i for i in range(days)]
    return pd.DataFrame(
        {"high": closes, "low": closes, "close": closes},
        index=index,
    )


def test_live_factor_matches_daily_factor_at_the_close():
    # Feeding the last close as the "live" price must reproduce the daily
    # score exactly, otherwise the intraday baseline is not comparable.
    frame = _rising_frame()
    daily = alphaagent_70(frame)
    live = alphaagent_70_live(frame, float(frame["close"].iloc[-1]))
    assert live == pytest.approx(daily, rel=1e-9)


def test_live_factor_drops_when_price_reverses_intraday():
    frame = _rising_frame()
    baseline = alphaagent_70(frame)
    crash = alphaagent_70_live(frame, float(frame["close"].iloc[-1]) * 0.85)
    assert crash < baseline - 0.2


def test_intraday_exit_sells_when_live_score_collapses(engine, monkeypatch):
    frame = _rising_frame()
    baseline = alphaagent_70(frame)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": baseline}}
    wallet = {"AAA": {"Free": 5.0, "Lock": 0.0}, "USD": {"Free": 0.0, "Lock": 0.0}}
    # Mark price far below the trend, so the live score collapses.
    tickers = {"AAA/USD": {"LastPrice": 60.0, "MaxBid": 59.9, "MinAsk": 60.1}}
    monkeypatch.setattr(engine.data, "candles", lambda *a, **k: frame)
    sold = engine._factor_intraday_exit(wallet, RULES, tickers, set())
    assert sold == ["AAA/USD"]
    assert [call[:2] for call in engine.executor.calls] == [("sell", "AAA/USD")]
    # The baseline survives on purpose: a LIMIT sell is not a fill. If the
    # order rests or is cancelled, the remaining quantity must still be
    # monitored on the next tick instead of becoming invisible.
    assert engine.bot_state["factor_rebalance"]["scores"]["AAA/USD"] == baseline
    assert engine.bot_state["factor_rebalance"]["scores"] == {"AAA/USD": baseline}


def test_intraday_exit_is_retried_after_a_resting_order_disappears(engine, monkeypatch):
    # A sell was submitted, then cancelled/expired without filling. The
    # position is still in the wallet and no order is pending, so the next
    # tick must re-submit rather than silently hold a collapsed name.
    frame = _rising_frame()
    baseline = alphaagent_70(frame)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": baseline}}
    wallet = {"AAA": {"Free": 5.0, "Lock": 0.0}, "USD": {"Free": 0.0, "Lock": 0.0}}
    tickers = {"AAA/USD": {"LastPrice": 60.0, "MaxBid": 59.9, "MinAsk": 60.1}}
    monkeypatch.setattr(engine.data, "candles", lambda *a, **k: frame)
    assert engine._factor_intraday_exit(wallet, RULES, tickers, {"AAA/USD"}) == []
    engine.executor.calls.clear()
    assert engine._factor_intraday_exit(wallet, RULES, tickers, set()) == ["AAA/USD"]
    assert [call[:2] for call in engine.executor.calls] == [("sell", "AAA/USD")]


def test_intraday_exit_holds_when_score_is_stable(engine, monkeypatch):
    frame = _rising_frame()
    baseline = alphaagent_70(frame)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": baseline}}
    wallet = {"AAA": {"Free": 5.0, "Lock": 0.0}, "USD": {"Free": 0.0, "Lock": 0.0}}
    tickers = {"AAA/USD": {"LastPrice": float(frame["close"].iloc[-1]), "MaxBid": 129.9, "MinAsk": 130.1}}
    monkeypatch.setattr(engine.data, "candles", lambda *a, **k: frame)
    assert engine._factor_intraday_exit(wallet, RULES, tickers, set()) == []
    assert engine.executor.calls == []


def test_intraday_exit_skips_pairs_with_pending_orders(engine, monkeypatch):
    frame = _rising_frame()
    baseline = alphaagent_70(frame)
    engine.bot_state["factor_rebalance"] = {"scores": {"AAA/USD": baseline}}
    wallet = {"AAA": {"Free": 5.0, "Lock": 0.0}, "USD": {"Free": 0.0, "Lock": 0.0}}
    tickers = {"AAA/USD": {"LastPrice": 60.0, "MaxBid": 59.9, "MinAsk": 60.1}}
    monkeypatch.setattr(engine.data, "candles", lambda *a, **k: frame)
    assert engine._factor_intraday_exit(wallet, RULES, tickers, {"AAA/USD"}) == []
