from datetime import date

from roostoo_bot.risk_manager import RiskManager


def test_daily_loss_halts_trading() -> None:
    risk = RiskManager()
    today = date(2026, 9, 21)
    assert risk.update_equity(100000, today) is None
    assert risk.update_equity(94999, today) == "DAILY_LOSS"
    assert risk.state.halted


def test_total_drawdown_halts_trading() -> None:
    risk = RiskManager(daily_loss_limit=0.50)
    today = date(2026, 9, 21)
    risk.update_equity(100000, today)
    assert risk.update_equity(87999, today) == "MAX_DRAWDOWN"


def test_long_stop_and_trailing_stop() -> None:
    risk = RiskManager()
    assert risk.should_stop("LONG", 100, 96.9, 100) == "STOP_LOSS"
    assert risk.should_stop("LONG", 100, 104, 107) == "TRAILING_STOP"


def test_short_stop() -> None:
    risk = RiskManager()
    assert risk.should_stop("SHORT", 100, 104, 100) == "STOP_LOSS"

