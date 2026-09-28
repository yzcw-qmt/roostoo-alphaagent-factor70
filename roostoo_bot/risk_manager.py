from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class RiskState:
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day: str = ""
    halted: bool = False


class RiskManager:
    def __init__(
        self,
        max_position_fraction: float = 0.30,
        fallback_position_fraction: float = 0.10,
        kelly_fraction: float = 0.30,
        stop_loss: float = 0.03,
        daily_loss_limit: float = 0.05,
        max_drawdown: float = 0.12,
        trailing_activation: float = 0.03,
        trailing_gap: float = 0.025,
    ) -> None:
        self.max_position_fraction = max_position_fraction
        self.fallback_position_fraction = fallback_position_fraction
        self.kelly_fraction = kelly_fraction
        self.stop_loss = stop_loss
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.trailing_activation = trailing_activation
        self.trailing_gap = trailing_gap
        self.state = RiskState()

    def update_equity(self, equity: float, today: date | None = None) -> str | None:
        day = (today or date.today()).isoformat()
        if self.state.day != day:
            self.state.day = day
            self.state.day_start_equity = equity
            self.state.halted = False
        self.state.peak_equity = max(self.state.peak_equity, equity)
        daily_return = equity / self.state.day_start_equity - 1 if self.state.day_start_equity else 0
        drawdown = equity / self.state.peak_equity - 1 if self.state.peak_equity else 0
        if drawdown <= -self.max_drawdown:
            self.state.halted = True
            return "MAX_DRAWDOWN"
        if daily_return <= -self.daily_loss_limit:
            self.state.halted = True
            return "DAILY_LOSS"
        return None

    def position_fraction(
        self, confidence: float, win_rate: float | None = None, avg_win: float | None = None, avg_loss: float | None = None
    ) -> float:
        confidence_scale = min(1.0, max(0.0, abs(confidence - 0.5) / 0.5))
        if win_rate is None or not avg_win or not avg_loss:
            # 10% base allocation, scaling linearly to 30% at maximum confidence.
            return min(self.max_position_fraction, self.fallback_position_fraction + 0.20 * confidence_scale)
        odds = avg_win / avg_loss
        full_kelly = win_rate - (1 - win_rate) / odds
        return max(0.0, min(self.max_position_fraction, self.kelly_fraction * full_kelly))

    def should_stop(self, side: str, entry: float, current: float, best: float) -> str | None:
        pnl = current / entry - 1 if side == "LONG" else entry / current - 1
        if pnl <= -self.stop_loss:
            return "STOP_LOSS"
        if side == "LONG" and best / entry - 1 >= self.trailing_activation and current <= best * (1 - self.trailing_gap):
            return "TRAILING_STOP"
        if side == "SHORT" and entry / best - 1 >= self.trailing_activation and current >= best * (1 + self.trailing_gap):
            return "TRAILING_STOP"
        return None
