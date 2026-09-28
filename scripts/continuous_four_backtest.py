from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from roostoo_bot.ma_macd_strategy import _atr, _online_extrema


LONG_LIMIT_FEE = float(os.getenv("LONG_LIMIT_FEE", "0.0005"))
OTHER_FEE = float(os.getenv("OTHER_FEE", "0.001"))


def load(path: str) -> pd.DataFrame:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def updated_extrema(frame: pd.DataFrame, atr: pd.Series) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    atr_values = atr.to_numpy(float)
    highs: list[tuple[int, int]] = []
    lows: list[tuple[int, int]] = []
    state, current_high, high_point = "HIGH", -np.inf, None
    current_low, low_point = np.inf, None
    for t in range(len(frame)):
        if state == "HIGH":
            if high_point is None or high[t] >= current_high:
                current_high, high_point = high[t], t
            if current_high - low[high_point:t + 1].min() >= 1.2 * atr_values[t]:
                highs.append((high_point, t))
                state, start = "LOW", high_point + 1
                if start <= t:
                    low_point = start + int(low[start:t + 1].argmin())
                    current_low = low[low_point]
                else:
                    low_point, current_low = None, np.inf
        else:
            if low_point is None or low[t] <= current_low:
                current_low, low_point = low[t], t
            if high[low_point:t + 1].max() - current_low >= 1.2 * atr_values[t]:
                lows.append((low_point, t))
                state, start = "HIGH", low_point + 1
                if start <= t:
                    high_point = start + int(high[start:t + 1].argmax())
                    current_high = high[high_point]
                else:
                    high_point, current_high = None, -np.inf
    return highs, lows


def scores(frame: pd.DataFrame, original: bool, long_threshold: float = 0.62, short_threshold: float = 0.38) -> list[tuple[str, bool, bool, float, float]]:
    close = frame["close"].astype(float)
    ma10, ma60 = close.rolling(10).mean(), close.rolling(60).mean()
    diff = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea, atr = diff.ewm(span=9, adjust=False).mean(), _atr(frame, 14)
    if original:
        high_points, low_points = _online_extrema(frame, atr, diff, 20, 1.5, 0.02, False)
        highs = {x.confirmed_at: x.point for x in high_points}
        lows = {x.confirmed_at: x.point for x in low_points}
        lookback = 20
    else:
        high_points, low_points = updated_extrema(frame, atr)
        highs = {confirmed: point for point, confirmed in high_points}
        lows = {confirmed: point for point, confirmed in low_points}
        lookback = 15
    output = []
    weights = (0.24, 0.20, 0.22, 0.16, 0.09, 0.09)
    for t in range(len(frame)):
        exit_long, exit_short = bool(diff.iloc[t] <= 0), bool(diff.iloc[t] >= 0)
        if t < 60:
            output.append(("HOLD", exit_long, exit_short, 0.0, 0.5))
            continue
        cross_up = ma10.iloc[t] > ma60.iloc[t] and ma10.iloc[t - 1] <= ma60.iloc[t - 1]
        cross_down = ma10.iloc[t] < ma60.iloc[t] and ma10.iloc[t - 1] >= ma60.iloc[t - 1]
        low_point, high_point = lows.get(t), highs.get(t)
        low_extreme = low_point is not None and diff.iloc[low_point] <= diff.iloc[max(0, low_point - lookback):low_point].min()
        high_extreme = high_point is not None and diff.iloc[high_point] >= diff.iloc[max(0, high_point - lookback):high_point].max()
        evidence = (
            1 if ma10.iloc[t] > ma60.iloc[t] else -1,
            1 if cross_up else -1 if cross_down else 0,
            1 if diff.iloc[t] > 0 and dea.iloc[t] > 0 else -1 if diff.iloc[t] < 0 and dea.iloc[t] < 0 else 0,
            1 if diff.iloc[t] > dea.iloc[t] else -1 if diff.iloc[t] < dea.iloc[t] else 0,
            1 if low_extreme else 0,
            -1 if high_extreme else 0,
        )
        score = float(np.clip(0.5 + 0.5 * sum(a * b for a, b in zip(evidence, weights)), 0, 1))
        action = "LONG" if score >= long_threshold else "SHORT" if score <= short_threshold else "HOLD"
        output.append((action, exit_long, exit_short, abs(score - 0.5) / 0.5, score))
    return output


def simulate(frame: pd.DataFrame, signal: list[tuple[str, bool, bool, float, float]], stop: bool, signal_exit: bool = False, long_threshold: float = 0.62, short_threshold: float = 0.38) -> dict:
    cash, quantity, entry = 1.0, 0.0, 0.0
    side, trades, fees = None, 0, 0.0
    cooldown = 0  # settings.cooldown_minutes=55; two 30-minute bars
    equity_curve = []
    for index in range(60, len(frame) - 1):
        cooldown = max(0, cooldown - 1)
        action, exit_long, exit_short, confidence, score = signal[index]
        price, next_price = float(frame["close"].iloc[index]), float(frame["close"].iloc[index + 1])
        pnl = price / entry - 1 if side == "LONG" else entry / price - 1 if side == "SHORT" else 0.0
        stop_out = stop and side is not None and pnl <= -0.02
        indicator_exit = side is not None and pnl >= 0 and (
            (side == "LONG" and (score < long_threshold if signal_exit else exit_long))
            or (side == "SHORT" and (score > short_threshold if signal_exit else exit_short))
        )
        reverse = side is not None and pnl >= 0 and action in ("LONG", "SHORT") and action != side
        if stop_out or indicator_exit or reverse:
            cash += quantity * price if side == "LONG" else quantity * (entry - price)
            # Long limit orders are maker orders on both entry and exit.
            # Short orders and other executions use the standard fee.
            fee = abs(cash) * (LONG_LIMIT_FEE if side == "LONG" else OTHER_FEE)
            cash -= fee
            fees += fee
            quantity, entry, side = 0.0, 0.0, None
            trades += 1
            cooldown = 2
        if side is None and cooldown == 0 and action in ("LONG", "SHORT"):
            notional = cash * min(0.30, 0.10 + 0.20 * confidence)
            fee = notional * (LONG_LIMIT_FEE if action == "LONG" else OTHER_FEE)
            cash -= fee
            fees += fee
            if action == "LONG":
                cash -= notional
            quantity, entry, side = notional / price, price, action
            trades += 1
        equity_curve.append(cash + (quantity * next_price if side == "LONG" else quantity * (entry - next_price) if side == "SHORT" else 0))
    equity = np.asarray(equity_curve)
    returns = pd.Series(equity).pct_change().dropna()
    drawdown = equity / np.maximum.accumulate(equity) - 1
    sharpe = float(returns.mean() / returns.std() * math.sqrt(48 * 365)) if returns.std() else 0.0
    return {"return": float(equity[-1] - 1), "max_drawdown": float(drawdown.min()), "sharpe": sharpe, "trades": trades, "fees": fees}


def main() -> None:
    report = []
    for pair, path in (("BTC/USD", "data/history/BTC-USD_30m_1y.json"), ("ETH/USD", "data/history/ETH-USD_30m_1y.json")):
        frame = load(path)
        for original in (False, True):
            signal = scores(frame, original)
            for stop in (False, True):
                result = simulate(frame, signal, stop)
                result.update(pair=pair, variant=("original" if original else "new_diff") + ("_stop_2pct" if stop else "_no_stop"))
                report.append(result)
                print(result)
    output = Path(os.getenv(
        "BACKTEST_OUTPUT",
        "reports/ma_macd_30m_1y_continuous_four_variants.json",
    ))
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
