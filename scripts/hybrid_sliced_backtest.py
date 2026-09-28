from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from roostoo_bot.features import FEATURE_COLUMNS, build_features
from roostoo_bot.ma_macd_strategy import MaMacdSignalGenerator
from roostoo_bot.model import DirectionModel


def load(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def signals(pair: str, frame: pd.DataFrame, model: DirectionModel, weight: float) -> list[tuple[str, bool, bool]]:
    technical = MaMacdSignalGenerator().generate_series(pair, frame)
    features = build_features(frame)
    probability = np.full(len(frame), 0.5)
    valid = features[FEATURE_COLUMNS].notna().all(axis=1)
    if model.model is not None:
        probability[valid.to_numpy()] = model.model.predict_proba(features.loc[valid, FEATURE_COLUMNS])[:, 1]
    output = []
    for index, signal in enumerate(technical):
        strength = weight * probability[index] + (1 - weight) * signal.technical_score
        action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
        output.append((action, signal.exit_long, signal.exit_short))
    return output


def simulate(frame: pd.DataFrame, signal: list[tuple[str, bool, bool]], start: int, end: int, fee: float) -> dict:
    cash, quantity, entry = 1.0, 0.0, 0.0
    side, trades, fees = None, 0, 0.0
    equity_curve = []
    for index in range(start, end - 1):
        action, exit_long, exit_short = signal[index]
        price = float(frame["close"].iloc[index])
        next_price = float(frame["close"].iloc[index + 1])
        pnl = price / entry - 1 if side == "LONG" else entry / price - 1 if side == "SHORT" else 0.0
        should_exit = side is not None and pnl >= 0 and (
            (side == "LONG" and exit_long) or (side == "SHORT" and exit_short)
        )
        reverse = side is not None and pnl >= 0 and action in ("LONG", "SHORT") and action != side
        if should_exit or reverse:
            cash += quantity * price if side == "LONG" else quantity * (entry - price)
            paid = abs(cash) * fee
            cash -= paid
            fees += paid
            quantity, entry, side = 0.0, 0.0, None
            trades += 1
        if side is None and action in ("LONG", "SHORT"):
            notional = cash * 0.30
            paid = notional * fee
            cash -= paid
            fees += paid
            if action == "LONG":
                cash -= notional
            quantity, entry, side = notional / price, price, action
            trades += 1
        equity_curve.append(cash + (quantity * next_price if side == "LONG" else quantity * (entry - next_price) if side == "SHORT" else 0))
    equity = np.asarray(equity_curve)
    returns = pd.Series(equity).pct_change().dropna()
    drawdown = equity / np.maximum.accumulate(equity) - 1
    sharpe = float(returns.mean() / returns.std() * math.sqrt(48 * 15)) if returns.std() else 0.0
    return {
        "return": float(equity[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "trades": trades,
        "fees": fees,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 24 independent 15-day slices of the hybrid 30m strategy")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--model", type=Path, default=Path("models/xgb_30m_multi.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/hybrid_30m_24x15d.json"))
    parser.add_argument("--model-weight", type=float, default=0.60)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--days", type=int, default=15)
    args = parser.parse_args()
    model = DirectionModel(args.model)
    bars_per_segment = args.days * 48
    report = []
    for pair in ("BTC/USD", "ETH/USD"):
        frame = load(args.data_root / f"{pair.replace('/', '-')}_30m_1y.json")
        signal = signals(pair, frame, model, args.model_weight)
        total_bars = args.segments * bars_per_segment
        first = len(frame) - total_bars
        if first < 0:
            raise ValueError(f"{pair} has only {len(frame)} bars; need {total_bars}")
        for segment in range(args.segments):
            start = first + segment * bars_per_segment
            end = start + bars_per_segment
            for fee in (0.001, 0.0005):
                result = simulate(frame, signal, start, end, fee)
                result.update(
                    pair=pair,
                    segment=segment + 1,
                    start=str(frame.index[start]),
                    end=str(frame.index[end - 1]),
                    days=args.days,
                    fee_rate=fee,
                    model_loaded=model.is_ml,
                )
                report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for pair in ("BTC/USD", "ETH/USD"):
        for fee in (0.001, 0.0005):
            rows = [row for row in report if row["pair"] == pair and row["fee_rate"] == fee]
            print(pair, f"fee={fee:.2%}", f"mean={np.mean([row['return'] for row in rows]):.2%}", f"median={np.median([row['return'] for row in rows]):.2%}", f"positive={sum(row['return'] > 0 for row in rows)}/{len(rows)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
