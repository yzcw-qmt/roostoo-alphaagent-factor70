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


def hybrid_signals(pair: str, frame: pd.DataFrame, model: DirectionModel, model_weight: float) -> list[tuple[str, bool, bool]]:
    technical = MaMacdSignalGenerator().generate_series(pair, frame)
    features = build_features(frame)
    probabilities = np.full(len(frame), 0.5)
    valid = features[FEATURE_COLUMNS].notna().all(axis=1)
    if model.model is not None:
        probabilities[valid.to_numpy()] = model.model.predict_proba(features.loc[valid, FEATURE_COLUMNS])[:, 1]
    result = []
    for index, signal in enumerate(technical):
        strength = model_weight * probabilities[index] + (1 - model_weight) * signal.technical_score
        action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
        result.append((action, signal.exit_long, signal.exit_short))
    return result


def simulate(frame: pd.DataFrame, signals: list[tuple[str, bool, bool]], fee_rate: float) -> dict:
    cash, quantity, entry = 1.0, 0.0, 0.0
    side, trades, fees = None, 0, 0.0
    equity_curve = []
    for index in range(60, len(frame) - 1):
        action, exit_long, exit_short = signals[index]
        price = float(frame["close"].iloc[index])
        next_price = float(frame["close"].iloc[index + 1])
        pnl = price / entry - 1 if side == "LONG" else entry / price - 1 if side == "SHORT" else 0.0
        indicator_exit = side is not None and pnl >= 0 and (
            (side == "LONG" and exit_long) or (side == "SHORT" and exit_short)
        )
        reverse = side is not None and pnl >= 0 and action in ("LONG", "SHORT") and action != side
        if indicator_exit or reverse:
            cash += quantity * price if side == "LONG" else quantity * (entry - price)
            fee = abs(cash) * fee_rate
            cash -= fee
            fees += fee
            quantity, entry, side = 0.0, 0.0, None
            trades += 1
        if side is None and action in ("LONG", "SHORT"):
            notional = cash * 0.30
            fee = notional * fee_rate
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
    return {
        "return": float(equity[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "trades": trades,
        "fees": fees,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest current MA/MACD + XGBoost signals on 30m data")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--model", type=Path, default=Path("models/xgb_30m_multi.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/hybrid_30m_1y_fee_comparison.json"))
    parser.add_argument("--model-weight", type=float, default=0.60)
    args = parser.parse_args()
    model = DirectionModel(args.model)
    report = []
    for pair in ("BTC/USD", "ETH/USD"):
        frame = load(args.data_root / f"{pair.replace('/', '-')}_30m_1y.json")
        signals = hybrid_signals(pair, frame, model, args.model_weight)
        for fee_rate in (0.001, 0.0005):
            result = simulate(frame, signals, fee_rate)
            result.update(pair=pair, fee_rate=fee_rate, model_loaded=model.is_ml, bars=len(frame))
            report.append(result)
            print(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
