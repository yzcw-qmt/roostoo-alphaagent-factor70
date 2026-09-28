from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from roostoo_bot.features import FEATURE_COLUMNS, build_features
from roostoo_bot.ma_macd_strategy import MaMacdSignalGenerator


def load(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def model() -> XGBClassifier:
    return XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, eval_metric="logloss",
                         random_state=42, n_jobs=2)


def simulate(frame: pd.DataFrame, actions: list[tuple[str, bool, bool]], start: int, end: int, fee: float) -> dict:
    cash, quantity, entry = 1.0, 0.0, 0.0
    side, trades, fees = None, 0, 0.0
    curve = []
    for index in range(start, end - 1):
        action, exit_long, exit_short = actions[index]
        price, next_price = float(frame.close.iloc[index]), float(frame.close.iloc[index + 1])
        pnl = price / entry - 1 if side == "LONG" else entry / price - 1 if side == "SHORT" else 0.0
        close = side is not None and pnl >= 0 and ((side == "LONG" and exit_long) or (side == "SHORT" and exit_short))
        reverse = side is not None and pnl >= 0 and action in ("LONG", "SHORT") and action != side
        if close or reverse:
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
        curve.append(cash + (quantity * next_price if side == "LONG" else quantity * (entry - next_price) if side == "SHORT" else 0))
    equity = np.asarray(curve)
    returns = pd.Series(equity).pct_change().dropna()
    drawdown = equity / np.maximum.accumulate(equity) - 1
    sharpe = float(returns.mean() / returns.std() * math.sqrt(48 * 15)) if returns.std() else 0.0
    return {"return": float(equity[-1] - 1), "max_drawdown": float(drawdown.min()), "sharpe": sharpe, "trades": trades, "fees": fees}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep XGBoost/technical weights for scheme one")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("reports/scheme1_weight_sweep_24x15d.json"))
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--train-days", type=int, default=90)
    args = parser.parse_args()
    weights = [round(x / 10, 1) for x in range(11)]
    bars, train_bars = args.days * 48, args.train_days * 48
    report = []
    for pair in ("BTC/USD", "ETH/USD"):
        frame = load(args.data_root / f"{pair.replace('/', '-')}_30m_1y.json")
        features = build_features(frame)
        technical = MaMacdSignalGenerator().generate_series(pair, frame)
        labels = (frame.close.shift(-6) / frame.close > 1).astype(int)
        first = len(frame) - args.segments * bars
        for segment in range(args.segments):
            start, end = first + segment * bars, first + (segment + 1) * bars
            train_start, train_end = start - train_bars, start - 6
            train = features.iloc[train_start:train_end]
            valid = train.notna().all(axis=1)
            fitted = model()
            fitted.fit(train.loc[valid, FEATURE_COLUMNS], labels.iloc[train_start:train_end].loc[valid], verbose=False)
            test = features.iloc[start:end]
            valid_test = test.notna().all(axis=1)
            probability = np.full(len(test), 0.5)
            probability[valid_test.to_numpy()] = fitted.predict_proba(test.loc[valid_test, FEATURE_COLUMNS])[:, 1]
            for xgb_weight in weights:
                actions = [("HOLD", False, False)] * len(frame)
                for offset, signal in enumerate(technical[start:end]):
                    strength = xgb_weight * probability[offset] + (1 - xgb_weight) * signal.strength
                    action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
                    actions[start + offset] = (action, signal.exit_long, signal.exit_short)
                for fee in (0.001, 0.0005):
                    result = simulate(frame, actions, start, end, fee)
                    result.update(pair=pair, xgb_weight=xgb_weight, technical_weight=round(1 - xgb_weight, 1), segment=segment + 1, fee_rate=fee, start=str(frame.index[start]), end=str(frame.index[end - 1]))
                    report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for pair in ("BTC/USD", "ETH/USD"):
        for fee in (0.001, 0.0005):
            rows = [x for x in report if x["pair"] == pair and x["fee_rate"] == fee]
            print("\n", pair, f"fee={fee:.2%}")
            for weight in weights:
                subset = [x for x in rows if x["xgb_weight"] == weight]
                compounded = math.prod(1 + x["return"] for x in subset) - 1
                print(f"xgb={weight:.1f} tech={1-weight:.1f} compound={compounded:.2%} mean={np.mean([x['return'] for x in subset]):.2%} positive={sum(x['return'] > 0 for x in subset)}/{len(subset)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
