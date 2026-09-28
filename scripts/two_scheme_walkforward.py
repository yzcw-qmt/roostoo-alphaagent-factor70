from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from roostoo_bot.features import FEATURE_COLUMNS, build_features, make_training_set
from roostoo_bot.ma_macd_strategy import MaMacdSignalGenerator


def load(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def macd_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    generator = MaMacdSignalGenerator()
    signals = generator.generate_series("PAIR", frame)
    close = frame["close"].astype(float)
    ma10, ma60 = close.rolling(10).mean(), close.rolling(60).mean()
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    atr = (frame["high"] - frame["low"]).rolling(14).mean()
    rows = []
    for index, signal in enumerate(signals):
        components = signal.score_components or {}
        rows.append({
            "ma_trend": components.get("ma_trend", 0.0),
            "ma_cross": components.get("ma_cross", 0.0),
            "macd_zero_axis": components.get("macd_zero_axis", 0.0),
            "diff_dea": components.get("diff_dea", 0.0),
            "diff_low_extremum": components.get("diff_low_extremum", 0.0),
            "diff_high_extremum": components.get("diff_high_extremum", 0.0),
            "ma_gap": (ma10.iloc[index] / ma60.iloc[index] - 1) if ma60.iloc[index] else np.nan,
            "macd_diff": diff.iloc[index],
            "macd_dea": dea.iloc[index],
            "macd_hist": diff.iloc[index] - dea.iloc[index],
            "atr_ratio": atr.iloc[index] / close.iloc[index],
        })
    return pd.DataFrame(rows, index=frame.index), signals


def make_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )


def simulate(frame: pd.DataFrame, actions: list[tuple[str, bool, bool]], start: int, end: int, fee: float) -> dict:
    cash, quantity, entry = 1.0, 0.0, 0.0
    side, trades, fees = None, 0, 0.0
    curve = []
    for index in range(start, end - 1):
        action, exit_long, exit_short = actions[index]
        price, next_price = float(frame.close.iloc[index]), float(frame.close.iloc[index + 1])
        pnl = price / entry - 1 if side == "LONG" else entry / price - 1 if side == "SHORT" else 0.0
        exit_signal = side is not None and pnl >= 0 and ((side == "LONG" and exit_long) or (side == "SHORT" and exit_short))
        reverse = side is not None and pnl >= 0 and action in ("LONG", "SHORT") and action != side
        if exit_signal or reverse:
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
    parser = argparse.ArgumentParser(description="Walk-forward comparison of two MACD/XGBoost schemes")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("reports/two_scheme_walkforward_24x15d.json"))
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--train-days", type=int, default=90)
    args = parser.parse_args()
    bars = args.days * 48
    train_bars = args.train_days * 48
    report = []
    for pair in ("BTC/USD", "ETH/USD"):
        frame = load(args.data_root / f"{pair.replace('/', '-')}_30m_1y.json")
        general = build_features(frame)
        macd, technical_signals = macd_features(frame)
        labels = (frame.close.shift(-6) / frame.close > 1).astype(int)
        first = len(frame) - args.segments * bars
        for segment in range(args.segments):
            start, end = first + segment * bars, first + (segment + 1) * bars
            train_start, train_end = start - train_bars, start - 6
            sets = {
                "scheme_1": (general[FEATURE_COLUMNS], technical_signals, 0.5),
                "scheme_2": (macd, technical_signals, 1.0),
            }
            for scheme, (features, signals, technical_weight) in sets.items():
                valid = features.iloc[train_start:train_end].notna().all(axis=1) & labels.iloc[train_start:train_end].notna()
                x_train = features.iloc[train_start:train_end].loc[valid]
                y_train = labels.iloc[train_start:train_end].loc[valid]
                model = make_model()
                model.fit(x_train, y_train, verbose=False)
                x_test = features.iloc[start:end].copy()
                valid_test = x_test.notna().all(axis=1)
                probability = np.full(len(x_test), 0.5)
                probability[valid_test.to_numpy()] = model.predict_proba(x_test.loc[valid_test])[:, 1]
                actions = []
                for offset, signal in enumerate(signals[start:end]):
                    if scheme == "scheme_1":
                        strength = 0.5 * probability[offset] + 0.5 * signal.strength
                    else:
                        strength = probability[offset]
                    action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
                    actions.append((action, signal.exit_long, signal.exit_short))
                full_actions = [("HOLD", False, False)] * start + actions + [("HOLD", False, False)] * (len(frame) - end)
                for fee in (0.001, 0.0005):
                    result = simulate(frame, full_actions, start, end, fee)
                    result.update(pair=pair, scheme=scheme, segment=segment + 1, fee_rate=fee, start=str(frame.index[start]), end=str(frame.index[end - 1]), train_days=args.train_days)
                    report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for scheme in ("scheme_1", "scheme_2"):
        for pair in ("BTC/USD", "ETH/USD"):
            for fee in (0.001, 0.0005):
                rows = [x for x in report if x["scheme"] == scheme and x["pair"] == pair and x["fee_rate"] == fee]
                compounded = math.prod(1 + x["return"] for x in rows) - 1
                print(scheme, pair, f"fee={fee:.2%}", f"compound={compounded:.2%}", f"mean={np.mean([x['return'] for x in rows]):.2%}", f"positive={sum(x['return'] > 0 for x in rows)}/{len(rows)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
