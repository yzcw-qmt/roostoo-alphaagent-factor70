from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from roostoo_bot.ma_macd_strategy import MaMacdSignalGenerator
from roostoo_bot.features import FEATURE_COLUMNS, build_features
from roostoo_bot.model import DirectionModel


def load_candles(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def evaluate(
    pair: str,
    frame: pd.DataFrame,
    horizons: tuple[int, ...],
    model: DirectionModel | None = None,
    model_weight: float = 0.60,
    start: int = 0,
) -> dict:
    signals = MaMacdSignalGenerator().generate_series(pair, frame)
    closes = frame["close"].to_numpy(float)
    probabilities: list[float] | None = None
    if model is not None and model.model is not None:
        features = build_features(frame)
        valid = features[FEATURE_COLUMNS].notna().all(axis=1)
        probabilities = [0.5] * len(frame)
        predicted = model.model.predict_proba(features.loc[valid, FEATURE_COLUMNS])[:, 1]
        for index, probability in zip(features.index[valid], predicted):
            probabilities[frame.index.get_loc(index)] = float(probability)
    report: dict[str, object] = {"pair": pair, "bars": len(frame), "start": start, "actions": {}}
    actions = report["actions"]
    assert isinstance(actions, dict)
    for action in ("LONG", "SHORT", "HOLD"):
        actions[action] = sum(
            (signal.action if probabilities is None else
             ("LONG" if model_weight * probabilities[index] + (1 - model_weight) * signal.strength >= 0.62
              else "SHORT" if model_weight * probabilities[index] + (1 - model_weight) * signal.strength <= 0.38
              else "HOLD")) == action
            for index, signal in enumerate(signals[start:], start)
        )

    for horizon in horizons:
        samples: list[tuple[str, float]] = []
        for index, signal in enumerate(signals[:-horizon]):
            if index < start:
                continue
            action = signal.action
            if probabilities is not None:
                strength = model_weight * probabilities[index] + (1 - model_weight) * signal.strength
                action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
            if action == "HOLD":
                continue
            future_return = closes[index + horizon] / closes[index] - 1
            signed_return = future_return if action == "LONG" else -future_return
            samples.append((action, float(signed_return)))
        key = f"{horizon}_bars"
        if not samples:
            report[key] = {"signals": 0, "hit_rate": None, "mean_signed_return": None}
            continue
        report[key] = {
            "signals": len(samples),
            "long_signals": sum(action == "LONG" for action, _ in samples),
            "short_signals": sum(action == "SHORT" for action, _ in samples),
            "hit_rate": sum(value > 0 for _, value in samples) / len(samples),
            "mean_signed_return": sum(value for _, value in samples) / len(samples),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pure technical final signal without model features")
    parser.add_argument("--pair", choices=("BTC/USD", "ETH/USD"), default="BTC/USD")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("reports/technical_signal_evaluation.json"))
    parser.add_argument("--model", type=Path, help="Optional XGBoost model for hybrid evaluation")
    parser.add_argument("--model-weight", type=float, default=0.60)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    args = parser.parse_args()
    symbol = args.pair.replace("/", "-")
    path = args.data_root / f"{symbol}_30m_1y.json"
    frame = load_candles(path)
    start = int(len(frame) * (1 - args.test_fraction)) if args.test_fraction else 0
    model = DirectionModel(args.model) if args.model else None
    result = evaluate(args.pair, frame, (1, 4, 8, 16, 48), model, args.model_weight, start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
