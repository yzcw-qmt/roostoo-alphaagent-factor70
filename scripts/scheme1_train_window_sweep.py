from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from roostoo_bot.features import FEATURE_COLUMNS, build_features
from roostoo_bot.ma_macd_strategy import MaMacdSignalGenerator
from scripts.scheme1_weight_sweep import load, model, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare separate per-asset rolling training windows")
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("reports/scheme1_train_window_sweep.json"))
    parser.add_argument("--xgb-weight", type=float, default=0.2)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--windows", nargs="+", type=int, default=[30, 60, 90, 120, 180])
    args = parser.parse_args()
    bars = args.days * 48
    report = []
    for pair in ("BTC/USD", "ETH/USD"):
        frame = load(args.data_root / f"{pair.replace('/', '-')}_30m_1y.json")
        features = build_features(frame)
        technical = MaMacdSignalGenerator().generate_series(pair, frame)
        labels = (frame.close.shift(-6) / frame.close > 1).astype(int)
        for train_days in args.windows:
            train_bars = train_days * 48
            segments = args.segments
            first = len(frame) - segments * bars
            if first < train_bars:
                raise ValueError(
                    f"{pair} needs at least {train_days + segments * args.days} days "
                    f"for {segments} slices, but only has {len(frame) / 48:.1f} days"
                )
            for segment in range(segments):
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
                actions = [("HOLD", False, False)] * len(frame)
                for offset, signal in enumerate(technical[start:end]):
                    strength = args.xgb_weight * probability[offset] + (1 - args.xgb_weight) * signal.strength
                    action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
                    actions[start + offset] = (action, signal.exit_long, signal.exit_short)
                for fee in (0.001, 0.0005):
                    result = simulate(frame, actions, start, end, fee)
                    result.update(pair=pair, train_days=train_days, segment=segment + 1, segments=segments, fee_rate=fee, start=str(frame.index[start]), end=str(frame.index[end - 1]))
                    report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for pair in ("BTC/USD", "ETH/USD"):
        print("\n", pair)
        for train_days in args.windows:
            for fee in (0.001, 0.0005):
                rows = [x for x in report if x["pair"] == pair and x["train_days"] == train_days and x["fee_rate"] == fee]
                vals = [x["return"] for x in rows]
                print(f"train={train_days:>3}d fee={fee:.2%} n={len(vals):>2} mean={statistics.mean(vals):+.2%} median={statistics.median(vals):+.2%} std={statistics.pstdev(vals):.2%} positive={sum(v > 0 for v in vals)}/{len(vals)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
