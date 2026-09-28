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
    parser = argparse.ArgumentParser(description="Grid search separate-asset training windows and XGBoost weights")
    parser.add_argument("--data-root", type=Path, default=Path("data/history/550d"))
    parser.add_argument("--output", type=Path, default=Path("reports/scheme1_grid_search_550d_24x15d.json"))
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--windows", nargs="+", type=int, default=[30, 60, 90, 120, 180])
    parser.add_argument("--weights", nargs="+", type=float, default=[round(x / 10, 1) for x in range(11)])
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
            first = len(frame) - args.segments * bars
            if first < train_bars:
                raise ValueError(f"{pair}: not enough data for train={train_days}d and {args.segments} segments")
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
                for weight in args.weights:
                    actions = [("HOLD", False, False)] * len(frame)
                    for offset, signal in enumerate(technical[start:end]):
                        strength = weight * probability[offset] + (1 - weight) * signal.strength
                        action = "LONG" if strength >= 0.62 else "SHORT" if strength <= 0.38 else "HOLD"
                        actions[start + offset] = (action, signal.exit_long, signal.exit_short)
                    for fee in (0.001, 0.0005):
                        result = simulate(frame, actions, start, end, fee)
                        result.update(pair=pair, train_days=train_days, xgb_weight=weight, technical_weight=round(1 - weight, 1), segment=segment + 1, fee_rate=fee)
                        report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summaries = []
    for pair in ("BTC/USD", "ETH/USD"):
        for fee in (0.001, 0.0005):
            for train_days in args.windows:
                for weight in args.weights:
                    rows = [x for x in report if x["pair"] == pair and x["fee_rate"] == fee and x["train_days"] == train_days and x["xgb_weight"] == weight]
                    values = [x["return"] for x in rows]
                    summaries.append({
                        "pair": pair, "fee_rate": fee, "train_days": train_days, "xgb_weight": weight,
                        "mean": statistics.mean(values), "median": statistics.median(values),
                        "std": statistics.pstdev(values), "positive": sum(v > 0 for v in values), "segments": len(values),
                    })
            ranked = sorted([x for x in summaries if x["pair"] == pair and x["fee_rate"] == fee], key=lambda x: (x["median"], x["mean"]), reverse=True)
            print("\n", pair, f"fee={fee:.2%} top by median")
            for row in ranked[:10]:
                print(f"train={row['train_days']:>3}d xgb={row['xgb_weight']:.1f} mean={row['mean']:+.2%} median={row['median']:+.2%} std={row['std']:.2%} positive={row['positive']}/{row['segments']}")
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
