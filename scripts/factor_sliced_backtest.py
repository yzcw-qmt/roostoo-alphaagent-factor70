from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roostoo_bot.factor_strategy import CrossSectionalFactorStrategy


def load_daily(history_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in sorted(history_dir.glob("*-USD_1h_*.csv")):
        pair = path.name.split("_1h_")[0].replace("-", "/")
        frame = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC").normalize()
        daily = frame.resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
        }).dropna()
        result[pair] = daily
    return result


def factor_targets(
    daily: dict[str, pd.DataFrame], strategy_name: str, top_n: int,
) -> dict[pd.Timestamp, dict[str, float]]:
    strategy = CrossSectionalFactorStrategy(strategy_name, top_n)
    dates = sorted(set.intersection(*(set(frame.index) for frame in daily.values())))
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    for day in dates:
        history = {pair: frame.loc[:day] for pair, frame in daily.items()}
        signals = strategy.rank(history)
        chosen = [signal.pair for signal in signals if signal.selected]
        if chosen:
            targets[day] = {pair: 1.0 / len(chosen) for pair in chosen}
    return targets


def simulate_segments(
    daily: dict[str, pd.DataFrame], targets: dict[pd.Timestamp, dict[str, float]],
    count: int = 24, segment_days: int = 15, fee_rate: float = 0.0005,
) -> list[dict]:
    dates = sorted(set.intersection(*(set(frame.index) for frame in daily.values())))
    # Use the latest complete 24 segments, matching the previous 24x15d reports.
    total_days = count * segment_days
    start_index = len(dates) - total_days - 1  # one extra day for the final open-to-open return
    if start_index < 1:
        raise ValueError("Not enough common daily history for requested segments")
    dates = dates[start_index:]
    opens = pd.DataFrame({pair: frame["open"] for pair, frame in daily.items()}).reindex(dates)
    results: list[dict] = []
    for segment in range(count):
        start = dates[segment * segment_days]
        end_index = (segment + 1) * segment_days
        segment_dates = dates[segment * segment_days:end_index]
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        previous = {pair: 0.0 for pair in opens.columns}
        turnover = 0.0
        daily_returns: list[float] = []
        for index, day in enumerate(segment_dates):
            next_day = dates[segment * segment_days + index + 1]
            # Signal is computed after the prior daily close and is held at
            # today's open. This avoids using today's open/high/low/close.
            target = targets.get(dates[dates.index(day) - 1], {})
            current = {pair: float(target.get(pair, 0.0)) for pair in opens.columns}
            turnover += sum(abs(current[pair] - previous[pair]) for pair in opens.columns)
            previous = current
            asset_returns = (opens.loc[next_day] / opens.loc[day] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            gross = sum(current[pair] * float(asset_returns[pair]) for pair in opens.columns)
            fee = sum(abs(current[pair] - (0.0 if index == 0 else 0.0)) for pair in opens.columns) * 0.0
            # Transaction cost is charged on every rebalance turnover. The
            # turnover above includes the prior target; reconstruct the prior
            # target fee directly from the target transition.
            prior_target = targets.get(dates[dates.index(day) - 2], {}) if dates.index(day) >= 2 else {}
            rebalance_turnover = sum(abs(current[pair] - float(prior_target.get(pair, 0.0))) for pair in opens.columns)
            if index == 0:
                rebalance_turnover = sum(abs(current[pair]) for pair in opens.columns)
            net = gross - rebalance_turnover * fee_rate
            fee = rebalance_turnover * fee_rate
            daily_returns.append(float(net))
            equity *= 1.0 + net
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
        results.append({
            "segment": segment + 1,
            "start": str(start.date()),
            "end": str((segment_dates[-1]).date()),
            "initial_equity": 100000.0,
            "final_equity": round(100000.0 * equity, 2),
            "return": float(equity - 1.0),
            "max_drawdown": float(max_drawdown),
            "turnover": float(turnover),
            "fee_rate": fee_rate,
            "rebalance_count": segment_days,
            "daily_net_returns": daily_returns,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["alphaagent_70"], default="alphaagent_70")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    daily = load_daily(Path("data/history"))
    results = simulate_segments(daily, factor_targets(daily, args.strategy, 5))
    payload = {
        "strategy": args.strategy,
        "assets": sorted(daily),
        "requested_top_n": 5,
        "effective_top_n": min(5, len(daily)),
        "fee_rate": 0.0005,
        "segment_days": 15,
        "segments": results,
        "summary": {
            "mean_return": float(np.mean([item["return"] for item in results])),
            "median_return": float(np.median([item["return"] for item in results])),
            "win_rate": float(np.mean([item["return"] > 0 for item in results])),
            "mean_max_drawdown": float(np.mean([item["max_drawdown"] for item in results])),
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
