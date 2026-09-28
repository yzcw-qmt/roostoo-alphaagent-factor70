from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from roostoo_bot.data_fetcher import BinanceDataFetcher
from roostoo_bot.portfolio_backtest import (
    build_portfolio_weights,
    random_non_overlapping_segments,
    simulate_segment,
)


def parse_args() -> argparse.Namespace:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    parser = argparse.ArgumentParser(description="Random 15-day multi-asset portfolio backtest")
    parser.add_argument("--pairs", nargs="+", default=["BTC/USD", "ETH/USD", "BNB/USD", "TON/USD", "TRX/USD"])
    parser.add_argument("--start", default=str((now - timedelta(days=365)).date()))
    parser.add_argument("--end", default=str(now.date()))
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--segment-days", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("reports/multi_asset_backtest.json"))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def load_pair(
    fetcher: BinanceDataFetcher,
    pair: str,
    download_start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{pair.replace('/', '-')}_1h_{download_start.date()}_{end.date()}.csv"
    path = cache_dir / filename
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
    else:
        frame = fetcher.historical_candles(pair, download_start, end, "1h")
        frame.reset_index().to_csv(path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if end <= start:
        raise SystemExit("--end must be after --start")
    download_start = start - timedelta(days=10)  # warm-up only; never included in reported returns
    fetcher = BinanceDataFetcher()
    closes: dict[str, pd.Series] = {}
    sources: dict[str, dict] = {}
    for pair in args.pairs:
        frame = load_pair(fetcher, pair, download_start, end, args.cache_dir, args.refresh)
        if frame.empty:
            raise SystemExit(f"No data returned for {pair}")
        closes[pair] = frame["close"].astype(float)
        sources[pair] = {"rows": len(frame), "first": str(frame.index.min()), "last": str(frame.index.max())}
    close_frame = pd.DataFrame(closes).sort_index().ffill().dropna()
    weights, diagnostics = build_portfolio_weights(close_frame)
    segments = random_non_overlapping_segments(start, end, args.segment_days, args.segments, args.seed)
    results = [
        simulate_segment(close_frame, weights, left, right, args.initial_capital, args.fee_rate)
        for left, right in segments
    ]
    returns = np.array([item.total_return for item in results])
    drawdowns = np.array([item.max_drawdown for item in results])
    composites = np.array([item.composite_score for item in results])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "source": "Binance public market-data-only API (data-api.binance.vision)",
            "interval": "1h",
            "period_start": str(start),
            "period_end": str(end),
            "segment_days": args.segment_days,
            "requested_segments": args.segments,
            "selected_segments": len(results),
            "random_seed": args.seed,
            "selection": "random without replacement from non-overlapping fixed-length candidates, before scoring",
            "initial_capital_each": args.initial_capital,
            "fee_rate": args.fee_rate,
            "rebalance_frequency": "12h",
            "minimum_portfolio_change": 0.15,
            "signal_policy": "stay flat unless strong thresholds are met",
            "pairs": args.pairs,
        },
        "data": sources,
        "cross_asset_diagnostics": diagnostics,
        "aggregate": {
            "mean_return": float(returns.mean()),
            "median_return": float(np.median(returns)),
            "positive_segment_rate": float((returns > 0).mean()),
            "worst_return": float(returns.min()),
            "best_return": float(returns.max()),
            "mean_max_drawdown": float(drawdowns.mean()),
            "worst_max_drawdown": float(drawdowns.min()),
            "median_composite_score": float(np.median(composites)),
        },
        "segments": [item.to_dict() for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["method"], indent=2, ensure_ascii=False))
    print(json.dumps(report["cross_asset_diagnostics"], indent=2, ensure_ascii=False))
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    print(f"Full report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
