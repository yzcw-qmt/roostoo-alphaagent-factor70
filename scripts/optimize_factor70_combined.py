"""Re-score the existing Factor 70 sweep using the competition metrics.

All results are in-sample for 4–8 October 2026 and use Binance price proxies.
Daily rebalances pay the maker fee; intraday market exits pay the taker fee.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import pandas as pd

from factor70_oct_window import Parameters, load_prices, load_universe, prepare_market, simulate, START, END


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/factor70_oct_window"


def metrics(result: dict) -> dict:
    equity = [100000.0, *result["day_end_equity"].values()]
    daily = [new / old - 1.0 for old, new in zip(equity, equity[1:])]
    average = statistics.mean(daily)
    volatility = statistics.stdev(daily)
    downside = math.sqrt(sum(min(0.0, value) ** 2 for value in daily) / len(daily))
    peak = equity[0]
    drawdown = 0.0
    for value in equity[1:]:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    annualized = (equity[-1] / equity[0]) ** (365.0 / len(daily)) - 1.0
    sharpe = average / volatility * math.sqrt(365) if volatility else None
    sortino = average / downside * math.sqrt(365) if downside else None
    calmar = annualized / abs(drawdown) if drawdown else None
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar if None not in (sortino, sharpe, calmar) else None
    market_exits = sum(
        trade["side"] == "SELL" and trade["time"][11:16] != "00:00"
        for trade in result["trade_log"]
    )
    return {
        "return": result["total_return"],
        "sortino": sortino,
        "sharpe": sharpe,
        "calmar": calmar,
        "composite": composite,
        "max_drawdown_daily": drawdown,
        "max_drawdown_30m": result["max_drawdown_30m"],
        "final_equity": result["final_equity"],
        "fees": result["fees"],
        "trade_count": result["trade_count"],
        "market_exits": market_exits,
        "day_end_equity": result["day_end_equity"],
    }


def main() -> None:
    universe = load_universe(ROOT / "data/research/eligible_pairs_88_2026-10-19.json")
    daily, intraday, _ = load_prices(universe, ROOT / "data/research/factor70_oct_candles")
    valid = [
        pair for pair in universe
        if pair in daily
        and len(daily[pair].loc[:START - pd.Timedelta(days=1)]) >= 22
        and len(intraday[pair].loc[START:END - pd.Timedelta(minutes=30)]) == 240
    ]
    market = prepare_market({pair: daily[pair] for pair in valid}, {pair: intraday[pair] for pair in valid})
    old_sweep = list(csv.DictReader((REPORT / "sweep.csv").open()))
    choices = {(float(row["spike_limit"]), float(row["min_score"]), float(row["exit_drop"]), row["exit_anchor"])
               for row in old_sweep if row["spike_limit"] and row["exit_drop"]}
    # A finer local search around the previously selected entry settings.
    choices.update((spike, score, drop, anchor)
                   for spike in (0.025, 0.035, 0.045, 0.055, 0.065)
                   for score in (0.545, 0.555, 0.565, 0.575)
                   for drop in (0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075)
                   for anchor in ("entry", "peak"))
    choices.update((spike / 10000, score / 10000, drop / 10000, "entry")
                   for spike in range(25, 201, 25)
                   for score in range(5500, 5701, 25)
                   for drop in range(100, 401, 25))
    choices.update((spike, score / 1000, drop / 1000, "entry")
                   for spike in (0.02, 0.03, 0.05, 0.07)
                   for score in range(580, 701, 5)
                   for drop in range(10, 41, 5))
    results = []
    for spike, score, drop, anchor in sorted(choices):
        params = Parameters(spike, score, drop, anchor)
        outcome = simulate(market, params)
        results.append({"spike_limit": spike, "min_score": score, "exit_drop": drop,
                        "exit_anchor": anchor, **metrics(outcome)})
    with (REPORT / "combined_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [key for key in results[0] if key != "day_end_equity"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in results)
    active = [row for row in results if row["market_exits"] and row["composite"] is not None]
    representative_params = Parameters(0.05, 0.60, 0.02)
    representative = {"parameters": {"spike_limit": 0.05, "min_score": 0.60,
                                      "exit_drop": 0.02, "exit_anchor": "entry"},
                      **metrics(simulate(market, representative_params))}
    no_exit = {"parameters": {"spike_limit": 0.05, "min_score": 0.56, "exit_drop": None},
               **metrics(simulate(market, Parameters(0.05, 0.56)))}
    payload = {
        "assumptions": {"maker_fee": 0.0005, "taker_market_exit_fee": 0.001,
                        "slippage": 0.0, "days": 5, "annualization_days": 365,
                        "risk_free_rate": 0.0, "eligible_pairs": len(valid)},
        "tested": len(results), "effective_exit_tested": len(active),
        "best_active_by_composite": max(active, key=lambda row: row["composite"]),
        "best_active_by_return": max(active, key=lambda row: row["return"]),
        "best_active_by_30m_drawdown": max(active, key=lambda row: (row["max_drawdown_30m"], row["return"])),
        "best_active_spike_2pct_or_more": max((row for row in active if row["spike_limit"] >= 0.02),
                                               key=lambda row: row["composite"]),
        "best_active_spike_5pct_or_more": max((row for row in active if row["spike_limit"] >= 0.05),
                                               key=lambda row: row["composite"]),
        "representative_best_composite": representative,
        "representative_ablations": {
            "without_spike_filter": metrics(simulate(market, Parameters(None, 0.60, 0.02))),
            "without_score_floor": metrics(simulate(market, Parameters(0.05, -1.0, 0.02))),
            "without_intraday_exit": metrics(simulate(market, Parameters(0.05, 0.60))),
        },
        "reference_no_exit": no_exit,
        "representative_slippage_stress": {
            str(slip): metrics(simulate(market, Parameters(0.05, 0.60, 0.02, "entry", 1, 0.0005, slip, 0.001)))
            for slip in (0.001, 0.002)
        },
        "top_active_by_composite": sorted(active, key=lambda row: row["composite"], reverse=True)[:20],
    }
    assert abs(representative["composite"] - payload["best_active_by_composite"]["composite"]) < 1e-9
    (REPORT / "combined_optimization.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("top_active_by_composite",)}, indent=2))


if __name__ == "__main__":
    main()
