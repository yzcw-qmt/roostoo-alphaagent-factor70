"""Causal, short-window research harness for Factor 70 entry/exit filters.

This script deliberately keeps research results separate from the live bot.
It uses Binance UTC candles and a caller-supplied Roostoo pair universe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


START = pd.Timestamp("2026-10-04", tz="UTC")
END = pd.Timestamp("2026-10-09", tz="UTC")
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def symbol_for_pair(pair: str) -> str:
    base, quote = pair.upper().split("/")
    if quote != "USD":
        raise ValueError(f"Expected a Roostoo /USD pair: {pair}")
    return f"{base}USDT"


def load_universe(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "TradePairs" in payload:
        pairs = [pair for pair, rules in payload["TradePairs"].items() if rules.get("CanTrade")]
    elif isinstance(payload, list):
        pairs = payload
    else:
        raise ValueError("Universe must be a JSON list or Roostoo exchangeInfo response")
    return sorted({pair.upper() for pair in pairs if pair.upper().endswith("/USD")})


def fetch_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> list[list]:
    rows: list[list] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        query = urlencode({
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1000,
        })
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(f"{BINANCE_URL}?{query}", timeout=15) as response:
                    batch = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code in {400, 404}:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:
            assert last_error is not None
            raise last_error
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < 1000:
            break
    return rows


def candle_frame(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame[0], unit="ms", utc=True)
    frame = frame.rename(columns={1: "open", 2: "high", 3: "low", 4: "close"})
    return frame[["open", "high", "low", "close"]].astype(float).sort_index()


def load_prices(pairs: list[str], cache_dir: Path, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    daily: dict[str, pd.DataFrame] = {}
    intraday: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for pair in pairs:
        symbol = symbol_for_pair(pair)
        frames = {}
        for interval, start, end in (
            ("1d", START - pd.Timedelta(days=45), END),
            ("30m", START, END),
        ):
            path = cache_dir / f"{symbol}_{interval}_2026-10-04_2026-10-08.json"
            try:
                if refresh or not path.exists():
                    rows = fetch_klines(symbol, interval, start, end)
                    path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
                else:
                    rows = json.loads(path.read_text(encoding="utf-8"))
                frames[interval] = candle_frame(rows)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                failures[pair] = f"{interval}: {exc}"
                break
        if pair not in failures:
            daily[pair] = frames["1d"]
            intraday[pair] = frames["30m"]
    return daily, intraday, failures


def score(closes: list[float]) -> float:
    """Match the live Factor 70: 20 daily differences over 21 closes."""
    if len(closes) < 21:
        return math.nan
    values = closes[-21:]
    changes = [new - old for old, new in zip(values, values[1:])]
    denominator = sum(abs(value) for value in changes)
    return sum(changes) / denominator if denominator else 0.0


def daily_snapshot(daily: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict[str, tuple[float, float, int]]:
    """Return completed-day score, one-day return, and pre-spike rank."""
    intermediate = {}
    cutoff = day - pd.Timedelta(days=1)
    for pair, frame in daily.items():
        history = frame.loc[:cutoff, "close"].dropna()
        if len(history) < 22 or history.index[-1] != cutoff:
            continue
        values = history.to_list()
        intermediate[pair] = (score(values), values[-1] / values[-2] - 1.0, score(values[:-1]))
    prior_order = sorted(intermediate, key=lambda pair: (-intermediate[pair][2], pair))
    prior_rank = {pair: rank for rank, pair in enumerate(prior_order, 1)}
    return {pair: (factor, rise, prior_rank[pair]) for pair, (factor, rise, _) in intermediate.items()}


@dataclass(frozen=True)
class Parameters:
    spike_limit: float | None = None
    min_score: float = -1.0
    exit_drop: float | None = None
    exit_anchor: str = "entry"  # first completed 30m score, or intraday peak
    min_qualifying: int = 1
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0
    market_exit_fee_rate: float = 0.001


def intraday_score(daily: pd.DataFrame, day: pd.Timestamp, current_close: float) -> float:
    history = daily.loc[:day - pd.Timedelta(days=1), "close"].dropna()
    if len(history) < 20:
        return math.nan
    return score(history.to_list()[-20:] + [current_close])


def select_targets(snapshot: dict[str, tuple[float, float, int]], params: Parameters) -> list[str]:
    current_order = sorted(snapshot, key=lambda pair: (-snapshot[pair][0], pair))
    current_rank = {pair: rank for rank, pair in enumerate(current_order, 1)}
    eligible = [
        (pair, factor)
        for pair, (factor, one_day_return, prior_rank) in snapshot.items()
        if factor >= params.min_score
        and (params.spike_limit is None or not (
            one_day_return > params.spike_limit
            and prior_rank > 3
            and current_rank[pair] <= 3
        ))
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [pair for pair, _ in eligible[:3]] if len(eligible) >= params.min_qualifying else []


@dataclass
class MarketData:
    bars: list[pd.Timestamp]
    snapshots: dict[pd.Timestamp, dict[str, tuple[float, float, int]]]
    opens: dict[pd.Timestamp, dict[str, float]]
    closes: dict[pd.Timestamp, dict[str, float]]
    scores: dict[pd.Timestamp, dict[str, float]]


def prepare_market(daily: dict[str, pd.DataFrame], intraday: dict[str, pd.DataFrame]) -> MarketData:
    expected = pd.date_range(START, END, freq="30min", inclusive="left")
    for pair, frame in intraday.items():
        if not expected.isin(frame.index).all():
            raise ValueError(f"Missing 30-minute bars for {pair}")
    bars = list(expected)
    snapshots = {day: daily_snapshot(daily, day) for day in bars[::48]}
    opens = {bar: {pair: float(frame.at[bar, "open"]) for pair, frame in intraday.items()} for bar in bars}
    closes = {bar: {pair: float(frame.at[bar, "close"]) for pair, frame in intraday.items()} for bar in bars}
    scores = {bar: {pair: intraday_score(daily[pair], bar.normalize(), closes[bar][pair]) for pair in intraday} for bar in bars}
    return MarketData(bars, snapshots, opens, closes, scores)


def simulate(market: MarketData, params: Parameters) -> dict:
    """Daily decisions at 00:00 UTC; intraday exits at the *next* 30m open."""
    common_bars = market.bars
    cash = 100000.0
    quantities: dict[str, float] = {}
    equity_curve: list[float] = [cash]
    turnover = 0.0
    fees = 0.0
    trades = 0
    exposure_sum = 0.0
    pending_exits: set[str] = set()
    anchors: dict[str, float] = {}
    day_targets: list[dict] = []
    trade_log: list[dict] = []

    def transact(pair: str, side: str, quantity: float, price: float, bar: pd.Timestamp, market_exit: bool = False) -> None:
        nonlocal cash, turnover, fees, trades
        if quantity <= 1e-12:
            return
        price *= 1.0 + params.slippage_rate if side == "BUY" else 1.0 - params.slippage_rate
        notional = quantity * price
        fee = notional * (params.market_exit_fee_rate if market_exit else params.fee_rate)
        cash += notional - fee if side == "SELL" else -notional - fee
        quantities[pair] = quantities.get(pair, 0.0) + (quantity if side == "BUY" else -quantity)
        if quantities[pair] < 1e-12:
            quantities.pop(pair, None)
        turnover += notional
        fees += fee
        trades += 1
        trade_log.append({"time": bar.isoformat(), "pair": pair, "side": side, "price": price, "quantity": quantity, "notional": notional, "fee": fee})

    for bar in common_bars:
        day = bar.normalize()
        open_prices = market.opens[bar]
        close_prices = market.closes[bar]
        if bar == day:
            snapshot = market.snapshots[day]
            targets = select_targets(snapshot, params)
            current_order = sorted(snapshot, key=lambda pair: (-snapshot[pair][0], pair))
            current_rank = {pair: rank for rank, pair in enumerate(current_order, 1)}
            day_targets.append({"day": str(day.date()), "eligible_count": sum(
                factor >= params.min_score and (params.spike_limit is None or not (
                    rise > params.spike_limit and prior_rank > 3 and current_rank[pair] <= 3
                ))
                for pair, (factor, rise, prior_rank) in snapshot.items()
            ), "targets": targets})
            equity = cash + sum(qty * open_prices[pair] for pair, qty in quantities.items())
            # Each asset gets one of three equal slots. Unfilled slots stay in USD.
            target_notional = equity / 3.0 if targets else 0.0
            for pair, quantity in list(quantities.items()):
                desired = target_notional / open_prices[pair] if pair in targets else 0.0
                if quantity > desired + 1e-12:
                    transact(pair, "SELL", quantity - desired, open_prices[pair], bar)
            for pair in targets:
                desired = target_notional / open_prices[pair]
                held = quantities.get(pair, 0.0)
                if desired > held + 1e-12:
                    affordable = cash / (open_prices[pair] * (1.0 + params.slippage_rate) * (1.0 + params.fee_rate))
                    transact(pair, "BUY", min(desired - held, affordable), open_prices[pair], bar)
            # The intraday score rolls the 20-day window forward. Compare it
            # with the first completed 30-minute score of this same day, not
            # yesterday's daily score, to avoid a mechanical rollover exit.
            anchors = {pair: market.scores[bar][pair] for pair in quantities}
            pending_exits.clear()
        else:
            for pair in sorted(pending_exits):
                if pair in quantities:
                    transact(pair, "SELL", quantities[pair], open_prices[pair], bar, market_exit=True)
                    anchors.pop(pair, None)
            pending_exits.clear()

        equity = cash + sum(qty * close_prices[pair] for pair, qty in quantities.items())
        exposure_sum += sum(qty * close_prices[pair] for pair, qty in quantities.items()) / equity
        equity_curve.append(equity)
        if params.exit_drop is not None and (bar.hour, bar.minute) != (23, 30):
            for pair in quantities:
                current = market.scores[bar][pair]
                if not math.isfinite(current):
                    continue
                reference = anchors[pair]
                if params.exit_anchor == "peak":
                    anchors[pair] = max(reference, current)
                    reference = anchors[pair]
                if current <= reference - params.exit_drop:
                    pending_exits.add(pair)

    final_equity = equity_curve[-1]
    day_end_equity = {str(day.date()): equity_curve[(index + 1) * 48] for index, day in enumerate(common_bars[::48])}
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve[1:]:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    return {
        "total_return": final_equity / 100000.0 - 1.0,
        "final_equity": final_equity,
        "max_drawdown_30m": max_drawdown,
        "traded_notional": turnover,
        "fees": fees,
        "trade_count": trades,
        "mean_exposure": exposure_sum / len(common_bars),
        "day_end_equity": day_end_equity,
        "day_targets": day_targets,
        "trade_log": trade_log,
    }


def sweep(market: MarketData, fee_rate: float) -> list[dict]:
    candidates = [Parameters(fee_rate=fee_rate)]
    for spike in (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30):
        for threshold in (0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.60):
            for anchor in ("entry", "peak"):
                for exit_drop in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20):
                    candidates.append(Parameters(spike, threshold, exit_drop, anchor, 1, fee_rate))
    rows = []
    for params in candidates:
        result = simulate(market, params)
        rows.append({**asdict(params), **{key: result[key] for key in (
            "total_return", "final_equity", "max_drawdown_30m", "traded_notional", "fees", "trade_count", "mean_exposure"
        )}})
    return sorted(rows, key=lambda row: (row["total_return"], row["max_drawdown_30m"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True, help="JSON list of exact eligible Roostoo /USD pairs")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/research/factor70_oct_candles"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/factor70_oct_window"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    args = parser.parse_args()
    pairs = load_universe(args.universe)
    daily, intraday, failures = load_prices(pairs, args.cache_dir, args.refresh)
    valid = [pair for pair in pairs if pair in daily and len(daily[pair].loc[:START - pd.Timedelta(days=1)]) >= 22 and len(intraday[pair].loc[START:END - pd.Timedelta(minutes=30)]) == 240]
    daily = {pair: daily[pair] for pair in valid}
    intraday = {pair: intraday[pair] for pair in valid}
    if len(valid) < 3:
        raise ValueError(f"Fewer than 3 assets have complete data: {len(valid)}")
    market = prepare_market(daily, intraday)
    rows = sweep(market, args.fee_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "requested_pairs": pairs,
        "usable_pairs": valid,
        "failures": failures,
        "start_utc": START.isoformat(),
        "end_exclusive_utc": END.isoformat(),
        "fee_rate": args.fee_rate,
    }, indent=2), encoding="utf-8")
    scenarios = {
        "original_top3": Parameters(fee_rate=args.fee_rate),
        "spike_only_5pct": Parameters(spike_limit=0.05, fee_rate=args.fee_rate),
        "score_only_0p56": Parameters(min_score=0.56, fee_rate=args.fee_rate),
        "entry_filters_only": Parameters(spike_limit=0.05, min_score=0.56, fee_rate=args.fee_rate),
        "entry_filters_and_exit_0p06": Parameters(0.05, 0.56, 0.06, "entry", 1, args.fee_rate),
    }
    diagnostics = {
        name: {"parameters": asdict(params), **simulate(market, params)}
        for name, params in scenarios.items()
    }
    diagnostics["stress"] = {
        f"fee_{fee:.4f}_slip_{slip:.4f}": {
            "entry_filters_only": simulate(market, Parameters(0.05, 0.56, None, "entry", 1, fee, slip))["total_return"],
            "entry_filters_and_exit_0p06": simulate(market, Parameters(0.05, 0.56, 0.06, "entry", 1, fee, slip))["total_return"],
        }
        for fee in (0.0005, 0.001)
        for slip in (0.0, 0.001, 0.002)
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": next(row for row in rows if row["spike_limit"] is None),
        "best_in_sample": rows[0],
        "provisional_risk_oriented": {
            key: diagnostics["entry_filters_and_exit_0p06"][key]
            for key in ("total_return", "max_drawdown_30m", "mean_exposure", "trade_count")
        },
        "usable_assets": len(valid),
    }, indent=2))


if __name__ == "__main__":
    main()
