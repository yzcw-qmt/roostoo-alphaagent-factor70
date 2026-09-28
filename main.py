from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace

from roostoo_bot.api_client import RoostooClient
from roostoo_bot.config import Settings
from roostoo_bot.data_fetcher import BinanceDataFetcher, BinancePaperClient
from roostoo_bot.demo import DemoDataFetcher, DemoRoostooClient
from roostoo_bot.engine import TradingEngine


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Roostoo risk-first autonomous trading bot")
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one cycle or the hourly loop")
    run.add_argument("--once", action="store_true", help="Run exactly one cycle")
    run.add_argument("--live", action="store_true", help="Enable real API orders (credentials required)")
    sub.add_parser("demo", help="Run one fully offline dry-run cycle with synthetic candles")
    paper = sub.add_parser("paper", help="Use Binance public data without Roostoo or API keys")
    paper.add_argument("--loop", action="store_true", help="Continue at the configured hourly interval")
    sub.add_parser("check-api", help="Check public server time and exchange information")
    return result


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()

    if args.command == "demo":
        settings = replace(settings, dry_run=True)
        engine = TradingEngine(settings, DemoRoostooClient(settings.pairs), DemoDataFetcher())
        print(json.dumps(engine.cycle(), indent=2, ensure_ascii=False))
        return
    if args.command == "paper":
        settings = replace(settings, dry_run=True)
        fetcher = BinanceDataFetcher(settings.timeout_seconds)
        engine = TradingEngine(settings, BinancePaperClient(settings.pairs, fetcher), fetcher)
        if args.loop:
            engine.run_forever()
        else:
            print(json.dumps(engine.cycle(), indent=2, ensure_ascii=False))
        return
    if args.command == "check-api":
        client = RoostooClient(base_url=settings.base_url)
        print(json.dumps({"time_offset_ms": client.sync_time(), "exchange": client.exchange_info()}, indent=2))
        return

    settings = replace(settings, dry_run=not args.live)
    if args.live:
        settings.validate_live()
    engine = TradingEngine(settings)
    if args.once:
        print(json.dumps(engine.cycle(), indent=2, ensure_ascii=False))
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
