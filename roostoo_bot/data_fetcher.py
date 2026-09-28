from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests


class BinanceDataFetcher:
    """Fetches public Binance candles; no Binance credentials are required."""

    # Binance's market-data-only host exposes public endpoints without API keys.
    URL = "https://data-api.binance.vision/api/v3/klines"
    TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"

    def __init__(self, timeout: float = 15.0, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.log = logging.getLogger(__name__)

    @staticmethod
    def to_symbol(pair: str) -> str:
        base, quote = pair.upper().split("/")
        return f"{base}{'USDT' if quote == 'USD' else quote}"

    def candles(self, pair: str, interval: str = "1h", limit: int = 240) -> pd.DataFrame:
        response = self.session.get(
            self.URL,
            params={"symbol": self.to_symbol(pair), "interval": interval, "limit": min(limit, 1000)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows: list[list[Any]] = response.json()
        columns = [
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trade_count", "taker_base", "taker_quote", "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        for column in ("open", "high", "low", "close", "volume", "quote_volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        return frame.set_index("open_time")

    def historical_candles(
        self,
        pair: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Download a paginated, closed-open [start, end) OHLCV range."""
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            response = self.session.get(
                self.URL,
                params={
                    "symbol": self.to_symbol(pair),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            batch: list[list[Any]] = response.json()
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) == 1000:
                time.sleep(0.05)
        columns = [
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trade_count", "taker_base", "taker_quote", "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return pd.DataFrame(columns=columns).set_index("open_time")
        for column in ("open", "high", "low", "close", "volume", "quote_volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        return frame.drop_duplicates("open_time").set_index("open_time").sort_index()

    def ticker(self, pair: str) -> dict[str, float]:
        response = self.session.get(
            self.TICKER_URL,
            params={"symbol": self.to_symbol(pair)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "MaxBid": float(data["bidPrice"]),
            "MinAsk": float(data["askPrice"]),
            "LastPrice": float(data["lastPrice"]),
            "Change": float(data["priceChangePercent"]) / 100,
            "CoinTradeValue": float(data["volume"]),
            "UnitTradeValue": float(data["quoteVolume"]),
        }


class BinancePaperClient:
    """Read-only Roostoo-shaped adapter backed by Binance public data."""

    def __init__(self, pairs: tuple[str, ...], fetcher: BinanceDataFetcher | None = None) -> None:
        self.pairs = pairs
        self.fetcher = fetcher or BinanceDataFetcher()

    def exchange_info(self) -> dict[str, Any]:
        return {
            "IsRunning": True,
            "InitialWallet": {"USD": 100000.0},
            "TradePairs": {
                pair: {
                    "Coin": pair.split("/")[0],
                    "Unit": "USD",
                    "CanTrade": True,
                    "PricePrecision": 2,
                    "AmountPrecision": 6,
                    "MiniOrder": 1.0,
                }
                for pair in self.pairs
            },
        }

    def ticker(self, pair: str | None = None) -> dict[str, Any]:
        selected = (pair,) if pair else self.pairs
        return {"Success": True, "Data": {item: self.fetcher.ticker(item) for item in selected}}

    def balance(self) -> dict[str, Any]:
        return {"Success": True, "Wallet": {"USD": {"Free": 100000.0, "Lock": 0.0}}}

    def short_positions(self) -> dict[str, Any]:
        return {"Success": True, "Positions": []}
