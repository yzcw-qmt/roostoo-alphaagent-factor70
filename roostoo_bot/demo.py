from __future__ import annotations

import numpy as np
import pandas as pd


def sample_candles(seed: int = 7, periods: int = 240, start_price: float = 50000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00035, 0.006, periods)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.003, 0.001, periods))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(8.0, 0.4, periods)
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=periods, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


class DemoDataFetcher:
    def candles(self, pair: str, interval: str = "1h", limit: int = 240) -> pd.DataFrame:
        seed = sum(ord(char) for char in pair)
        price = 50000.0 if pair.startswith("BTC") else 2500.0
        return sample_candles(seed, limit, price)


class DemoRoostooClient:
    def __init__(self, pairs: tuple[str, ...]) -> None:
        self.pairs = pairs

    def exchange_info(self) -> dict:
        return {
            "IsRunning": True,
            "InitialWallet": {"USD": 100000},
            "TradePairs": {
                pair: {
                    "Coin": pair.split("/")[0], "Unit": "USD", "CanTrade": True,
                    "PricePrecision": 2, "AmountPrecision": 6, "MiniOrder": 1.0,
                }
                for pair in self.pairs
            },
        }

    def ticker(self, pair: str | None = None) -> dict:
        selected = (pair,) if pair else self.pairs
        data = {}
        for item in selected:
            last = float(DemoDataFetcher().candles(item)["close"].iloc[-1])
            data[item] = {"MaxBid": last * 0.9999, "MinAsk": last * 1.0001, "LastPrice": last, "Change": 0.01}
        return {"Success": True, "Data": data}

    def balance(self) -> dict:
        return {"Success": True, "Wallet": {"USD": {"Free": 100000.0, "Lock": 0.0}}}

    def short_positions(self) -> dict:
        return {"Success": True, "Positions": []}

