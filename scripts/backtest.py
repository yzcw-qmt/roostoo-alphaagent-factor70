from __future__ import annotations

import argparse
import math

import numpy as np

from roostoo_bot.data_fetcher import BinanceDataFetcher
from roostoo_bot.features import build_features
from roostoo_bot.model import DirectionModel
from roostoo_bot.signals import technical_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple fee-aware chronological signal backtest")
    parser.add_argument("--pair", default="BTC/USD")
    parser.add_argument("--candles", type=int, default=1000)
    args = parser.parse_args()
    candles = BinanceDataFetcher().candles(args.pair, "1h", args.candles)
    features = build_features(candles)
    model = DirectionModel("models/xgb_btc_1h.json")
    probabilities = []
    technical = []
    for end in range(120, len(candles)):
        current = features.iloc[: end + 1]
        probabilities.append(model.predict_probability(current))
        technical.append(technical_score(current))
    score = 0.6 * np.array(probabilities) + 0.4 * np.array(technical)
    position = np.where(score > 0.65, 1.0, np.where(score < 0.35, -1.0, 0.0))
    returns = candles["close"].pct_change().iloc[120:].to_numpy()
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    strategy = np.roll(position, 1) * returns - turnover * 0.001
    strategy[0] = 0
    equity = np.cumprod(1 + np.nan_to_num(strategy))
    hourly = np.nan_to_num(strategy)
    downside = hourly[hourly < 0]
    sharpe = hourly.mean() / (hourly.std() or 1) * math.sqrt(24 * 365)
    sortino = hourly.mean() / (downside.std() if len(downside) and downside.std() else 1) * math.sqrt(24 * 365)
    drawdown = equity / np.maximum.accumulate(equity) - 1
    annualized = equity[-1] ** (24 * 365 / max(len(equity), 1)) - 1
    calmar = annualized / abs(drawdown.min()) if drawdown.min() < 0 else 0
    print({"return": float(equity[-1] - 1), "max_drawdown": float(drawdown.min()), "sharpe": sharpe, "sortino": sortino, "calmar": calmar})


if __name__ == "__main__":
    main()

