from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "sma_ratio", "ema_ratio", "donchian_position", "adx", "candle_position",
    "rsi", "roc", "williams_r", "volume_ratio", "bb_width", "atr_ratio",
    "volatility", "return_6h", "trend_4h", "trend_24h",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return pd.concat(
        [(frame["high"] - frame["low"]), (frame["high"] - previous).abs(), (frame["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)


def _adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _true_range(frame).ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if missing := required - set(candles.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = candles.copy().sort_index()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    sma10, sma30 = close.rolling(10).mean(), close.rolling(30).mean()
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    highest20, lowest20 = high.rolling(20).max(), low.rolling(20).min()
    middle = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    atr = _true_range(frame).ewm(alpha=1 / 14, adjust=False).mean()

    out = pd.DataFrame(index=frame.index)
    out["sma_ratio"] = sma10 / sma30 - 1
    out["ema_ratio"] = ema12 / ema26 - 1
    out["donchian_position"] = (close - lowest20) / (highest20 - lowest20).replace(0, np.nan)
    out["adx"] = _adx(frame) / 100
    out["candle_position"] = (close - low) / (high - low).replace(0, np.nan)
    out["rsi"] = _rsi(close) / 100
    out["roc"] = close.pct_change(10)
    out["williams_r"] = (highest20 - close) / (highest20 - lowest20).replace(0, np.nan)
    out["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    out["bb_width"] = (4 * std20) / middle.replace(0, np.nan)
    out["atr_ratio"] = atr / close.replace(0, np.nan)
    out["volatility"] = close.pct_change().rolling(20).std()
    out["return_6h"] = close.pct_change(6)
    out["trend_4h"] = close / close.shift(4) - 1
    out["trend_24h"] = close / close.shift(24) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def make_training_set(candles: pd.DataFrame, horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    features = build_features(candles)
    future_return = candles["close"].shift(-horizon) / candles["close"] - 1
    valid = features.notna().all(axis=1) & future_return.notna()
    return features.loc[valid, FEATURE_COLUMNS], (future_return.loc[valid] > 0).astype(int)

