from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://mock-api.roostoo.com"
    pairs: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    interval: str = "30m"
    candle_limit: int = 400
    loop_interval_seconds: int = 1800
    dry_run: bool = True
    request_limit_per_minute: int = 28
    timeout_seconds: float = 15.0
    retries: int = 3
    # Final selected strategy: 60% XGBoost + 40% causal MA/MACD score.
    strategy: str = "ma_macd_xgb"
    order_type: str = "LIMIT"
    limit_offset_bps: float = 0.0

    long_threshold: float = 0.62
    short_threshold: float = 0.38
    exit_long_threshold: float = 0.48
    exit_short_threshold: float = 0.52
    model_weight: float = 0.60
    technical_weight: float = 0.40

    max_position_fraction: float = 0.30
    max_gross_exposure: float = 0.90
    max_net_exposure: float = 0.60
    kelly_fraction: float = 0.30
    fallback_position_fraction: float = 0.10
    stop_loss: float = 0.0
    daily_loss_limit: float = 0.05
    max_drawdown: float = 0.12
    trailing_activation: float = 0.03
    trailing_gap: float = 0.025
    min_notional_buffer: float = 1.05
    cooldown_minutes: int = 55
    macd_extrema_lookback: int = 20
    atr_period: int = 14
    extrema_k: float = 1.5
    extrema_risk_pct: float = 0.02
    dynamic_extrema_n: bool = False
    correlation_window: int = 168
    correlation_threshold: float = 0.70
    same_direction_risk_multiplier: float = 0.85
    opposite_direction_risk_multiplier: float = 0.75

    model_path: Path = Path("models/xgb_30m_multi.json")
    btc_model_path: Path = Path("models/xgb_btc_30m_180d.json")
    eth_model_path: Path = Path("models/xgb_eth_30m_120d.json")
    state_path: Path = Path("data/state.json")
    log_dir: Path = Path("logs")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(env_file, override=False)
        pairs = tuple(
            value.strip().upper()
            for value in os.getenv("TRADING_PAIRS", "BTC/USD,ETH/USD").split(",")
            if value.strip()
        )
        return cls(
            api_key=os.getenv("ROOSTOO_API_KEY", ""),
            secret_key=os.getenv("ROOSTOO_SECRET_KEY", ""),
            base_url=os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").rstrip("/"),
            pairs=pairs,
            dry_run=_bool("DRY_RUN", True),
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "1800")),
            strategy=os.getenv("STRATEGY", "ma_macd_xgb").strip().lower(),
            order_type=os.getenv("ORDER_TYPE", "LIMIT").strip().upper(),
            limit_offset_bps=_float("LIMIT_OFFSET_BPS", 0.0),
            long_threshold=_float("LONG_THRESHOLD", 0.62),
            short_threshold=_float("SHORT_THRESHOLD", 0.38),
            model_weight=_float("MODEL_WEIGHT", 0.60),
            model_path=Path(os.getenv("MODEL_PATH", "models/xgb_30m_multi.json")),
            btc_model_path=Path(os.getenv("BTC_MODEL_PATH", "models/xgb_btc_30m_180d.json")),
            eth_model_path=Path(os.getenv("ETH_MODEL_PATH", "models/xgb_eth_30m_120d.json")),
        )

    def validate_live(self) -> None:
        if not self.api_key or not self.secret_key:
            raise ValueError("Live mode requires ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY")
        if self.api_key.startswith("YOUR_") or self.secret_key.startswith("YOUR_"):
            raise ValueError("Replace placeholder Roostoo credentials before live trading")
