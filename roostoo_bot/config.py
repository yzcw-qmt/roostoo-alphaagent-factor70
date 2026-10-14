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
    strategy: str = "alphaagent_70"
    order_type: str = "LIMIT"
    limit_offset_bps: float = 0.0

    factor_top_n: int = 3

    max_position_fraction: float = 0.30
    kelly_fraction: float = 0.30
    fallback_position_fraction: float = 0.10
    stop_loss: float = 0.0
    daily_loss_limit: float = 0.05
    max_drawdown: float = 0.12
    trailing_activation: float = 0.03
    trailing_gap: float = 0.025
    min_notional_buffer: float = 1.05

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
            strategy=os.getenv("STRATEGY", "alphaagent_70").strip().lower(),
            factor_top_n=int(os.getenv("FACTOR_TOP_N", "3")),
            order_type=os.getenv("ORDER_TYPE", "LIMIT").strip().upper(),
            limit_offset_bps=_float("LIMIT_OFFSET_BPS", 0.0),
        )

    def validate_live(self) -> None:
        if not self.api_key or not self.secret_key:
            raise ValueError("Live mode requires ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY")
        if self.api_key.startswith("YOUR_") or self.secret_key.startswith("YOUR_"):
            raise ValueError("Replace placeholder Roostoo credentials before live trading")
