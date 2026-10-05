from __future__ import annotations

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any

from .api_client import RoostooClient
from .journal import Journal


def round_down(value: float, precision: int) -> float:
    step = Decimal(1).scaleb(-precision)
    return float(Decimal(str(value)).quantize(step, rounding=ROUND_DOWN))


def round_up(value: float, precision: int) -> float:
    """Round a price upward to avoid making a marketable buy non-marketable."""
    step = Decimal(1).scaleb(-precision)
    return float(Decimal(str(value)).quantize(step, rounding=ROUND_UP))


class TradeExecutor:
    def __init__(self, client: RoostooClient, journal: Journal, dry_run: bool = True) -> None:
        self.client = client
        self.journal = journal
        self.dry_run = dry_run
        self.log = logging.getLogger(__name__)

    def _execute(self, kind: str, call: Any, **details: Any) -> dict[str, Any]:
        if self.dry_run:
            result = {"Success": True, "DryRun": True, **details}
            self.journal.write("dry_run_order", kind=kind, **details)
            self.log.info("DRY RUN %s %s", kind, details)
            return result
        try:
            result = call()
            detail = result.get("OrderDetail", {}) if isinstance(result, dict) else {}
            if isinstance(detail, dict) and detail.get("Role") == "TAKER":
                self.log.warning(
                    "Order matched as TAKER despite LIMIT type: %s %s",
                    kind,
                    details,
                )
            self.journal.write("order_success", kind=kind, request=details, response=result)
            return result
        except Exception as exc:
            self.journal.write("order_failure", kind=kind, request=details, error=str(exc))
            raise

    def buy(self, pair: str, quantity: float) -> dict[str, Any]:
        return self._execute("BUY", lambda: self.client.place_order(pair, "BUY", quantity), pair=pair, quantity=quantity)

    def sell(self, pair: str, quantity: float) -> dict[str, Any]:
        return self._execute("SELL", lambda: self.client.place_order(pair, "SELL", quantity), pair=pair, quantity=quantity)

    def buy_limit(self, pair: str, quantity: float, price: float) -> dict[str, Any]:
        return self._execute("BUY_LIMIT", lambda: self.client.place_order(pair, "BUY", quantity, "LIMIT", price), pair=pair, quantity=quantity, price=price)

    def sell_limit(self, pair: str, quantity: float, price: float) -> dict[str, Any]:
        return self._execute("SELL_LIMIT", lambda: self.client.place_order(pair, "SELL", quantity, "LIMIT", price), pair=pair, quantity=quantity, price=price)

    def open_short(self, pair: str, collateral: float) -> dict[str, Any]:
        return self._execute(
            "SHORT_OPEN", lambda: self.client.short_open(pair, collateral), pair=pair, collateral=collateral
        )

    def open_short_limit(self, pair: str, collateral: float, price: float) -> dict[str, Any]:
        return self._execute("SHORT_OPEN_LIMIT", lambda: self.client.short_open(pair, collateral, "LIMIT", price), pair=pair, collateral=collateral, price=price)

    def close_short(self, pair: str) -> dict[str, Any]:
        return self._execute("SHORT_CLOSE", lambda: self.client.short_close(pair), pair=pair)
