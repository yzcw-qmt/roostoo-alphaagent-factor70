from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import deque
from typing import Any, Mapping
from urllib.parse import urlencode

import requests


class RoostooAPIError(RuntimeError):
    pass


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int = 28, period_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period_seconds - (now - self._calls[0]) + 0.01
            time.sleep(max(wait, 0.01))


class RoostooClient:
    """Official Roostoo REST client using sorted HMAC-SHA256 form payloads."""

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        base_url: str = "https://mock-api.roostoo.com",
        request_limit_per_minute: int = 28,
        timeout: float = 15.0,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.limiter = SlidingWindowRateLimiter(request_limit_per_minute)
        self.time_offset_ms = 0
        self.log = logging.getLogger(__name__)

    @staticmethod
    def canonical_payload(params: Mapping[str, Any]) -> str:
        normalized = {str(key): str(value) for key, value in params.items() if value is not None}
        return urlencode(sorted(normalized.items()), safe="/")

    def sign(self, params: Mapping[str, Any]) -> str:
        payload = self.canonical_payload(params)
        return hmac.new(self.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000) + self.time_offset_ms)

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        signed: bool = False,
        timestamped: bool = False,
    ) -> dict[str, Any]:
        payload = dict(params or {})
        if signed or timestamped:
            payload["timestamp"] = self._timestamp()
        encoded = self.canonical_payload(payload)
        headers: dict[str, str] = {}
        if signed:
            if not self.api_key or not self.secret_key:
                raise RoostooAPIError("Signed endpoint requires API credentials")
            headers.update({"RST-API-KEY": self.api_key, "MSG-SIGNATURE": self.sign(payload)})
        if method.upper() == "POST":
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        error: Exception | None = None
        for attempt in range(self.retries):
            self.limiter.acquire()
            try:
                response = self.session.request(
                    method.upper(),
                    f"{self.base_url}{path}",
                    params=payload if method.upper() == "GET" else None,
                    data=encoded if method.upper() == "POST" else None,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("Success") is False:
                    raise RoostooAPIError(result.get("ErrMsg") or f"Roostoo rejected {path}")
                return result
            except (requests.RequestException, ValueError, RoostooAPIError) as exc:
                error = exc
                if attempt + 1 >= self.retries or isinstance(exc, RoostooAPIError):
                    break
                time.sleep(2**attempt)
        raise RoostooAPIError(f"{method.upper()} {path} failed: {error}") from error

    def sync_time(self) -> int:
        before = int(time.time() * 1000)
        result = self._request("GET", "/v3/serverTime")
        after = int(time.time() * 1000)
        server = int(result["ServerTime"])
        self.time_offset_ms = server - ((before + after) // 2)
        return self.time_offset_ms

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/v3/ticker", {"pair": pair}, timestamped=True)

    def balance(self) -> dict[str, Any]:
        return self._request("GET", "/v3/balance", signed=True)

    def pending_count(self) -> dict[str, Any]:
        return self._request("GET", "/v3/pending_count", signed=True)

    def place_order(
        self, pair: str, side: str, quantity: float, order_type: str = "MARKET", price: float | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            payload["price"] = price
        return self._request("POST", "/v3/place_order", payload, signed=True)

    def query_order(self, **filters: Any) -> dict[str, Any]:
        allowed = {"order_id", "pair", "offset", "limit", "pending_only"}
        return self._request("POST", "/v3/query_order", {k: v for k, v in filters.items() if k in allowed}, signed=True)

    def cancel_order(self, order_id: int | None = None, pair: str | None = None) -> dict[str, Any]:
        if order_id is not None and pair is not None:
            raise ValueError("Pass order_id or pair, not both")
        return self._request("POST", "/v3/cancel_order", {"order_id": order_id, "pair": pair}, signed=True)

    def short_open(
        self, pair: str, collateral: float, order_type: str = "MARKET", price: float | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"pair": pair, "collateral": collateral}
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT short requires price")
            payload.update({"order_type": "LIMIT", "price": price})
        return self._request("POST", "/v6/short_open", payload, signed=True)

    def short_close(
        self, pair: str, close_qty: float | None = None, close_pct: float | None = None
    ) -> dict[str, Any]:
        if close_qty is not None and close_pct is not None:
            raise ValueError("Pass close_qty or close_pct, not both")
        return self._request(
            "POST", "/v6/short_close", {"pair": pair, "close_qty": close_qty, "close_pct": close_pct}, signed=True
        )

    def short_positions(self) -> dict[str, Any]:
        return self._request("GET", "/v6/short_positions", signed=True)

