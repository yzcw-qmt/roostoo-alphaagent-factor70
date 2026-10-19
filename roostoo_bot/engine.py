from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from .api_client import RoostooClient, RoostooAPIError
from .config import Settings
from .data_fetcher import BinanceDataFetcher
from .executor import TradeExecutor, round_down, round_up
from .factor_strategy import CrossSectionalFactorStrategy, FactorSignal
from .journal import Journal, StateStore
from .risk_manager import RiskManager


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        client: RoostooClient | None = None,
        data: BinanceDataFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or RoostooClient(
            settings.api_key,
            settings.secret_key,
            settings.base_url,
            settings.request_limit_per_minute,
            settings.timeout_seconds,
            settings.retries,
        )
        self.data = data or BinanceDataFetcher(settings.timeout_seconds)
        self.journal = Journal(settings.log_dir)
        self.store = StateStore(settings.state_path)
        self.bot_state = self.store.load()
        self.risk = RiskManager(
            settings.max_position_fraction,
            settings.fallback_position_fraction,
            settings.kelly_fraction,
            settings.stop_loss,
            settings.daily_loss_limit,
            settings.max_drawdown,
            settings.trailing_activation,
            settings.trailing_gap,
        )
        self.factor_strategy = CrossSectionalFactorStrategy(settings.strategy, settings.factor_top_n)
        self.executor = TradeExecutor(self.client, self.journal, settings.dry_run)
        self.log = logging.getLogger(__name__)

    @staticmethod
    def _ticker_map(payload: dict[str, Any]) -> dict[str, Any]:
        return payload.get("Data", payload)

    @staticmethod
    def _wallet(payload: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized wallet, never ``None``.

        Roostoo can return ``Wallet: null`` when the portfolio is not
        initialized or when the credential is for a different portfolio.
        Treating that value as an empty, valid wallet would make the bot
        report equity 0 and continue as if the account were tradable.
        """
        if not isinstance(payload, dict):
            return {}
        wallet = payload.get("Wallet")
        if isinstance(wallet, dict):
            return wallet
        spot_wallet = payload.get("SpotWallet")
        if isinstance(spot_wallet, dict):
            return spot_wallet
        data = payload.get("Data")
        if isinstance(data, dict) and isinstance(data.get("Wallet"), dict):
            return data["Wallet"]
        if isinstance(data, dict) and isinstance(data.get("SpotWallet"), dict):
            return data["SpotWallet"]
        return {}

    @staticmethod
    def _wallet_present(payload: dict[str, Any]) -> bool:
        """Whether the API supplied an actual wallet object."""
        if not isinstance(payload, dict):
            return False
        if isinstance(payload.get("Wallet"), dict):
            return True
        if isinstance(payload.get("SpotWallet"), dict):
            return True
        data = payload.get("Data")
        return isinstance(data, dict) and (
            isinstance(data.get("Wallet"), dict)
            or isinstance(data.get("SpotWallet"), dict)
        )

    @staticmethod
    def _short_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        positions = payload.get("Positions")
        if positions is None and isinstance(payload.get("Data"), dict):
            positions = payload["Data"].get("Positions")
        return positions if isinstance(positions, list) else []

    def _equity(self, wallet: dict[str, Any], tickers: dict[str, Any], shorts: list[dict[str, Any]]) -> float:
        equity = sum(float(v.get("Free", 0)) + float(v.get("Lock", 0)) for k, v in wallet.items() if k == "USD")
        for coin, balance in wallet.items():
            pair = f"{coin}/USD"
            if coin != "USD" and pair in tickers:
                quantity = float(balance.get("Free", 0)) + float(balance.get("Lock", 0))
                equity += quantity * float(tickers[pair]["LastPrice"])
        equity += sum(float(position.get("PositionValue", 0)) for position in shorts)
        return equity

    def _factor_rebalance(
        self,
        signals: list[FactorSignal],
        wallet: dict[str, Any],
        rules_all: dict[str, Any],
        tickers: dict[str, Any],
        equity: float,
        pending_pairs: set[str] | None = None,
    ) -> None:
        """Rebalance a daily long-only top-N factor portfolio with LIMIT orders."""
        if not signals or equity <= 0:
            return
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        state = self.bot_state.setdefault("factor_rebalance", {})
        retry = bool(state.get("retry"))
        last_attempt = float(state.get("last_attempt", 0.0) or 0.0)
        # Daily candles are UTC-calendar candles.  The backtest uses the
        # previous completed daily candle and rebalances at the next candle
        # open: UTC 00:00 (08:00 Hong Kong time).  A short window handles
        # normal loop jitter while preventing a late-starting bot from
        # silently rebalancing at an arbitrary time of day.
        if (now.hour != 0 or now.minute >= 30) and not retry:
            return
        # The aligned half-hour loop can run a few seconds before the exact
        # 30-minute mark. Leave tolerance so an exit-recovery retry is not
        # skipped at the next scheduled cycle.
        retry_cooldown = max(60, self.settings.loop_interval_seconds - 120)
        if retry and now.timestamp() - last_attempt < retry_cooldown:
            return
        if state.get("date") == today and not retry:
            return
        selected = [signal for signal in signals if signal.selected]
        if not selected:
            return
        target_notional = equity / len(selected)
        selected_pairs = {signal.pair for signal in selected}
        pending_pairs = pending_pairs or set()
        available_usd = float(wallet.get("USD", {}).get("Free", 0))
        # Keep room for the 0.05% Maker commission and small price movement.
        cash_buffer = 1.001
        errors = False
        waiting_for_exits = False

        # When changing from five holdings to three, liquidate dropped
        # assets first so their filled proceeds can fund the new 33.33%
        # targets. The previous rank order bought selected assets first.
        ordered_signals = [
            signal for signal in signals if not signal.selected
        ] + [
            signal for signal in signals if signal.selected
        ]

        for signal in ordered_signals:
            pair = signal.pair
            if pair in pending_pairs:
                if pair not in selected_pairs:
                    waiting_for_exits = True
                self.journal.write("factor_pending_skip", pair=pair)
                continue
            rules = rules_all.get(pair, {})
            ticker = tickers.get(pair)
            if not ticker or not rules.get("CanTrade", False):
                continue
            coin = pair.split("/")[0]
            quantity = float(wallet.get(coin, {}).get("Free", 0))
            price = float(ticker["LastPrice"])
            current_notional = quantity * price
            amount_precision = int(rules.get("AmountPrecision", 6))
            price_precision = int(rules.get("PricePrecision", 2))
            minimum = float(rules.get("MiniOrder", 0))

            if pair not in selected_pairs:
                if current_notional >= minimum:
                    try:
                        self.executor.sell_limit(
                            pair, round_down(quantity, amount_precision),
                            # Quote one tick above the ask to avoid crossing
                            # if the book moves between ticker and order.
                            round_up(
                                float(ticker["MinAsk"]) + 10 ** (-price_precision),
                                price_precision,
                            ),
                        )
                        # A Maker sell does not release USD until it fills.
                        # Do not fund new targets from this notional yet.
                        waiting_for_exits = True
                    except Exception as exc:
                        errors = True
                        waiting_for_exits = True
                        self.log.warning("Factor sell failed for %s: %s", pair, exc)
                continue

            # Complete all exits before submitting any new buys. This avoids
            # assuming that a passive sell has filled immediately.
            if waiting_for_exits:
                continue

            difference = target_notional - current_notional
            if abs(difference) < minimum:
                continue
            if difference > 0:
                # Never spend more than the currently free USD balance. The
                # old implementation used the same stale equity target for
                # every pair, so pending orders could starve later buys.
                spend = min(difference, available_usd / cash_buffer)
                buy_qty = round_down(spend / price, amount_precision)
                if buy_qty > 0:
                    # Quote one tick below the bid. A best-bid order can still
                    # become a taker when the book moves before submission.
                    order_price = round_down(
                        max(10 ** (-price_precision), float(ticker["MaxBid"]) - 10 ** (-price_precision)),
                        price_precision,
                    )
                    if buy_qty * order_price < minimum:
                        continue
                    try:
                        self.executor.buy_limit(pair, buy_qty, order_price)
                        available_usd -= buy_qty * order_price * cash_buffer
                    except Exception as exc:
                        errors = True
                        self.log.warning("Factor buy failed for %s: %s", pair, exc)
            else:
                sell_qty = round_down(min(quantity, -difference / price), amount_precision)
                if sell_qty > 0:
                    self.executor.sell_limit(
                        pair, sell_qty,
                        # Quote one tick above the ask to avoid crossing.
                        round_up(
                            float(ticker["MinAsk"]) + 10 ** (-price_precision),
                            price_precision,
                        ),
                    )

        state["last_attempt"] = now.timestamp()
        state["selected"] = sorted(selected_pairs)
        state["retry"] = errors or waiting_for_exits
        if not errors and not waiting_for_exits:
            state["date"] = today
        self.store.save(self.bot_state)

    def _pending_factor_pairs(self) -> set[str]:
        """Return pairs with open orders so a rebalance does not duplicate them."""
        query = getattr(self.client, "query_order", None)
        if not callable(query):
            return set()
        try:
            payload = query(pending_only=1, limit=100)
        except Exception as exc:
            self.log.warning("Could not query pending orders: %s", exc)
            return set()
        orders = payload.get("OrderMatched", []) if isinstance(payload, dict) else []
        return {
            str(order.get("Pair"))
            for order in orders
            if isinstance(order, dict) and order.get("Pair")
        }

    def cycle(self) -> dict[str, Any]:
        exchange = self.client.exchange_info()
        tickers = self._ticker_map(self.client.ticker())
        balance_payload: dict[str, Any] | None = None
        if self.settings.dry_run and not self.settings.api_key:
            wallet = {"USD": {"Free": 100000.0, "Lock": 0.0}}
            shorts_list: list[dict[str, Any]] = []
        else:
            balance_payload = self.client.balance()
            wallet = self._wallet(balance_payload)
            shorts_list = self._short_positions(self.client.short_positions())
        equity = self._equity(wallet, tickers, shorts_list)
        # A credentialed run must never trade against an absent wallet.  This
        # also makes the current ``Wallet: null`` response visible in logs
        # instead of silently producing an equity of zero.
        if balance_payload is not None and not self._wallet_present(balance_payload):
            self.journal.write(
                "account_error",
                reason="WALLET_UNAVAILABLE",
                response=balance_payload,
            )
            return {"equity": equity, "halted": "WALLET_UNAVAILABLE", "signals": []}

        halt_reason = self.risk.update_equity(equity)
        shorts = {position["Pair"]: position for position in shorts_list}
        results: list[dict[str, Any]] = []
        self.journal.write("portfolio_snapshot", equity=equity, wallet=wallet, shorts=shorts_list)

        if halt_reason:
            # A halt prevents new trades, but does not violate the no-loss exit policy.
            self.journal.write("portfolio_halt", reason=halt_reason, equity=equity)
            return {"equity": equity, "halted": halt_reason, "signals": []}

        rules_all = exchange.get("TradePairs", {})
        candles_by_pair: dict[str, Any] = {}
        for pair in self.settings.pairs:
            try:
                if pair in rules_all and rules_all[pair].get("CanTrade", False):
                    candles_by_pair[pair] = self.data.candles(pair, "1d", max(self.settings.candle_limit, 80))
            except Exception as exc:
                self.log.warning("Factor data unavailable for %s: %s", pair, exc)
                self.journal.write("pair_error", pair=pair, error=str(exc))
        factor_signals = self.factor_strategy.rank(candles_by_pair)
        for signal in factor_signals:
            self.journal.write(
                "factor_signal", pair=signal.pair, factor=signal.factor_value,
                rank=signal.rank, selected=signal.selected, price=signal.price,
                strategy=self.settings.strategy,
            )
        self._factor_rebalance(
            factor_signals, wallet, rules_all, tickers, equity,
            self._pending_factor_pairs(),
        )
        return {
            "equity": equity,
            "halted": False,
            "strategy": self.settings.strategy,
            "signals": [{**signal.__dict__, "action": signal.action} for signal in factor_signals],
        }

    def run_forever(self) -> None:
        while True:
            try:
                result = self.cycle()
                self.log.info("Cycle complete: %s", result)
            except (RoostooAPIError, OSError) as exc:
                self.log.exception("Cycle failed")
                self.journal.write("cycle_error", error=str(exc))
            # Align cycles to UTC half-hour boundaries.  This makes the 00:00
            # UTC rebalance deterministic even when the process is started at
            # an arbitrary time; UTC 00:00 is 08:00 in Hong Kong.
            interval = max(1800, self.settings.loop_interval_seconds)
            now = time.time()
            next_boundary = (int(now // interval) + 1) * interval
            time.sleep(max(1.0, next_boundary - now))
