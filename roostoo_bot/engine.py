from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

from .api_client import RoostooClient, RoostooAPIError
from .config import Settings
from .data_fetcher import BinanceDataFetcher
from .executor import TradeExecutor, round_down, round_up
from .factor_strategy import (
    CrossSectionalFactorStrategy,
    FactorSignal,
    alphaagent_70_live,
)
from .journal import Journal, StateStore
from .risk_manager import RiskManager


MAX_REBALANCE_ATTEMPTS = 6


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
        self.factor_strategy = CrossSectionalFactorStrategy(
            settings.strategy,
            settings.factor_top_n,
            min_score=settings.factor_min_score,
            max_daily_gain=settings.factor_max_daily_gain,
        )
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

    def _factor_targets(
        self,
        signals: list[FactorSignal],
        state: dict[str, Any],
        holdings: set[str],
    ) -> tuple[list[FactorSignal], set[str]]:
        """Build the target basket for this rebalance.

        With ``factor_exit_rel_drop == 0`` this is a plain daily top-N
        rebalance.  With a positive value the book becomes *sticky*: a held
        name keeps its slot even if it slips down the ranking, and is only
        sold once its factor collapses by that fraction against the score
        recorded at the previous rebalance.  That is the rule that was
        actually backtested -- applying the drop only as an extra filter on
        top of a full daily rotation does not reproduce the result, because
        the rotation sells every name the sticky rule would have kept.
        """
        by_pair = {signal.pair: signal for signal in signals}
        drop = float(self.settings.factor_exit_rel_drop)
        previous = state.get("scores") or {}

        if drop <= 0:
            targets = [signal for signal in signals if signal.selected]
            return targets[: self.settings.factor_top_n], set()

        keep: list[FactorSignal] = []
        forced: set[str] = set()
        for pair in sorted(holdings):
            signal = by_pair.get(pair)
            if signal is None:
                # Untradable or not in today's universe: let the sell path
                # unwind it rather than keeping a name we cannot score.
                continue
            old = previous.get(pair)
            if old and old > 0 and signal.factor_value / float(old) - 1.0 < -drop:
                forced.add(pair)
                self.journal.write(
                    "factor_forced_exit", pair=pair,
                    score=signal.factor_value, previous_score=old,
                )
                continue
            if not old and signal.blocked:
                # No score was recorded for this position (first run after an
                # upgrade, or a state reset), so the exit rule has nothing to
                # compare against. Fall back to the entry filters: a name we
                # would not buy today should not be carried on faith.
                forced.add(pair)
                self.journal.write(
                    "factor_forced_exit", pair=pair,
                    score=signal.factor_value, previous_score=None,
                    reason=signal.blocked,
                )
                continue
            keep.append(signal)

        keep.sort(key=lambda item: (-item.factor_value, item.pair))
        keep = keep[: self.settings.factor_top_n]
        chosen = {signal.pair for signal in keep}
        # A name that was just dropped must not be re-admitted by the refill
        # loop below; otherwise the forced exit is silently undone.
        rejected = set(holdings) - chosen
        # Entry filters apply only to new positions; an existing holding is
        # governed by the exit rule above.
        for signal in signals:
            if len(keep) >= self.settings.factor_top_n:
                break
            if signal.blocked or signal.pair in chosen or signal.pair in rejected:
                continue
            keep.append(signal)
            chosen.add(signal.pair)
        return keep, forced

    def _factor_rebalance(
        self,
        signals: list[FactorSignal],
        wallet: dict[str, Any],
        rules_all: dict[str, Any],
        tickers: dict[str, Any],
        equity: float,
        pending_pairs: set[str] | None = None,
    ) -> None:
        """Rebalance a daily long-only top-N factor portfolio with LIMIT orders.

        Exits run first and new buys wait until every exit order is filled,
        because a resting Maker sell does not free USD.  The basket can be
        empty: when nothing clears the entry filters the portfolio goes to
        cash instead of holding a weak position overnight.
        """
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

        holdings = {
            f"{coin}/USD"
            for coin, balance in wallet.items()
            if coin != "USD"
            and float(balance.get("Free", 0)) + float(balance.get("Lock", 0)) > 0
            and f"{coin}/USD" in rules_all
        }
        targets, forced_exit = self._factor_targets(signals, state, holdings)
        target_pairs = {signal.pair for signal in targets}
        target_notional = equity / len(targets) if targets else 0.0
        pending_pairs = pending_pairs or set()
        available_usd = float(wallet.get("USD", {}).get("Free", 0))
        # Keep room for the 0.05% Maker commission and small price movement.
        cash_buffer = 1.001
        errors = False
        waiting_for_exits = False
        incomplete = False
        self.journal.write(
            "factor_targets", targets=sorted(target_pairs),
            forced_exit=sorted(forced_exit), equity=round(equity, 2),
        )

        # Liquidate dropped assets first so their filled proceeds can fund the
        # new equal-weight targets.  Rank order alone would buy first and
        # starve the later names.
        ordered = [signal for signal in signals if signal.pair not in target_pairs] + [
            signal for signal in signals if signal.pair in target_pairs
        ]

        for signal in ordered:
            pair = signal.pair
            rules = rules_all.get(pair, {})
            ticker = tickers.get(pair)
            if not ticker or not rules.get("CanTrade", False):
                if pair in target_pairs:
                    incomplete = True
                continue
            coin = pair.split("/")[0]
            quantity = float(wallet.get(coin, {}).get("Free", 0))
            price = float(ticker["LastPrice"])
            current_notional = quantity * price
            amount_precision = int(rules.get("AmountPrecision", 6))
            price_precision = int(rules.get("PricePrecision", 2))
            minimum = float(rules.get("MiniOrder", 0))

            if pair in pending_pairs:
                if pair not in target_pairs:
                    waiting_for_exits = True
                else:
                    incomplete = True
                self.journal.write("factor_pending_skip", pair=pair)
                continue

            if pair not in target_pairs:
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
                incomplete = True
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
                # Quote one tick below the bid. A best-bid order can still
                # become a taker when the book moves before submission.
                order_price = round_down(
                    max(10 ** (-price_precision), float(ticker["MaxBid"]) - 10 ** (-price_precision)),
                    price_precision,
                )
                if buy_qty <= 0 or buy_qty * order_price < minimum:
                    # Cash actually available is below the target. Flag this
                    # so the next cycle retries instead of marking the day
                    # complete with an underweight book (the bug behind the
                    # original "retry=False while Top 3 unfilled" behaviour).
                    incomplete = True
                    self.journal.write(
                        "factor_underfunded", pair=pair,
                        target=round(target_notional, 2),
                        available=round(available_usd, 2),
                    )
                    continue
                try:
                    self.executor.buy_limit(pair, buy_qty, order_price)
                    available_usd -= buy_qty * order_price * cash_buffer
                except Exception as exc:
                    errors = True
                    incomplete = True
                    self.log.warning("Factor buy failed for %s: %s", pair, exc)
            else:
                sell_qty = round_down(min(quantity, -difference / price), amount_precision)
                if sell_qty > 0:
                    try:
                        self.executor.sell_limit(
                            pair, sell_qty,
                            # Quote one tick above the ask to avoid crossing.
                            round_up(
                                float(ticker["MinAsk"]) + 10 ** (-price_precision),
                                price_precision,
                            ),
                        )
                    except Exception as exc:
                        errors = True
                        self.log.warning("Factor trim failed for %s: %s", pair, exc)

        state["last_attempt"] = now.timestamp()
        state["selected"] = sorted(target_pairs)
        attempts = int(state.get("attempts", 0)) + 1
        state["attempts"] = attempts
        retry = errors or waiting_for_exits or incomplete
        # A limit order that never fills would otherwise keep the day open
        # forever and let later retries trim positions that are already at
        # target. Give up after a bounded number of attempts and let the next
        # daily rebalance handle the residual.
        gave_up = retry and attempts >= MAX_REBALANCE_ATTEMPTS
        if gave_up:
            self.journal.write(
                "factor_rebalance_gave_up", attempts=attempts,
                targets=sorted(target_pairs), equity=round(equity, 2),
            )
        state["retry"] = retry and not gave_up
        if not state["retry"]:
            state["date"] = today
            state["attempts"] = 0
            state["scores"] = {
                signal.pair: signal.factor_value for signal in signals
            }
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

    def _factor_intraday_exit(
        self,
        wallet: dict[str, Any],
        rules_all: dict[str, Any],
        tickers: dict[str, Any],
        pending_pairs: set[str] | None = None,
    ) -> list[str]:
        """Sell a held name as soon as its live factor collapses.

        The daily rebalance records one baseline score per holding.  Between
        rebalances this runs on every loop tick and re-scores each position
        using the live price in place of the unfinished daily candle.  When a
        score falls further than ``factor_exit_rel_drop`` below its baseline
        the position is sold immediately instead of waiting for the next
        UTC 00:00 review -- which is what turned a 6% one-day drop into a
        multi-day hold in the 10-12 PLUME trade.

        Entries are *not* touched here: buying stays on the daily schedule so
        the book is not rebuilt intraday on noise.
        """
        drop = float(self.settings.factor_exit_rel_drop)
        if drop <= 0:
            return []
        state = self.bot_state.get("factor_rebalance") or {}
        baseline = state.get("scores") or {}
        if not baseline:
            return []
        pending_pairs = pending_pairs or set()
        sold: list[str] = []
        for coin, balance in wallet.items():
            if coin == "USD":
                continue
            pair = f"{coin}/USD"
            if pair not in rules_all or pair not in baseline:
                continue
            quantity = float(balance.get("Free", 0))
            if quantity <= 0 or pair in pending_pairs:
                continue
            ticker = tickers.get(pair)
            if not ticker or not rules_all[pair].get("CanTrade", False):
                continue
            try:
                frame = self.data.candles(pair, "1d", max(self.settings.candle_limit, 80))
            except Exception as exc:
                self.log.warning("Intraday factor data unavailable for %s: %s", pair, exc)
                continue
            live = alphaagent_70_live(frame, float(ticker["LastPrice"]))
            old = float(baseline[pair])
            if not math.isfinite(live) or old <= 0:
                continue
            change = live / old - 1.0
            if change >= -drop:
                continue
            precision = int(rules_all[pair].get("AmountPrecision", 6))
            price_precision = int(rules_all[pair].get("PricePrecision", 2))
            try:
                self.executor.sell_limit(
                    pair, round_down(quantity, precision),
                    # A sell limit above the ask is passive and may sit open
                    # while the position keeps falling. Quote at the best bid
                    # (rounded down to a valid tick) so the exit is marketable
                    # while retaining a price floor.
                    round_down(float(ticker["MaxBid"]), price_precision),
                )
                sold.append(pair)
            except Exception as exc:
                self.log.warning("Intraday exit failed for %s: %s", pair, exc)
                continue
            # Keep the baseline on purpose. A LIMIT sell is not a fill: it
            # can rest, be cancelled, or fill only in part, and the resting
            # quantity stays Locked (Free drops to 0). Deleting the baseline
            # here would leave the remaining position unmonitored forever.
            # Duplicate orders are already prevented by the pending-order
            # check above and by only selling the Free quantity.
            self.journal.write(
                "factor_intraday_exit", pair=pair, score=live,
                previous_score=old, change=round(change, 4),
            )
        if sold:
            self.store.save(self.bot_state)
        return sold

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
        pending = self._pending_factor_pairs()
        self._factor_intraday_exit(wallet, rules_all, tickers, pending)
        return {
            "equity": equity,
            "halted": False,
            "strategy": self.settings.strategy,
            "signals": [
                {**signal.__dict__, "action": signal.action} for signal in factor_signals
            ],
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
