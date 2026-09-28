from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from .api_client import RoostooClient, RoostooAPIError
from .config import Settings
from .data_fetcher import BinanceDataFetcher
from .executor import TradeExecutor, round_down
from .features import build_features
from .journal import Journal, StateStore
from .model import DirectionModel
from .ma_macd_strategy import MaMacdSignalGenerator, MaMacdXGBSignalGenerator
from .risk_manager import RiskManager
from .signals import Signal, SignalGenerator


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
        if settings.strategy in {"ma_macd", "ma_macd_xgb"}:
            technical = MaMacdSignalGenerator(
                atr_period=settings.atr_period,
                n=settings.macd_extrema_lookback,
                k=settings.extrema_k,
                risk_pct=settings.extrema_risk_pct,
                dynamic_n=settings.dynamic_extrema_n,
                long_threshold=settings.long_threshold,
                short_threshold=settings.short_threshold,
            )
            if settings.strategy == "ma_macd_xgb":
                self.signals = MaMacdXGBSignalGenerator(
                    technical,
                    DirectionModel(settings.model_path),
                    settings.model_weight,
                )
            else:
                self.signals = technical
        else:
            self.signals = SignalGenerator(
                DirectionModel(settings.model_path),
                settings.long_threshold,
                settings.short_threshold,
                settings.model_weight,
            )
        self.executor = TradeExecutor(self.client, self.journal, settings.dry_run)
        self.log = logging.getLogger(__name__)

    @staticmethod
    def _ticker_map(payload: dict[str, Any]) -> dict[str, Any]:
        return payload.get("Data", payload)

    @staticmethod
    def _wallet(payload: dict[str, Any]) -> dict[str, Any]:
        return payload.get("Wallet", {})

    def _equity(self, wallet: dict[str, Any], tickers: dict[str, Any], shorts: list[dict[str, Any]]) -> float:
        equity = sum(float(v.get("Free", 0)) + float(v.get("Lock", 0)) for k, v in wallet.items() if k == "USD")
        for coin, balance in wallet.items():
            pair = f"{coin}/USD"
            if coin != "USD" and pair in tickers:
                quantity = float(balance.get("Free", 0)) + float(balance.get("Lock", 0))
                equity += quantity * float(tickers[pair]["LastPrice"])
        equity += sum(float(position.get("PositionValue", 0)) for position in shorts)
        return equity

    def _cooldown_ready(self, pair: str) -> bool:
        value = self.bot_state.setdefault("last_trade", {}).get(pair)
        if not value:
            return True
        elapsed = datetime.now(timezone.utc).timestamp() - float(value)
        return elapsed >= self.settings.cooldown_minutes * 60

    def _remember_trade(self, pair: str, side: str, price: float, quantity: float) -> None:
        now = datetime.now(timezone.utc).timestamp()
        self.bot_state.setdefault("last_trade", {})[pair] = now
        self.bot_state.setdefault("positions", {})[pair] = {
            "side": side, "entry": price, "best": price, "quantity": quantity
        }
        self.store.save(self.bot_state)

    def _forget_position(self, pair: str) -> None:
        self.bot_state.setdefault("positions", {}).pop(pair, None)
        self.bot_state.setdefault("last_trade", {})[pair] = datetime.now(timezone.utc).timestamp()
        self.store.save(self.bot_state)

    @staticmethod
    def _unrealized_pnl(side: str, entry: float, current: float) -> float:
        return current / entry - 1 if side == "LONG" else entry / current - 1

    def _correlation_multipliers(self, candles: dict[str, Any], signals: list[Signal]) -> dict[str, float]:
        """Use BTC/ETH correlation only as a risk overlay, never as a direction signal."""
        multipliers = {signal.pair: 1.0 for signal in signals}
        btc, eth = "BTC/USD", "ETH/USD"
        if btc not in candles or eth not in candles or btc not in multipliers or eth not in multipliers:
            return multipliers
        btc_returns = candles[btc]["close"].astype(float).pct_change()
        eth_returns = candles[eth]["close"].astype(float).pct_change()
        correlation = btc_returns.rolling(self.settings.correlation_window).corr(eth_returns).iloc[-1]
        if not correlation >= self.settings.correlation_threshold:
            return multipliers
        btc_signal, eth_signal = next(s for s in signals if s.pair == btc), next(s for s in signals if s.pair == eth)
        if btc_signal.action == eth_signal.action and btc_signal.action != "HOLD":
            multipliers[btc] = multipliers[eth] = self.settings.same_direction_risk_multiplier
        elif {btc_signal.action, eth_signal.action} == {"LONG", "SHORT"}:
            multipliers[btc] = multipliers[eth] = self.settings.opposite_direction_risk_multiplier
        return multipliers

    def _trade_pair(
        self,
        signal: Signal,
        wallet: dict[str, Any],
        shorts: dict[str, dict[str, Any]],
        rules: dict[str, Any],
        equity: float,
        allocation_multiplier: float = 1.0,
        ticker: dict[str, Any] | None = None,
    ) -> None:
        pair = signal.pair
        coin = pair.split("/")[0]
        long_qty = float(wallet.get(coin, {}).get("Free", 0))
        short = shorts.get(pair)
        remembered = self.bot_state.setdefault("positions", {}).get(pair)

        if remembered:
            remembered["best"] = (
                max(float(remembered["best"]), signal.price)
                if remembered["side"] == "LONG"
                else min(float(remembered["best"]), signal.price)
            )
            pnl = self._unrealized_pnl(remembered["side"], float(remembered["entry"]), signal.price)
            if self.settings.strategy == "ma_macd":
                # A losing position is never sold.  This applies to both the
                # stop-loss path and the indicator exit path.
                signal_exit = pnl >= 0 and (
                    remembered["side"] == "LONG" and getattr(signal, "exit_long", False)
                ) or (
                    pnl >= 0 and remembered["side"] == "SHORT" and getattr(signal, "exit_short", False)
                )
                if signal_exit:
                    if remembered["side"] == "LONG" and long_qty > 0:
                        quantity = round_down(long_qty, int(rules["AmountPrecision"]))
                        if self.settings.order_type == "LIMIT" and ticker:
                            price = float(ticker["MinAsk"]) * (1 + self.settings.limit_offset_bps / 10000)
                            self.executor.sell_limit(pair, quantity, round_down(price, int(rules["PricePrecision"])))
                        else:
                            self.executor.sell(pair, quantity)
                    elif remembered["side"] == "SHORT" and short:
                        self.executor.close_short(pair)
                    self.journal.write("strategy_exit", pair=pair, reason="MACD_EXIT", price=signal.price, pnl=pnl)
                    self._forget_position(pair)
                    return
            # Legacy strategy keeps the no-realized-loss policy.
            elif pnl >= 0:
                exit_signal = (
                    remembered["side"] == "LONG" and signal.strength < self.settings.exit_long_threshold
                ) or (
                    remembered["side"] == "SHORT" and signal.strength > self.settings.exit_short_threshold
                )
                if exit_signal:
                    if remembered["side"] == "LONG" and long_qty > 0:
                        quantity = round_down(long_qty, int(rules["AmountPrecision"]))
                        if self.settings.order_type == "LIMIT" and ticker:
                            price = float(ticker["MinAsk"]) * (1 + self.settings.limit_offset_bps / 10000)
                            self.executor.sell_limit(pair, quantity, round_down(price, int(rules["PricePrecision"])))
                        else:
                            self.executor.sell(pair, quantity)
                    elif remembered["side"] == "SHORT" and short:
                        self.executor.close_short(pair)
                    self.journal.write("signal_exit", pair=pair, price=signal.price, pnl=pnl)
                    self._forget_position(pair)
                    return

        if signal.action == "LONG" and short:
            if remembered and self._unrealized_pnl(remembered["side"], float(remembered["entry"]), signal.price) >= 0:
                self.executor.close_short(pair)
                self._forget_position(pair)
            return
        if signal.action == "SHORT" and long_qty * signal.price >= float(rules["MiniOrder"]):
            if not remembered or self._unrealized_pnl(remembered["side"], float(remembered["entry"]), signal.price) >= 0:
                self.executor.sell(pair, round_down(long_qty, int(rules["AmountPrecision"])))
                self._forget_position(pair)
            return
        if not self._cooldown_ready(pair):
            return

        fraction = self.risk.position_fraction(signal.strength) * allocation_multiplier
        notional = min(equity * fraction, equity * self.settings.max_position_fraction)
        if notional < float(rules["MiniOrder"]) * self.settings.min_notional_buffer:
            return
        if signal.action == "LONG" and long_qty * signal.price < float(rules["MiniOrder"]):
            quantity = round_down(notional / signal.price, int(rules["AmountPrecision"]))
            if quantity > 0:
                if self.settings.order_type == "LIMIT" and ticker:
                    price = float(ticker["MaxBid"]) * (1 - self.settings.limit_offset_bps / 10000)
                    self.executor.buy_limit(pair, quantity, round_down(price, int(rules["PricePrecision"])))
                else:
                    self.executor.buy(pair, quantity)
                self._remember_trade(pair, "LONG", signal.price, quantity)
        elif signal.action == "SHORT" and not short:
            collateral = round(notional, 2)
            if self.settings.order_type == "LIMIT" and ticker:
                price = float(ticker["MinAsk"]) * (1 + self.settings.limit_offset_bps / 10000)
                self.executor.open_short_limit(pair, collateral, round_down(price, int(rules["PricePrecision"])))
            else:
                self.executor.open_short(pair, collateral)
            self._remember_trade(pair, "SHORT", signal.price, collateral / signal.price)

    def cycle(self) -> dict[str, Any]:
        exchange = self.client.exchange_info()
        tickers = self._ticker_map(self.client.ticker())
        if self.settings.dry_run and not self.settings.api_key:
            wallet = {"USD": {"Free": 100000.0, "Lock": 0.0}}
            shorts_list: list[dict[str, Any]] = []
        else:
            wallet = self._wallet(self.client.balance())
            shorts_list = self.client.short_positions().get("Positions", [])
        equity = self._equity(wallet, tickers, shorts_list)
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
        signals: list[Signal] = []
        for pair in self.settings.pairs:
            try:
                if pair not in rules_all or not rules_all[pair].get("CanTrade", False):
                    continue
                candles = self.data.candles(pair, self.settings.interval, self.settings.candle_limit)
                candles_by_pair[pair] = candles
                signal = self.signals.generate(pair, candles, build_features(candles))
                signals.append(signal)
                results.append(signal.__dict__)
                self.journal.write("signal", **signal.__dict__)
            except Exception as exc:
                self.log.exception("Pair cycle failed for %s", pair)
                self.journal.write("pair_error", pair=pair, error=str(exc))
        multipliers = (
            {signal.pair: 1.0 for signal in signals}
            if self.settings.strategy in {"ma_macd", "ma_macd_xgb"}
            else self._correlation_multipliers(candles_by_pair, signals)
        )
        for signal in signals:
            self._trade_pair(
                signal, wallet, shorts, rules_all[signal.pair], equity,
                multipliers.get(signal.pair, 1.0),
                tickers.get(signal.pair),
            )
        self.store.save(self.bot_state)
        return {"equity": equity, "halted": False, "signals": results}

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            try:
                result = self.cycle()
                self.log.info("Cycle complete: %s", result)
            except (RoostooAPIError, OSError) as exc:
                self.log.exception("Cycle failed")
                self.journal.write("cycle_error", error=str(exc))
            elapsed = time.monotonic() - started
            time.sleep(max(1.0, self.settings.loop_interval_seconds - elapsed))
