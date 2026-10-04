from dataclasses import replace

from roostoo_bot.config import Settings
from roostoo_bot.data_fetcher import BinancePaperClient
from roostoo_bot.demo import DemoDataFetcher, DemoRoostooClient
from roostoo_bot.engine import TradingEngine


def test_offline_cycle(tmp_path) -> None:
    settings = replace(
        Settings(),
        pairs=("BTC/USD",),
        state_path=tmp_path / "state.json",
        log_dir=tmp_path / "logs",
            model_path=tmp_path / "missing.json",
            dry_run=True,
            strategy="ma_macd_xgb",
        )
    engine = TradingEngine(settings, DemoRoostooClient(settings.pairs), DemoDataFetcher())
    result = engine.cycle()
    assert result["equity"] == 100000
    assert result["halted"] is False
    assert len(result["signals"]) == 1
    assert result["signals"][0]["action"] in {"LONG", "SHORT", "HOLD"}


def test_binance_paper_adapter_uses_roostoo_shape() -> None:
    class FakeFetcher:
        def ticker(self, pair: str) -> dict:
            return {"MaxBid": 99, "MinAsk": 101, "LastPrice": 100, "Change": 0.01}

    client = BinancePaperClient(("BTC/USD",), FakeFetcher())
    assert client.exchange_info()["TradePairs"]["BTC/USD"]["CanTrade"] is True
    assert client.ticker()["Data"]["BTC/USD"]["LastPrice"] == 100
