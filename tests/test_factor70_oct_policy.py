from __future__ import annotations

import unittest

import pandas as pd

from scripts.factor70_oct_window import END, START, MarketData, Parameters, select_targets, simulate


class Factor70OctPolicyTests(unittest.TestCase):
    def test_new_top_rank_from_one_day_spike_is_blocked(self) -> None:
        snapshot = {
            "A/USD": (0.80, 0.06, 4),
            "B/USD": (0.70, 0.00, 1),
            "C/USD": (0.65, 0.00, 2),
            "D/USD": (0.62, 0.00, 3),
        }
        self.assertEqual(select_targets(snapshot, Parameters()), ["A/USD", "B/USD", "C/USD"])
        self.assertEqual(select_targets(snapshot, Parameters(spike_limit=0.05)), ["B/USD", "C/USD", "D/USD"])

    @staticmethod
    def flat_market(with_drop: bool = False) -> MarketData:
        bars = list(pd.date_range(START, END, freq="30min", inclusive="left"))
        snapshots = {day: {"A/USD": (0.6, 0.0, 1)} for day in bars[::48]}
        opens = {bar: {"A/USD": 100.0} for bar in bars}
        closes = {bar: {"A/USD": 100.0} for bar in bars}
        scores = {bar: {"A/USD": 0.6} for bar in bars}
        if with_drop:
            scores[bars[1]]["A/USD"] = 0.5
        return MarketData(bars, snapshots, opens, closes, scores)

    def test_one_qualifying_asset_uses_one_third_of_equity(self) -> None:
        result = simulate(self.flat_market(), Parameters(min_score=0.56))
        self.assertAlmostEqual(result["mean_exposure"], 1 / 3, delta=0.002)
        self.assertEqual(result["day_targets"][0]["targets"], ["A/USD"])

    def test_no_qualifying_asset_stays_in_cash(self) -> None:
        result = simulate(self.flat_market(), Parameters(min_score=0.7))
        self.assertEqual(result["final_equity"], 100000.0)
        self.assertEqual(result["trade_count"], 0)

    def test_intraday_drop_exits_on_next_bar(self) -> None:
        result = simulate(self.flat_market(with_drop=True), Parameters(min_score=0.56, exit_drop=0.06))
        exits = [trade for trade in result["trade_log"] if trade["side"] == "SELL"]
        self.assertEqual(exits[0]["time"], "2026-10-04T01:00:00+00:00")
        self.assertAlmostEqual(exits[0]["fee"] / exits[0]["notional"], 0.001)

if __name__ == "__main__":
    unittest.main()
