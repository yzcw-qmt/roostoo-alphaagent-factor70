# Factor 70: 4–8 October 2026 optimization experiment

This is **post-competition, in-sample research**. The five trading days were used both to choose parameters and to measure the reported returns. None of the figures below are the team's realized Roostoo competition return or an out-of-sample estimate.

## Data and execution model

- Universe: the user-supplied list of 88 Roostoo `/USD` pairs, saved in [`eligible_pairs_88_2026-10-19.json`](../data/research/eligible_pairs_88_2026-10-19.json). The list is not a verified day-by-day exchange snapshot and does not prove each pair was in the deployed bot's `TRADING_PAIRS` setting.
- Prices: public Binance `/USDT` daily and 30-minute candles mapped by base symbol. 86 pairs had at least 22 completed daily closes at the start and all 240 intraday bars; `OMNI/USD` and `TON/USD` were omitted. This price proxy can differ from Roostoo `/USD` quotes, especially for tokenized stocks.
- Signals: at 00:00 UTC, rank using the previous completed daily candle. Older August–September candles only warm up the 20-day factor; portfolio returns are measured solely from 4 October 00:00 through 8 October 23:30 UTC.
- Fills: daily rebalances at the next 30-minute open and intraday score exits at the following 30-minute open. Base case charges the 0.05% maker fee for daily limit-order rebalances and the 0.10% taker fee for intraday market exits. It assumes zero slippage and full immediate fills. Stress cases add fees and slippage. Drawdown is computed from 30-minute closing equity, so within-bar lows are not captured.
- Allocation: each of the top three *qualifying* pairs receives one third of account equity. If only one or two qualify, the remaining slots stay in USD. The strategy is long-only and unlevered.

## Tested changes

1. **Spike after a rank jump.** If yesterday's close-to-close rise exceeds the limit, and that day's factor update moved the asset from outside the raw top three into the raw top three, exclude it from new buys. A high return alone does not exclude an existing top-three asset.
2. **Minimum score.** Rank only pairs with Factor 70 at or above the threshold. The initial 0.20–0.30 search did not change any top-three choice on these five days. The search was extended to 0.60; 0.56 gave the highest tested return with meaningful exposure.
3. **Intraday score exit.** After each completed 30-minute candle, replace the unfinished day's close in the Factor 70 calculation with that candle's close. Compare against the first completed 30-minute score of the same UTC day. A fall of at least the exit threshold schedules a market sell at the next 30-minute open; there is no same-day re-entry. The within-day reference avoids a false exit caused solely by rolling the 20-day window forward at midnight.

## Results

All values below are from the same five-day calibration sample, on a $100,000 starting account.

| Configuration | Total return | Max 30m drawdown | Average invested | Trades | Fees |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original daily top three | −3.32% | −5.35% | 100.0% | 20 | $216.70 |
| 5% spike filter only | −2.01% | −5.08% | 100.0% | 20 | $218.21 |
| Score ≥ 0.56 only | −0.07% | −2.51% | 66.7% | 14 | $150.13 |
| Both entry filters, no intraday exit | **+1.85%** | −1.72% | 60.1% | 12 | $118.21 |
| Both entry filters, score drop ≥ 0.06 | +1.43% | **−1.38%** | 49.3% | 13 | $201.03 |

The entry filters are complementary in this sample. The 5% rank-jump filter removes `ICP/USD` from the 6 October selection after a 6.69% prior-day rise; without it, `ICP/USD` enters the top three. The 0.56 score floor also leaves one or two slots in cash on all five days. These are concrete changes in positions, not evidence that Factor 70 has general predictive power.

The 0.06 intraday exit results in three intraday market sells and lowers the sampled maximum drawdown by about 0.34 percentage points, while reducing sampled return by about 0.43 points. At the same 5% spike threshold, a tighter 0.02 exit produced +1.13% return and −0.65% drawdown; an exit threshold of 0.15 never triggered and matched the no-exit result. The exit rule is therefore a risk preference, not a proven return improvement.

## Parameter sensitivity and chosen configuration

- Spike limits from **2% through 6%** produced the same selections and +1.85% return with the 0.56 floor and no intraday exit. A 7% limit failed to block the `ICP/USD` event and returned about −0.07%. The configuration uses **5%** as a representative point inside the observed plateau.
- At a 5% spike limit, a **0.54–0.55** floor returned about +1.28%, **0.56** returned +1.85%, and **0.57–0.60** returned about +1.07% without intraday exits. The **0.56** choice is a narrow five-day optimum, and its precision should not be interpreted as statistically established.
- With both entry filters fixed, the maximum-return setting **does not use an intraday exit** (+1.85%, −1.72% drawdown). The 0.06 absolute score-drop exit was an exploratory risk-oriented setting (+1.43%, −1.38% drawdown) because reducing drawdown was part of the requested risk control. This is a transparent return-for-risk tradeoff.
- Under **0.10% fee per daily rebalance and intraday exit plus 0.10% slippage per side**, the no-exit and 0.06-exit cases returned about +1.50% and +1.02%, respectively. At 0.10% fee plus 0.20% slippage, they returned about +1.26% and +0.72%. These stresses still assume immediate fills and no market impact.

## Combined optimization by the published metric weights

The follow-up [`optimize_factor70_combined.py`](../scripts/optimize_factor70_combined.py) re-scores 3,341 combinations of all three enabled rules, including finer local grids and score floors through 0.70. It uses the repository's daily metric convention: zero risk-free/target rate, sample-standard-deviation Sharpe, downside-RMS Sortino, and annualized five-day return divided by **daily-close** maximum drawdown for Calmar. The organizer publishes the 0.4/0.3/0.3 metric weights but not the exact measurement convention, so these are **research proxies, not official scores**.

| Objective and three-rule settings | Return | Sortino | Sharpe | Calmar | Weighted proxy | Max 30m drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highest weighted proxy: rank-jump rise > **5%**, score ≥ **0.60**, intraday score drop ≥ **0.02** | **+1.30%** | **41.76** | **11.30** | **590.75** | **197.32** | **−0.62%** |
| Highest return with an actual intraday exit: rise > **2%**, score ≥ **0.56**, drop ≥ **0.06** | **+1.43%** | 16.43 | 8.18 | 244.49 | 82.37 | −1.38% |
| Entry filters only, no intraday exit: rise > **5%**, score ≥ **0.56** | **+1.85%** | 28.83 | 11.91 | 515.80 | 169.84 | −1.72% |

The 5%/0.60/0.02 representative ties the highest tested weighted proxy. Several cutoffs produce identical trades, so 0.25%/0.5575/0.015 is the first sorted maximizer but **not a uniquely identified optimum**. The representative ends at $101,300.73 from $100,000, with 13 executed trades and $235.38 modeled fees. With 0.10% and 0.20% slippage per side, its return drops to +0.96% and +0.63%, respectively. Removing only the spike filter reduces return to +0.81%; removing only the score floor yields −0.53%; removing only the intraday exit yields +1.07%. These in-sample ablations show all three rules affect the result, but they do not establish causal or future benefit. The setting should remain a research candidate until checked on data outside the tuning window and on Roostoo quotes. The five-day annualization makes the Calmar and weighted proxy especially unstable.

The five-day sample contains one influential `ICP/USD` episode and few independent trades. It cannot identify stable optimal parameters. The values above are provisional settings for review, not a forecast or a claim that the competition's later loss was solved.

## Reproduce

Install the repository requirements, then run from the repository root:

```bash
python scripts/factor70_oct_window.py \
  --universe data/research/eligible_pairs_88_2026-10-19.json
```

The first run downloads public Binance candles into the ignored `data/research/factor70_oct_candles/` cache. It writes the 1,441-setting coarse sweep, metadata, daily selections, trades, ablations and execution-cost stresses under `reports/factor70_oct_window/`. Run `python scripts/optimize_factor70_combined.py` for the refined combined search. The research scripts never send orders, and the competition bot's default strategy settings are not changed by this experiment. Competition performance must continue to be reported separately from this post-competition experiment.

The committed research outputs are the [coarse parameter sweep](../reports/factor70_oct_window/sweep.csv), [full 3,341-setting combined sweep](../reports/factor70_oct_window/combined_sweep.csv), [combined summary and sensitivities](../reports/factor70_oct_window/combined_optimization.json), [scenario diagnostics](../reports/factor70_oct_window/diagnostics.json), and [universe metadata](../reports/factor70_oct_window/metadata.json). Raw Binance candle caches are omitted because the scripts can request them again from the public endpoint.
