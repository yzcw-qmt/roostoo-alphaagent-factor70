# Roostoo Quant Trading Bot

面向 Roostoo Quant Trading Hackathon 的本地可运行交易系统。它把方案中的 **XGBoost + 多因子融合 + 严格风控** 落成代码，并按比赛页面和官方 API 文档实现。

> 默认是 `dry-run`。只有显式执行 `python main.py run --live` 且提供有效密钥时才会发出真实交易请求。

## 比赛约束对应关系

- 全自动交易：主循环独立完成取数、信号、风控和执行，不提供手动下单入口。
- 非高频：默认每小时运行一次；客户端额外限制为每分钟 28 次请求，低于 30 次上限。
- 现货 1× 多空：多仓使用 `/v3/place_order`，空仓使用 `/v6/short_open` 与 `/v6/short_close`，不使用杠杆。
- 成本建模：回测按市场单 0.1% 手续费扣减；实盘日志记录 API 返回的实际费用。
- 订单执行：默认使用普通限价单，现货多仓开/平与空仓开仓按买一/卖一挂单；空仓平仓当前仍使用市价单，因为 Roostoo `/v6/short_close` 封装没有限价字段。普通 `LIMIT` 不等于保证 Maker，实盘需根据实际成交回报确认费率。
- 风险调整收益：亏损仓位不主动平仓；日亏损 5% 或总回撤 12% 后停止开新仓，盈利仓位才按信号退出。
- 可审计：`logs/events.jsonl` 记录信号、组合快照、订单成功/失败与风控退出。
- 比赛合规：所有执行均由策略触发，便于用提交历史与日志证明没有手工调用 API。

官方接口基准：[Roostoo API Documents](https://github.com/roostoo/Roostoo-API-Documents)

## 代码结构

```text
main.py                         # demo / API 检查 / 单次或持续运行
roostoo_bot/
  api_client.py                 # HMAC-SHA256、限流、重试、/v3 与 /v6
  data_fetcher.py               # Binance 公共 1h K 线
  features.py                   # 15 个趋势/动量/波动特征
  model.py                      # XGBoost 推理与无模型 fallback（旧策略）
  signals.py                    # 旧版模型 + 技术因子信号
  ma_macd_strategy.py           # 纯技术指标 MA10/MA60 + MACD 策略
  risk_manager.py               # 置信度仓位、日亏损、总回撤
  executor.py                   # 多/空订单执行与 dry-run
  engine.py                     # 自主交易主循环
  journal.py                    # JSONL 审计日志与持仓状态
scripts/
  train_model.py                # 时间顺序切分训练
  train_local_model.py          # 用本地历史行情训练 XGBoost
  backtest.py                   # 含手续费的基础回测
  evaluate_technical_signal.py  # 评估纯技术最终信号的方向命中率
  hybrid_backtest.py            # 30m MA/MACD + XGBoost 双费率回测
tests/                          # 签名、特征、风控、离线闭环测试
```

## 本地查看效果

### 1. 创建环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 完全离线演示

无需密钥，也不访问交易所：

```powershell
python main.py demo
```

输出会包含模拟组合净值、BTC/ETH 的模型概率、技术分、融合分和 `LONG` / `SHORT` / `HOLD` 动作。详细事件写入 `logs/events.jsonl`。

### 3. 测试

```powershell
python -m pytest
```

### 4. Binance 纸面交易模式（无需任何 Key）

```powershell
python main.py paper
python main.py paper --loop
```

该模式的 1h OHLCV 与实时 ticker 都来自 Binance 的无密钥纯市场数据域名 `data-api.binance.vision`。订单只写入本地日志，不会发送给 Binance 或 Roostoo。

### 5. 仅检查官方公共 API

```powershell
python main.py check-api
```

### 6. Roostoo 联网 dry-run

复制配置但暂不放密钥：

```powershell
Copy-Item .env.example .env
python main.py run --once
```

它会读取 Roostoo 公共 ticker 和 Binance K 线，但不会下单。若希望 dry-run 同时读取比赛账户余额，可以在 `.env` 中填入密钥；所有交易动作仍只记日志。

## 训练与回测

```powershell
python -m scripts.train_model --pair BTC/USD --candles 1000
python -m scripts.backtest --pair BTC/USD --candles 1000
```

训练出模型后可用 `STRATEGY=ma_macd_xgb` 启用融合信号；当前默认 `MODEL_WEIGHT=0.70`，也可通过环境变量覆盖。建议先做严格的时间切分样本外评估，再接入 Roostoo dry-run。

当前一年 30 分钟数据的双费率回测：

```bash
python -m scripts.hybrid_backtest
```

### 多资产、随机 15 日片段回测

```powershell
python -m scripts.multi_asset_backtest `
  --pairs BTC/USD ETH/USD BNB/USD TON/USD TRX/USD `
  --start 2025-09-21 `
  --end 2026-09-21 `
  --segments 16 `
  --segment-days 15 `
  --seed 42
```

该回测先用固定随机种子从一年内的非重叠 15 天候选窗口中抽样，再计算成绩；每个片段均重置为 100,000 美元。策略包括：

- BTC 72 小时滚动均值回归，不使用固定价格突破阈值；
- BTC 与 ETH 高相关同趋势时，按照相对价差做多低估方、做空高估方；
- BTC 大趋势足够强时改为单边方向，不强制对冲；
- 用滚动 lead-lag 相关性将领先币种最近三小时方向传递给滞后币种；
- BTC、ETH、BNB、TON、TRX 单品种不超过 30%，组合总敞口不超过 90%，净敞口不超过 60%；
- 空仓是默认状态：BTC 均值回归要求绝对 z-score 超过 2.0，市场趋势强度超过 2.25，配对价差绝对 z-score 超过 2.0 且相关性超过 0.70；
- lead-lag 除相关性超过 0.20 外，还要求领先资产最近 3 小时出现超过 1.5 倍波动的方向冲击；
- 每 12 小时最多再平衡一次，组合目标变化不足 15% 时不交易，避免小时级噪声和手续费消耗；
- 按每次换仓金额收取 0.1% 费用，信号在下一根 K 线生效，避免直接使用未来收益。

完整结果保存为 `reports/multi_asset_backtest.json`，历史行情缓存到 `data/history/`。使用相同时间、币种和随机种子可以复现实验；添加 `--refresh` 可重新下载行情。

当前实盘默认使用 30 分钟 MA10/MA60 + MACD 评分策略，默认只交易 BTC/USD 和 ETH/USD。该策略只使用 ATR、均线、MACD、DIFF/DEA 和因果极值确认，不使用 XGBoost。另有 `STRATEGY=ma_macd_xgb` 模式，将同一纯技术分与 XGBoost 概率按 `MODEL_WEIGHT` 融合，再由 `TradingEngine` 接入 Roostoo 的账户、行情和订单执行；如果模型文件不存在，日志中会标明使用 fallback，不应当作真正的 XGBoost 结果。最终信号总分 `>=0.62` 做多、`<=0.38` 做空，否则观望。可以运行 `python -m scripts.evaluate_technical_signal --pair BTC/USD` 评估纯技术信号。

Binance 单次 K 线接口最多返回 1000 根，正式训练应另行准备更长的历史 CSV，并采用 walk-forward 验证。当前脚本是可运行基线，不应把样本内结果视为未来收益保证。

## 实盘/比赛启动

在 `.env` 填写：

```dotenv
ROOSTOO_API_KEY=你的API_KEY
ROOSTOO_SECRET_KEY=你的SECRET_KEY
DRY_RUN=true
TRADING_PAIRS=BTC/USD,ETH/USD
```

先至少运行 24 小时 dry-run 并核对日志。确认后才使用：

```powershell
python main.py run --live
```

建议在 AWS EC2 上用 Docker 和 `restart: unless-stopped` 管理进程，并用外部监控检查 `events.jsonl` 是否持续产生。不要把 `.env`、密钥、比赛账户响应或日志提交到公开仓库。

## 重要边界

- 本项目不会承诺收益；回测和演示不代表实盘表现。
- 赛制要求至少 8 个活跃交易日，但不应为凑交易次数而强制下单。信号阈值和冷却时间应在比赛前验证后调整。
- API 的 `Success=false` 可能仍使用 HTTP 200；客户端会检查业务状态并记录失败。
- Roostoo 时间戳容差是 60 秒。部署机器应启用时间同步，启动前可运行 `check-api`。
