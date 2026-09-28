from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from roostoo_bot.data_fetcher import BinanceDataFetcher
from roostoo_bot.features import make_training_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 1h direction model with a chronological split")
    parser.add_argument("--pair", default="BTC/USD")
    parser.add_argument("--candles", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("models/xgb_btc_1h.json"))
    args = parser.parse_args()

    candles = BinanceDataFetcher().candles(args.pair, "1h", args.candles)
    x, y = make_training_set(candles)
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    probability = model.predict_proba(x_test)[:, 1]
    print(f"AUC: {roc_auc_score(y_test, probability):.4f}")
    print(classification_report(y_test, (probability >= 0.5).astype(int), digits=4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

