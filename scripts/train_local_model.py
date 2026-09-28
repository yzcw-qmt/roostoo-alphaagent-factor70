from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from roostoo_bot.features import FEATURE_COLUMNS, make_training_set


def load(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ["open_time", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame.set_index("open_time").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost from local chronological OHLCV files")
    parser.add_argument("--pairs", nargs="+", default=["BTC/USD", "ETH/USD"])
    parser.add_argument("--data-root", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("models/xgb_30m_multi.json"))
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = [], [], [], []
    for pair in args.pairs:
        path = args.data_root / f"{pair.replace('/', '-')}_30m_1y.json"
        x, y = make_training_set(load(path), horizon=args.horizon)
        split = int(len(x) * 0.8)
        train_x.append(x.iloc[:split])
        train_y.append(y.iloc[:split])
        test_x.append(x.iloc[split:])
        test_y.append(y.iloc[split:])

    x_train, y_train = pd.concat(train_x), pd.concat(train_y)
    x_test, y_test = pd.concat(test_x), pd.concat(test_y)
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
    model.fit(x_train[FEATURE_COLUMNS], y_train, eval_set=[(x_test[FEATURE_COLUMNS], y_test)], verbose=False)
    probability = model.predict_proba(x_test[FEATURE_COLUMNS])[:, 1]
    print(f"pairs={','.join(args.pairs)} train={len(x_train)} test={len(x_test)}")
    print(f"AUC: {roc_auc_score(y_test, probability):.4f}")
    print(classification_report(y_test, (probability >= 0.5).astype(int), digits=4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
