"""Train the final asset-specific rolling XGBoost models.

The selected windows come from the 24 x 15-day walk-forward grid:
BTC/USD: 180 days; ETH/USD: 120 days.  The model predicts whether the
close will be higher after six 30-minute bars (roughly three hours).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgboost import XGBClassifier

from roostoo_bot.features import FEATURE_COLUMNS, make_training_set
from scripts.train_local_model import load


WINDOWS = {"BTC/USD": 180, "ETH/USD": 120}


def train_pair(pair: str, data_root: Path, output_dir: Path, horizon: int) -> Path:
    frame = load(data_root / f"{pair.replace('/', '-')}_30m_1y.json")
    x, y = make_training_set(frame, horizon=horizon)
    window = WINDOWS[pair] * 48
    x, y = x.tail(window), y.tail(window)
    valid = x[FEATURE_COLUMNS].notna().all(axis=1)
    x, y = x.loc[valid], y.loc[valid]
    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x[FEATURE_COLUMNS], y, verbose=False)
    output = output_dir / f"xgb_{pair.split('/')[0].lower()}_30m_{WINDOWS[pair]}d.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    print(f"{pair}: window={WINDOWS[pair]}d samples={len(x)} output={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/history/550d"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()
    outputs = [str(train_pair(pair, args.data_root, args.output_dir, args.horizon)) for pair in WINDOWS]
    (args.output_dir / "final_model_manifest.json").write_text(
        json.dumps({"windows_days": WINDOWS, "horizon_bars": args.horizon, "outputs": outputs}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
