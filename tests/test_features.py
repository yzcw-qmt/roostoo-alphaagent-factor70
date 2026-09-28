from roostoo_bot.demo import sample_candles
from roostoo_bot.features import FEATURE_COLUMNS, build_features, make_training_set


def test_features_are_finite_after_warmup() -> None:
    candles = sample_candles()
    features = build_features(candles)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.iloc[60:].notna().all().all()


def test_training_labels_align() -> None:
    x, y = make_training_set(sample_candles())
    assert len(x) == len(y) > 100
    assert set(y.unique()).issubset({0, 1})

