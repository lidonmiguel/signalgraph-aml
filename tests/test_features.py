from signalgraph_aml.config import FEATURE_COLUMNS
from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.features import build_account_day_features, temporal_train_mask


def test_account_day_features_are_complete_and_unique():
    transactions = generate_demo_transactions(80, 600, 6)
    features = build_account_day_features(transactions)
    assert not features.duplicated(["date", "account_id"]).any()
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert features["is_laundering"].sum() > 0
    assert (features["total_value"] >= 0).all()


def test_temporal_mask_uses_earlier_dates_only():
    transactions = generate_demo_transactions(80, 600, 6)
    features = build_account_day_features(transactions)
    mask = temporal_train_mask(features)
    assert features.loc[mask, "date"].max() < features.loc[~mask, "date"].min()
