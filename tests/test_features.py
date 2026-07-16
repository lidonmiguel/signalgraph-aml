import pandas as pd

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


def test_temporal_mask_uses_case_volume_not_sparse_date_count():
    dates = pd.to_datetime(
        ["2024-01-01"] * 70
        + ["2024-01-02"] * 20
        + ["2024-01-03"] * 5
        + ["2024-01-04"] * 5
    )
    features = pd.DataFrame({"date": dates})

    mask = temporal_train_mask(features, train_fraction=0.7)

    assert mask.sum() == 70
    assert features.loc[mask, "date"].nunique() == 1
