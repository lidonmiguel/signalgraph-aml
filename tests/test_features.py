import pandas as pd

from signalgraph_aml.config import FEATURE_COLUMNS
from signalgraph_aml.data import generate_demo_transactions, normalize_transactions
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


def test_fan_out_uses_distinct_recipients_in_a_trailing_hour():
    payments = [
        ("2025-01-01 23:20:00", "X", "A"),
        ("2025-01-01 23:50:00", "X", "B"),
        ("2025-01-02 00:20:00", "X", "C"),  # 60 minutes after A: included
        ("2025-01-02 00:21:00", "X", "C"),  # repeated recipient
        ("2025-01-02 09:00:00", "Y", "A"),
        ("2025-01-02 09:20:00", "Y", "A"),
        ("2025-01-02 10:21:00", "Y", "B"),  # A has expired
        ("2025-01-02 10:30:00", "Y", "Y"),  # self-transfer is not fan-out
    ]
    transactions = normalize_transactions(
        pd.DataFrame(
            {
                "timestamp": [row[0] for row in payments],
                "from_bank": ["BANK-1"] * len(payments),
                "from_account": [row[1] for row in payments],
                "to_bank": ["BANK-2"] * len(payments),
                "to_account": [row[2] for row in payments],
                "amount_received": [10.0] * len(payments),
                "receiving_currency": ["Euro"] * len(payments),
                "amount_paid": [10.0] * len(payments),
                "payment_currency": ["Euro"] * len(payments),
                "payment_format": ["Wire"] * len(payments),
                "is_laundering": [0] * len(payments),
            }
        )
    )

    cases = build_account_day_features(transactions.iloc[::-1]).set_index(
        ["date", "account_id"]
    )
    fan_out = cases["max_out_recipients_60m"]

    assert fan_out.loc[(pd.Timestamp("2025-01-01"), "X")] == 2
    assert fan_out.loc[(pd.Timestamp("2025-01-02"), "X")] == 3
    assert fan_out.loc[(pd.Timestamp("2025-01-02"), "Y")] == 1
    assert fan_out.loc[(pd.Timestamp("2025-01-01"), "A")] == 0
    assert fan_out.loc[(pd.Timestamp("2025-01-02"), "C")] == 0
