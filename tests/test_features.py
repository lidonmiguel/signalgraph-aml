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


def _cases_for_edges(edges):
    transactions = normalize_transactions(
        pd.DataFrame(
            {
                "timestamp": [timestamp for timestamp, _, _ in edges],
                "from_bank": ["BANK-1"] * len(edges),
                "from_account": [source for _, source, _ in edges],
                "to_bank": ["BANK-2"] * len(edges),
                "to_account": [target for _, _, target in edges],
                "amount_received": [10.0] * len(edges),
                "receiving_currency": ["Euro"] * len(edges),
                "amount_paid": [10.0] * len(edges),
                "payment_currency": ["Euro"] * len(edges),
                "payment_format": ["Wire"] * len(edges),
                "is_laundering": [0] * len(edges),
            }
        )
    )
    return build_account_day_features(transactions.iloc[::-1]).set_index(["date", "account_id"])


def test_fan_in_counts_distinct_senders_across_midnight():
    cases = _cases_for_edges(
        [
            ("2025-01-01 23:20", "A", "H"),
            ("2025-01-01 23:50", "B", "H"),
            ("2025-01-02 00:20", "C", "H"),
            ("2025-01-02 00:21", "C", "H"),
            ("2025-01-02 02:00", "H", "H"),
        ]
    )
    assert cases.loc[(pd.Timestamp("2025-01-01"), "H"), "max_in_senders_60m"] == 2
    assert cases.loc[(pd.Timestamp("2025-01-02"), "H"), "max_in_senders_60m"] == 3
    assert cases.loc[(pd.Timestamp("2025-01-02"), "C"), "max_in_senders_60m"] == 0


def test_rapid_cycle_requires_ordered_edges_within_three_hours():
    cases = _cases_for_edges(
        [
            ("2025-01-01 23:30", "A", "B"),
            ("2025-01-01 23:40", "B", "C"),
            ("2025-01-02 00:00", "C", "A"),
            ("2025-01-02 00:05", "B", "Q"),  # B has a case on the closing day
            ("2025-01-02 01:00", "D", "E"),
            ("2025-01-02 01:10", "E", "F"),
            ("2025-01-02 04:01", "F", "D"),  # first edge expired
            ("2025-01-02 05:00", "G", "H"),
            ("2025-01-02 05:00", "H", "I"),
            ("2025-01-02 05:00", "I", "G"),  # same-time order is unknown
            ("2025-01-02 06:00", "J", "K"),
            ("2025-01-02 06:10", "K", "L"),
            ("2025-01-02 09:00", "L", "J"),  # exactly three hours: included
        ]
    )
    day_one = pd.Timestamp("2025-01-01")
    day_two = pd.Timestamp("2025-01-02")
    assert cases.loc[(day_one, "A"), "rapid_cycle_3h"] == 0
    assert all(cases.loc[(day_two, account), "rapid_cycle_3h"] == 1 for account in "ABC")
    assert all(cases.loc[(day_two, account), "rapid_cycle_3h"] == 0 for account in "DEFGHI")
    assert all(cases.loc[(day_two, account), "rapid_cycle_3h"] == 1 for account in "JKL")


def test_four_account_cycle_matches_the_demo_pattern():
    cases = _cases_for_edges(
        [
            ("2025-01-01 12:00", "A", "B"),
            ("2025-01-01 12:10", "B", "C"),
            ("2025-01-01 12:20", "C", "D"),
            ("2025-01-01 12:30", "D", "A"),
            ("2025-01-01 14:00", "E", "F"),
            ("2025-01-01 14:10", "F", "G"),
            ("2025-01-01 14:20", "G", "H"),
            ("2025-01-01 17:01", "H", "E"),  # outside the three-hour window
        ]
    )
    day = pd.Timestamp("2025-01-01")
    assert all(cases.loc[(day, account), "rapid_cycle_3h"] == 1 for account in "ABCD")
    assert all(cases.loc[(day, account), "rapid_cycle_3h"] == 0 for account in "EFGH")


def test_scatter_gather_requires_two_ordered_paths_and_distinct_intermediaries():
    cases = _cases_for_edges(
        [
            ("2025-01-01 23:40", "S", "X"),
            ("2025-01-01 23:50", "S", "Y"),
            ("2025-01-02 00:05", "X", "T"),
            ("2025-01-02 00:10", "Y", "T"),
            ("2025-01-02 00:12", "S", "Q"),
            ("2025-01-02 09:00", "P", "M"),
            ("2025-01-02 09:02", "N", "V"),
            ("2025-01-02 09:10", "P", "N"),
            ("2025-01-02 09:20", "M", "V"),  # N -> V preceded P -> N
            ("2025-01-02 10:00", "J", "K"),
            ("2025-01-02 10:02", "K", "L"),
            ("2025-01-02 10:10", "J", "O"),
            ("2025-01-02 10:20", "O", "L"),  # gather began before scatter finished
            ("2025-01-02 11:00", "R", "U"),
            ("2025-01-02 11:00", "U", "W"),
            ("2025-01-02 11:10", "R", "Z"),
            ("2025-01-02 11:20", "Z", "W"),  # first path was simultaneous
        ]
    )
    day_one = pd.Timestamp("2025-01-01")
    day_two = pd.Timestamp("2025-01-02")
    assert cases.loc[(day_one, "S"), "scatter_gather_3h"] == 0
    assert all(
        cases.loc[(day_two, account), "scatter_gather_3h"] == 1 for account in "SXYT"
    )
    assert all(
        cases.loc[(day_two, account), "scatter_gather_3h"] == 0
        for account in "PMNVJKOLRUWZ"
    )


def test_demo_contains_completed_cycles_and_scatter_gather_paths():
    cases = build_account_day_features(generate_demo_transactions(80, 600, 6))
    assert cases["rapid_cycle_3h"].sum() > 0
    assert cases["scatter_gather_3h"].sum() > 0
