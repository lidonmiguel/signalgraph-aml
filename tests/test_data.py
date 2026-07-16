import pandas as pd
import pytest

from signalgraph_aml.data import generate_demo_transactions, normalize_transactions


def test_normalize_ibm_column_names():
    raw = pd.DataFrame(
        {
            "Timestamp": ["2025-01-01 10:00"],
            "From Bank": [1],
            "Account": ["A"],
            "To Bank": [2],
            "Account.1": ["B"],
            "Amount Received": [10.0],
            "Receiving Currency": ["Euro"],
            "Amount Paid": [10.0],
            "Payment Currency": ["Euro"],
            "Payment Format": ["Wire"],
            "Is Laundering": [0],
        }
    )
    normalized = normalize_transactions(raw)
    assert normalized.loc[0, "from_account"] == "A"
    assert normalized.loc[0, "to_account"] == "B"
    assert pd.api.types.is_datetime64_any_dtype(normalized["timestamp"])


def test_demo_is_deterministic_and_contains_hidden_patterns():
    first = generate_demo_transactions(80, 500, 5, random_state=7)
    second = generate_demo_transactions(80, 500, 5, random_state=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["is_laundering"].sum() > 0
    assert len(first) > 500


def test_demo_rejects_tiny_inputs():
    with pytest.raises(ValueError):
        generate_demo_transactions(10, 20, 2)
