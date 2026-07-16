import pandas as pd
import pytest

from signalgraph_aml.evaluation import evaluate_alerts, rank_alerts


def sample_scores():
    return pd.DataFrame(
        {
            "risk_score": [99, 90, 20, 10],
            "is_laundering": [1, 0, 1, 0],
            "total_value": [1000, 500, 250, 100],
        }
    )


def test_top_k_operational_metrics():
    metrics = evaluate_alerts(sample_scores(), alert_budget=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert metrics["lift_at_k"] == 1.0
    assert metrics["detected_value"] == 1000


def test_rank_alerts_validates_budget():
    with pytest.raises(ValueError):
        rank_alerts(sample_scores(), 0)
