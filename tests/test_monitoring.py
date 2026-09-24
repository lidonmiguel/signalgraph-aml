import json

import pandas as pd
import pytest

from signalgraph_aml.config import FEATURE_COLUMNS
from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.monitoring import compare_feature_drift, fit_feature_reference
from signalgraph_aml.pipeline import run_pipeline


def _cases(size, date):
    frame = pd.DataFrame({column: [0.0] * size for column in FEATURE_COLUMNS})
    frame["date"] = pd.Timestamp(date)
    frame["account_id"] = [f"account-{index}" for index in range(size)]
    frame["is_laundering"] = 0
    return frame


def test_reference_is_training_only_and_catches_new_motifs_and_missingness():
    training = _cases(100, "2025-01-01")
    later = _cases(100, "2025-01-02")
    reference = fit_feature_reference(training, "2025-01-01T00:00:00")
    unchanged = compare_feature_drift(reference, later)
    assert unchanged["alert_features"] == []

    later["rapid_cycle_3h"] = 1
    later["flow_ratio"] = float("nan")
    later["is_laundering"] = 1
    later["account_id"] = "new-id"
    shifted = compare_feature_drift(reference, later)
    assert set(shifted["alert_features"]) == {"rapid_cycle_3h", "flow_ratio"}
    assert "is_laundering" not in json.dumps(reference) + json.dumps(shifted)
    assert "new-id" not in json.dumps(shifted)
    assert shifted["features"][FEATURE_COLUMNS.index("flow_ratio")]["later_missing_rate"] == 1.0
    assert reference["features"]["rapid_cycle_3h"]["counts"] == [100, 0, 0]


def test_monitor_rejects_overlap_and_changed_feature_schema():
    training = _cases(20, "2025-01-01")
    reference = fit_feature_reference(training, "2025-01-01T00:00:00")
    with pytest.raises(ValueError, match="later than"):
        compare_feature_drift(reference, training)
    with pytest.raises(ValueError, match="Missing feature"):
        compare_feature_drift(reference, _cases(20, "2025-01-02").drop(columns="out_total"))
    reference["features"].pop("out_total")
    with pytest.raises(ValueError, match="current model columns"):
        compare_feature_drift(reference, _cases(20, "2025-01-02"))


def test_pipeline_records_separate_runs_and_monitorable_reference(tmp_path):
    transactions = generate_demo_transactions(45, 300, 4)
    for _ in range(2):
        run_pipeline(transactions, output_dir=tmp_path, n_clusters=2, include_explanations=False)

    history = list((tmp_path / "experiments").iterdir())
    assert len(history) == 2
    latest = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    reference = json.loads((tmp_path / "feature_reference.json").read_text(encoding="utf-8"))
    later = pd.read_csv(
        tmp_path / "investigation_queue.csv", usecols=["date", *FEATURE_COLUMNS]
    )
    monitored = compare_feature_drift(reference, later)
    assert monitored["later_cases"] == latest["split"]["evaluation_cases"]
    assert (tmp_path / "feature_drift.csv").is_file()
    assert latest["run_id"] in {directory.name for directory in history}
    assert all((directory / "feature_reference.json").is_file() for directory in history)
