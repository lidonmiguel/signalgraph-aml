from pathlib import Path

import pytest

from signalgraph_aml.benchmark import run_benchmark
from signalgraph_aml.data import generate_demo_transactions


def test_benchmark_requires_a_real_input_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_benchmark(tmp_path / "HI-Small_Trans.csv")


def test_small_benchmark_publishes_aggregate_drift(tmp_path: Path):
    source = tmp_path / "demo.csv"
    generate_demo_transactions(45, 300, 4).to_csv(source, index=False)
    summary = run_benchmark(
        source,
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "report",
        n_clusters=2,
        validate_dataset=False,
    )
    assert summary["input_sha256"]
    assert summary["experiment_run_id"]
    assert (tmp_path / "report" / "feature_drift.csv").is_file()
    assert (tmp_path / "artifacts" / "experiments" / summary["experiment_run_id"]).is_dir()
