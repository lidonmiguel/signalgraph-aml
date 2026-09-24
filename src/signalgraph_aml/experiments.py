"""Local, append-only experiment records without a tracking service."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn

from signalgraph_aml.config import FEATURE_COLUMNS, RANDOM_STATE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def new_experiment_directory(output_dir: str | Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    directory = Path(output_dir) / "experiments" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def record_experiment(
    output_dir: str | Path,
    *,
    metrics: dict,
    training: pd.DataFrame,
    later: pd.DataFrame,
    n_clusters: int,
    alert_budget: int,
    input_file: str | None,
    input_sha256: str | None,
    drift_report: dict,
    elapsed_seconds: float,
) -> Path:
    """Write a new run, preserving earlier records even if output files are replaced."""
    directory = new_experiment_directory(output_dir)
    run_id = directory.name
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_file,
        "input_sha256": input_sha256,
        "code_revision": code_revision(),
        "parameters": {
            "n_clusters": n_clusters,
            "alert_budget": alert_budget,
            "random_state": RANDOM_STATE,
            "feature_columns": FEATURE_COLUMNS,
            "model": "SignalGraphModel",
        },
        "split": {
            "training_cases": len(training),
            "training_date_min": training["date"].min().isoformat(),
            "training_date_max": training["date"].max().isoformat(),
            "evaluation_cases": len(later),
            "evaluation_date_min": later["date"].min().isoformat(),
            "evaluation_date_max": later["date"].max().isoformat(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "drift_alert_features": drift_report["alert_features"],
        "artifacts": ["feature_reference.json", "feature_drift.json", "feature_drift.csv"],
    }
    (directory / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return directory
