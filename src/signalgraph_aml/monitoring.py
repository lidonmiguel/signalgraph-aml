"""Training-only feature reference and repeatable out-of-time drift checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signalgraph_aml.config import FEATURE_COLUMNS

REFERENCE_VERSION = 1
PSI_ALERT_THRESHOLD = 0.2  # Review heuristic, not a measure of model quality.
SMOOTHING = 1e-6


def _values(cases: pd.DataFrame, feature: str) -> np.ndarray:
    if feature not in cases:
        raise ValueError(f"Missing feature column: {feature}")
    values = pd.to_numeric(cases[feature], errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"Infinite feature values: {feature}")
    return values


def _counts(values: np.ndarray, spec: dict) -> list[int]:
    valid = values[~np.isnan(values)]
    if spec["mode"] == "categories":
        categories = np.asarray(spec["values"], dtype=float)
        if len(categories):
            indices = np.searchsorted(categories, valid)
            matched = (indices < len(categories)) & (
                categories[np.clip(indices, 0, len(categories) - 1)] == valid
            )
            indices = np.where(matched, indices, len(categories))
        else:
            indices = np.zeros(len(valid), dtype=int)
        bins = len(categories) + 1  # Last bucket: previously unseen values.
    else:
        cutpoints = np.asarray(spec["cutpoints"], dtype=float)
        indices = np.searchsorted(cutpoints, valid, side="left")
        bins = len(cutpoints) + 1
    counts = np.bincount(indices, minlength=bins).tolist()
    return [*counts, int(np.isnan(values).sum())]  # Dedicated missing-value bucket.


def fit_feature_reference(training: pd.DataFrame, training_date_max: str) -> dict:
    """Fit bin boundaries and frequencies on early account-days only; exclude labels/IDs."""
    if training.empty:
        raise ValueError("Training cases cannot be empty")
    specifications = {}
    for feature in FEATURE_COLUMNS:
        values = _values(training, feature)
        present = values[~np.isnan(values)]
        unique = np.unique(present)
        if len(unique) <= 10:
            spec = {"mode": "categories", "values": unique.tolist()}
        else:
            cutpoints = np.unique(np.quantile(present, np.arange(1, 10) / 10))
            spec = {"mode": "quantiles", "cutpoints": cutpoints.tolist()}
        spec["counts"] = _counts(values, spec)
        specifications[feature] = spec
    return {
        "schema_version": REFERENCE_VERSION,
        "training_date_max": training_date_max,
        "training_cases": len(training),
        "features": specifications,
    }


def compare_feature_drift(reference: dict, later: pd.DataFrame) -> dict:
    """Compare later cases to fixed training bins without inspecting outcome labels."""
    if reference.get("schema_version") != REFERENCE_VERSION:
        raise ValueError("Unsupported feature reference version")
    if later.empty:
        raise ValueError("Later cases cannot be empty")
    if "date" not in later:
        raise ValueError("Later cases must include date")
    dates = pd.to_datetime(later["date"], errors="raise")
    if dates.isna().any() or (dates <= pd.Timestamp(reference["training_date_max"])).any():
        raise ValueError("Every monitored case must be later than the training period")
    if list(reference["features"]) != FEATURE_COLUMNS:
        raise ValueError("Feature reference does not match the current model columns")

    rows = []
    for feature, spec in reference["features"].items():
        values = _values(later, feature)
        baseline = np.asarray(spec["counts"], dtype=float)
        current = np.asarray(_counts(values, spec), dtype=float)
        if baseline.size != current.size or int(baseline.sum()) != reference["training_cases"]:
            raise ValueError(f"Invalid training reference counts for {feature}")
        expected = (baseline + SMOOTHING) / (baseline.sum() + len(baseline) * SMOOTHING)
        observed = (current + SMOOTHING) / (current.sum() + len(current) * SMOOTHING)
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        rows.append({
            "feature": feature,
            "psi": round(psi, 6),
            "alert": psi >= PSI_ALERT_THRESHOLD,
            "training_missing_rate": float(baseline[-1] / baseline.sum()),
            "later_missing_rate": float(current[-1] / current.sum()),
            "training_counts": baseline.astype(int).tolist(),
            "later_counts": current.astype(int).tolist(),
        })
    return {
        "schema_version": REFERENCE_VERSION,
        "training_date_max": reference["training_date_max"],
        "training_cases": reference["training_cases"],
        "later_date_min": dates.min().isoformat(),
        "later_date_max": dates.max().isoformat(),
        "later_cases": len(later),
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "alert_features": [row["feature"] for row in rows if row["alert"]],
        "features": rows,
    }


def write_drift_report(report: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([
        {key: value for key, value in row.items() if key not in {"training_counts", "later_counts"}}
        for row in report["features"]
    ]).to_csv(output / "feature_drift.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="later account-day CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift-check"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    # Never load account IDs, outcomes, or scores into the monitoring command.
    later = pd.read_csv(args.cases, usecols=["date", *FEATURE_COLUMNS])
    report = compare_feature_drift(reference, later)
    write_drift_report(report, args.output_dir)
    print(
        f"Checked {report['later_cases']:,} later cases; "
        f"{len(report['alert_features'])} feature alerts"
    )
    print(f"Report: {args.output_dir / 'feature_drift.csv'}")


if __name__ == "__main__":
    main()
