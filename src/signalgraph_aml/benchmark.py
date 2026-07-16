"""Reproducible IBM HI-Small benchmark runner and report generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from signalgraph_aml.data import load_transactions
from signalgraph_aml.evaluation import capacity_curve
from signalgraph_aml.pipeline import run_pipeline
from signalgraph_aml.profiling import build_cluster_profiles

DEFAULT_CAPACITIES = [50, 100, 250, 500, 1_000]


def run_benchmark(
    input_path: str | Path,
    output_dir: str | Path = "artifacts/ibm-hi-small",
    report_dir: str | Path | None = None,
    capacities: list[int] | None = None,
    n_clusters: int = 5,
    validate_dataset: bool = True,
) -> dict[str, object]:
    """Run the full transaction benchmark and write auditable result artifacts."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_capacities = capacities or DEFAULT_CAPACITIES

    started = perf_counter()
    transactions = load_transactions(source)
    if validate_dataset and len(transactions) < 5_000_000:
        raise ValueError(
            "IBM HI-Small should contain about 5.08 million transactions. "
            "Refusing to publish a partial-data benchmark; use --allow-small-input only "
            "for development smoke tests."
        )
    scored_cases, evaluation_cases, metrics, _ = run_pipeline(
        transactions,
        output_dir=output,
        alert_budget=100,
        n_clusters=n_clusters,
        include_explanations=False,
    )
    training_cases = scored_cases.loc[~scored_cases.index.isin(evaluation_cases.index)]
    curve = capacity_curve(evaluation_cases, requested_capacities)
    profiles = build_cluster_profiles(evaluation_cases)
    elapsed_seconds = perf_counter() - started

    curve.to_csv(output / "capacity_curve.csv", index=False)
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    summary: dict[str, object] = {
        "benchmark": "IBM AML HI-Small" if validate_dataset else "Development smoke test",
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "input_size_bytes": source.stat().st_size,
        "transactions": len(transactions),
        "training_cases": len(training_cases),
        "evaluation_cases": len(evaluation_cases),
        "evaluation_positives": int(evaluation_cases["is_laundering"].sum()),
        "evaluation_negatives": int((evaluation_cases["is_laundering"] == 0).sum()),
        "evaluation_prevalence": float(evaluation_cases["is_laundering"].mean()),
        "pr_auc": float(metrics["pr_auc"]),
        "date_min": transactions["timestamp"].min().isoformat(),
        "date_max": transactions["timestamp"].max().isoformat(),
        "training_date_min": training_cases["date"].min().isoformat(),
        "training_date_max": training_cases["date"].max().isoformat(),
        "evaluation_date_min": evaluation_cases["date"].min().isoformat(),
        "evaluation_date_max": evaluation_cases["date"].max().isoformat(),
        "clusters": n_clusters,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "capacity_results": curve.to_dict(orient="records"),
    }
    summary_text = json.dumps(summary, indent=2)
    report_text = _render_report(summary, curve, profiles)
    (output / "benchmark_summary.json").write_text(summary_text, encoding="utf-8")
    (output / "BENCHMARK_REPORT.md").write_text(report_text, encoding="utf-8")
    if report_dir is not None:
        publish = Path(report_dir)
        publish.mkdir(parents=True, exist_ok=True)
        (publish / "benchmark_summary.json").write_text(summary_text, encoding="utf-8")
        (publish / "BENCHMARK_REPORT.md").write_text(report_text, encoding="utf-8")
        curve.to_csv(publish / "capacity_curve.csv", index=False)
        profiles.to_csv(publish / "cluster_profiles.csv", index=False)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_report(
    summary: dict[str, object],
    curve: pd.DataFrame,
    profiles: pd.DataFrame,
) -> str:
    metric_rows = [
        "| Capacity | Hits | Precision | Recall | Lift | Positive-case value selected |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve.itertuples(index=False):
        metric_rows.append(
            f"| {int(row.capacity):,} | {int(row.hits):,} | {row.precision:.1%} | "
            f"{row.recall:.1%} | {row.lift:.2f}× | ${row.detected_value:,.2f} |"
        )

    profile_rows = [
        "| Segment | Behavioral name | Cases | Median transactions | Median value |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in profiles.itertuples(index=False):
        profile_rows.append(
            f"| {int(row.cluster)} | {row.cluster_name} | {int(row.cases):,} | "
            f"{row.median_transactions:,.1f} | ${row.median_value:,.2f} |"
        )

    environment = summary["environment"]
    assert isinstance(environment, dict)
    return "\n".join(
        [
            f"# {summary['benchmark']}",
            "",
            "> Generated by `signalgraph-benchmark`; labels were used only after scoring.",
            "",
            "## Dataset and run",
            "",
            f"- Input: `{summary['input_file']}`",
            f"- SHA-256: `{summary['input_sha256']}`",
            f"- Transactions: {int(summary['transactions']):,}",
            f"- Training account-days: {int(summary['training_cases']):,}",
            f"- Evaluation account-days: {int(summary['evaluation_cases']):,}",
            f"- Positive evaluation account-days: {int(summary['evaluation_positives']):,}",
            f"- Negative evaluation account-days: {int(summary['evaluation_negatives']):,}",
            f"- Evaluation prevalence: {float(summary['evaluation_prevalence']):.3%}",
            f"- Training period: {summary['training_date_min']} to {summary['training_date_max']}",
            "- Evaluation period: "
            f"{summary['evaluation_date_min']} to {summary['evaluation_date_max']}",
            f"- PR-AUC: {float(summary['pr_auc']):.4f}",
            f"- Runtime: {float(summary['elapsed_seconds']):,.2f} seconds",
            f"- Python: {environment['python']}",
            f"- scikit-learn: {environment['scikit_learn']}",
            "",
            "## Capacity results",
            "",
            *metric_rows,
            "",
            "## Behavioral segments",
            "",
            *profile_rows,
            "",
            "## Interpretation",
            "",
            "Capacity changes the number of ranked cases reviewed; it does not refit the model. ",
            "These results are a synthetic benchmark, not evidence of production performance.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ibm-hi-small"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--capacities", type=int, nargs="+", default=DEFAULT_CAPACITIES)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument(
        "--allow-small-input",
        action="store_true",
        help="disable the full-dataset size guard for development smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        args.input,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        capacities=args.capacities,
        n_clusters=args.clusters,
        validate_dataset=not args.allow_small_input,
    )
    print(f"Benchmark complete in {summary['elapsed_seconds']:,.2f} seconds")
    print(f"Report: {args.output_dir / 'BENCHMARK_REPORT.md'}")


if __name__ == "__main__":
    main()
