"""Command-line training and scoring pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib

from signalgraph_aml.data import generate_demo_transactions, load_transactions
from signalgraph_aml.evaluation import evaluate_alerts
from signalgraph_aml.experiments import record_experiment, sha256_file
from signalgraph_aml.features import build_account_day_features, temporal_train_mask
from signalgraph_aml.modeling import SignalGraphModel
from signalgraph_aml.monitoring import (
    compare_feature_drift,
    fit_feature_reference,
    write_drift_report,
)


def run_pipeline(
    transactions,
    output_dir: str | Path = "artifacts",
    alert_budget: int = 100,
    n_clusters: int = 5,
    include_explanations: bool = True,
    input_file: str | None = None,
    input_sha256: str | None = None,
):
    """Build cases, fit on early dates, score later dates, and persist artifacts."""

    started = perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = build_account_day_features(transactions)
    train_mask = temporal_train_mask(features)
    model = SignalGraphModel(n_clusters=n_clusters).fit(features.loc[train_mask])
    scored = model.score(features, include_explanations=include_explanations)
    evaluation_cases = scored.loc[~train_mask].copy()
    metrics = evaluate_alerts(evaluation_cases, alert_budget=alert_budget)
    training = features.loc[train_mask]
    later = features.loc[~train_mask]
    reference = fit_feature_reference(training, training["date"].max().isoformat())
    drift_report = compare_feature_drift(reference, later)

    scored.to_csv(output / "scored_cases.csv", index=False)
    evaluation_cases.sort_values("risk_score", ascending=False).to_csv(
        output / "investigation_queue.csv", index=False
    )
    joblib.dump(model, output / "signalgraph_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history = record_experiment(
        output,
        metrics=metrics,
        training=training,
        later=later,
        n_clusters=n_clusters,
        alert_budget=alert_budget,
        input_file=input_file,
        input_sha256=input_sha256,
        drift_report=drift_report,
        elapsed_seconds=perf_counter() - started,
    )
    (output / "experiment.json").write_text(
        (history / "experiment.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    reference_text = json.dumps(reference, indent=2)
    (output / "feature_reference.json").write_text(reference_text, encoding="utf-8")
    (history / "feature_reference.json").write_text(reference_text, encoding="utf-8")
    write_drift_report(drift_report, output)
    write_drift_report(drift_report, history)
    return scored, evaluation_cases, metrics, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use deterministic demo data")
    source.add_argument("--input", type=Path, help="path to an IBM AML transaction CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--alert-budget", type=int, default=100)
    parser.add_argument("--clusters", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions = generate_demo_transactions() if args.demo else load_transactions(args.input)
    _, evaluation_cases, metrics, _ = run_pipeline(
        transactions,
        output_dir=args.output_dir,
        alert_budget=args.alert_budget,
        n_clusters=args.clusters,
        input_file="demo" if args.demo else args.input.name,
        input_sha256=None if args.demo else sha256_file(args.input),
    )
    print(f"Scored {len(evaluation_cases):,} out-of-time account-day cases")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
