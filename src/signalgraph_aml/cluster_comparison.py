"""Compare sampled K-Means and HDBSCAN on the same out-of-time account-days."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import hdbscan
import numpy as np
import pandas as pd
import psutil
import sklearn
from hdbscan.prediction import approximate_predict
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from signalgraph_aml.benchmark import DEFAULT_CAPACITIES, _sha256
from signalgraph_aml.config import FEATURE_COLUMNS, RANDOM_STATE
from signalgraph_aml.data import generate_demo_transactions, load_transactions
from signalgraph_aml.evaluation import capacity_curve, evaluate_alerts
from signalgraph_aml.features import build_account_day_features, temporal_train_mask
from signalgraph_aml.modeling import SignalGraphModel, _percentile_rank, prepare_model_features


def representative_account_sample(
    training: pd.DataFrame, target_cases: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Sample accounts proportionally across training-only activity quintiles.

    Keep *all* training account-days for selected accounts. The target is approximate
    because accounts have different numbers of active days. No outcomes or future
    dates enter the sampling rule.
    """

    if target_cases < 50:
        raise ValueError("sample-cases must be at least 50")
    if len(training) <= target_cases:
        return training.copy()

    activity = (
        training.groupby("account_id", sort=True, observed=True)["total_tx_count"]
        .sum()
        .sort_index()
    )
    groups = pd.qcut(activity.rank(method="first"), q=min(5, len(activity)), labels=False)
    fraction = target_cases / len(training)
    rng = np.random.default_rng(random_state)
    selected: list[str] = []
    for group in sorted(groups.unique()):
        accounts = activity.index[groups == group].to_numpy()
        count = min(len(accounts), max(1, round(len(accounts) * fraction)))
        selected.extend(rng.choice(accounts, size=count, replace=False).tolist())
    return training.loc[training["account_id"].isin(selected)].copy()


def _detector(seed: int, estimators: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=estimators,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )


def _resolve_percentile_ties(percentiles: np.ndarray, global_anomaly: np.ndarray) -> np.ndarray:
    """Rank equal cluster percentiles by the common global detector.

    Empirical percentiles saturate at one above each cluster's training maximum.
    With a 20k-case sample and 720k evaluation cases, this can leave hundreds
    tied at the top. The tiny secondary term cannot overtake a different
    percentile in the sampled training reference.
    """

    weight = 1e-9
    return 100 * ((1 - weight) * percentiles + weight * global_anomaly)


def _global_evaluation_anomaly(
    evaluation: pd.DataFrame,
    scaler: RobustScaler,
    global_detector: IsolationForest,
    chunk_size: int,
) -> np.ndarray:
    """Compute the same label-free tie breaker once for all sampled arms."""

    anomaly = np.empty(len(evaluation), dtype=float)
    for start in range(0, len(evaluation), chunk_size):
        end = min(start + chunk_size, len(evaluation))
        scaled = scaler.transform(prepare_model_features(evaluation.iloc[start:end]))
        anomaly[start:end] = -global_detector.score_samples(scaled)
    return anomaly


def _score_clustered(
    training_scaled: np.ndarray,
    training_labels: np.ndarray,
    evaluation: pd.DataFrame,
    scaler: RobustScaler,
    assign: Callable[[np.ndarray], np.ndarray],
    global_detector: IsolationForest,
    global_reference: np.ndarray,
    global_evaluation_anomaly: np.ndarray,
    *,
    random_state: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use identical cluster-relative anomaly scoring in both sampled arms.

    Noise (-1) uses the global detector calibrated against *all* sampled training
    cases. Small non-noise clusters also use the global detector, but retain their
    own reference distribution, matching the existing model's small-cluster rule.
    """

    detectors: dict[int, IsolationForest] = {}
    references: dict[int, np.ndarray] = {-1: global_reference}
    minimum = max(30, len(FEATURE_COLUMNS) * 3)
    for cluster in np.unique(training_labels):
        if cluster < 0:
            continue
        mask = training_labels == cluster
        detector = (
            _detector(random_state, 200).fit(training_scaled[mask])
            if mask.sum() >= minimum
            else global_detector
        )
        detectors[int(cluster)] = detector
        references[int(cluster)] = np.sort(-detector.score_samples(training_scaled[mask]))

    primary_percentiles = np.empty(len(evaluation), dtype=float)
    labels = np.empty(len(evaluation), dtype=int)
    for start in range(0, len(evaluation), chunk_size):
        end = min(start + chunk_size, len(evaluation))
        scaled = scaler.transform(prepare_model_features(evaluation.iloc[start:end]))
        assigned = np.asarray(assign(scaled), dtype=int)
        labels[start:end] = assigned
        if set(np.unique(assigned)) - set(detectors) - {-1}:
            raise ValueError("Prediction returned a cluster absent from training")
        global_raw = global_evaluation_anomaly[start:end]
        for cluster in np.unique(assigned):
            mask = assigned == cluster
            detector = global_detector if cluster == -1 else detectors[int(cluster)]
            raw = (
                global_raw[mask]
                if detector is global_detector
                else -detector.score_samples(scaled[mask])
            )
            primary_percentiles[start:end][mask] = _percentile_rank(raw, references[int(cluster)])
    scores = _resolve_percentile_ties(primary_percentiles, global_evaluation_anomaly)
    return scores, labels, primary_percentiles


def _result(
    name: str,
    training: pd.DataFrame,
    training_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    risk: np.ndarray,
    evaluation: pd.DataFrame,
    capacities: list[int],
    elapsed_seconds: float,
    observed_rss_mib: float,
    risk_definition: str,
    primary_percentiles: np.ndarray | None = None,
) -> dict[str, object]:
    scored = evaluation[["is_laundering", "total_value"]].copy()
    scored["risk_score"] = risk
    metrics = evaluate_alerts(scored, alert_budget=min(100, len(scored)))
    counts = pd.Series(training_labels).value_counts().sort_index()
    clustered = counts.loc[counts.index >= 0]
    primary = risk if primary_percentiles is None else primary_percentiles
    cutoff_ties: dict[str, int] = {}
    for capacity in capacities:
        budget = min(capacity, len(primary))
        cutoff = np.partition(primary, len(primary) - budget)[len(primary) - budget]
        cutoff_ties[str(budget)] = int(np.count_nonzero(primary == cutoff))
    profiles = training[
        ["total_tx_count", "total_value", "rapid_cycle_3h", "scatter_gather_3h"]
    ].copy()
    profiles["cluster"] = training_labels
    profiles = profiles.groupby("cluster", sort=True).agg(
        cases=("cluster", "size"),
        median_transactions=("total_tx_count", "median"),
        median_value=("total_value", "median"),
        rapid_cycle_share=("rapid_cycle_3h", "mean"),
        scatter_gather_share=("scatter_gather_3h", "mean"),
    )
    return {
        "method": name,
        "risk_definition": risk_definition,
        "training_clusters": int(len(clustered)),
        "training_cluster_sizes": {str(int(k)): int(v) for k, v in clustered.items()},
        "training_profiles": [
            {"cluster": int(cluster), **{key: float(value) for key, value in row.items()}}
            for cluster, row in profiles.to_dict(orient="index").items()
        ],
        "training_noise_fraction": float(np.mean(training_labels == -1)),
        "evaluation_noise_fraction": float(np.mean(evaluation_labels == -1)),
        "primary_score_saturation_cases": int(np.count_nonzero(primary == 1))
        if primary_percentiles is not None
        else int(np.count_nonzero(primary == 100)),
        "primary_score_ties_at_cutoff": cutoff_ties,
        "pr_auc": float(metrics["pr_auc"]),
        "capacity_results": capacity_curve(scored, capacities).to_dict(orient="records"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "observed_rss_mib": round(observed_rss_mib, 1),
    }


def compare_account_days(
    features: pd.DataFrame,
    *,
    sample_cases: int = 20_000,
    min_cluster_sizes: tuple[int, ...] = (50, 100, 200),
    min_samples: int = 15,
    n_clusters: int = 5,
    capacities: list[int] | None = None,
    random_state: int = RANDOM_STATE,
    chunk_size: int = 25_000,
) -> dict[str, object]:
    """Evaluate all models on the same held-out population, never fit on its labels."""

    if chunk_size < 1 or n_clusters < 2 or min_samples < 1:
        raise ValueError("chunk-size must be positive; clusters >= 2; min-samples >= 1")
    if not min_cluster_sizes or any(size < 2 for size in min_cluster_sizes):
        raise ValueError("min-cluster-sizes must contain positive sizes >= 2")
    train_mask = temporal_train_mask(features)
    training = features.loc[train_mask]
    evaluation = features.loc[~train_mask]
    sampled = representative_account_sample(training, sample_cases, random_state)
    if len(sampled) < n_clusters * 10:
        raise ValueError("The account sample is too small for the requested K-Means clusters")
    if max(min_cluster_sizes) >= len(sampled):
        raise ValueError("Every HDBSCAN min-cluster-size must be smaller than the sample")
    requested_capacities = capacities if capacities is not None else DEFAULT_CAPACITIES
    if not requested_capacities or any(capacity < 1 for capacity in requested_capacities):
        raise ValueError("At least one positive capacity is required")

    process = psutil.Process()
    mib = 1024 * 1024
    results: list[dict[str, object]] = []

    # This is the *current* full-training model, included as an operational reference.
    started = perf_counter()
    full_model = SignalGraphModel(n_clusters=n_clusters, random_state=random_state).fit(training)
    full_training_labels = full_model.clusterer.labels_
    operational = full_model.score(evaluation, include_explanations=False)
    results.append(
        _result(
            "kmeans_full_operational",
            training,
            full_training_labels,
            operational["cluster"].to_numpy(),
            operational["risk_score"].to_numpy(),
            evaluation,
            requested_capacities,
            perf_counter() - started,
            process.memory_info().rss / mib,
            "82% anomaly percentile + 18% cluster-center distance percentile",
        )
    )
    del operational, full_model

    # The experimental arms use the same sample, scaler, detector rule, and score.
    shared_started = perf_counter()
    scaler = RobustScaler(quantile_range=(10, 90))
    training_scaled = scaler.fit_transform(prepare_model_features(sampled))
    global_detector = _detector(random_state, 250).fit(training_scaled)
    global_reference = np.sort(-global_detector.score_samples(training_scaled))
    global_evaluation_anomaly = _global_evaluation_anomaly(
        evaluation, scaler, global_detector, chunk_size
    )
    shared_setup_seconds = round(perf_counter() - shared_started, 2)

    started = perf_counter()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=1_024, n_init=10, random_state=random_state
    )
    kmeans_labels = kmeans.fit_predict(training_scaled)
    risk, labels, primary = _score_clustered(
        training_scaled,
        kmeans_labels,
        evaluation,
        scaler,
        kmeans.predict,
        global_detector,
        global_reference,
        global_evaluation_anomaly,
        random_state=random_state,
        chunk_size=chunk_size,
    )
    results.append(
        _result(
            "kmeans_sampled_anomaly",
            sampled,
            kmeans_labels,
            labels,
            risk,
            evaluation,
            requested_capacities,
            perf_counter() - started,
            process.memory_info().rss / mib,
            "cluster-relative anomaly percentile; equal percentiles use global anomaly",
            primary,
        )
    )
    del risk, labels, primary, kmeans

    for size in dict.fromkeys(min_cluster_sizes):
        started = perf_counter()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=size,
            min_samples=min_samples,
            prediction_data=True,
            core_dist_n_jobs=1,
        )
        labels_train = clusterer.fit_predict(training_scaled)
        has_clusters = np.any(labels_train >= 0)

        def assign(scaled: np.ndarray) -> np.ndarray:
            if not has_clusters:
                return np.full(len(scaled), -1, dtype=int)
            predicted, _strength = approximate_predict(clusterer, scaled)
            return predicted

        risk, labels, primary = _score_clustered(
            training_scaled,
            labels_train,
            evaluation,
            scaler,
            assign,
            global_detector,
            global_reference,
            global_evaluation_anomaly,
            random_state=random_state,
            chunk_size=chunk_size,
        )
        results.append(
            _result(
                f"hdbscan_mcs_{size}_ms_{min_samples}",
                sampled,
                labels_train,
                labels,
                risk,
                evaluation,
                requested_capacities,
                perf_counter() - started,
                process.memory_info().rss / mib,
                "cluster anomaly percentile; ties use global anomaly; noise uses global reference",
                primary,
            )
        )
        del risk, labels, primary

    case_ids = "\n".join(sampled["case_id"].sort_values()).encode("utf-8")
    return {
        "experiment": "K-Means versus HDBSCAN on account-day behavior",
        "seed": random_state,
        "features": FEATURE_COLUMNS,
        "training_cases": len(training),
        "training_accounts": int(training["account_id"].nunique()),
        "sample_target_cases": sample_cases,
        "sample_cases": len(sampled),
        "sample_accounts": int(sampled["account_id"].nunique()),
        "sample_case_ids_sha256": hashlib.sha256(case_ids).hexdigest(),
        "sample_rule": (
            "Account sampling proportional across quintiles of training transaction activity; "
            "all selected training account-days retained"
        ),
        "training_date_max": training["date"].max().isoformat(),
        "evaluation_date_min": evaluation["date"].min().isoformat(),
        "evaluation_cases": len(evaluation),
        "evaluation_positives": int(evaluation["is_laundering"].sum()),
        "kmeans_clusters": n_clusters,
        "hdbscan_min_cluster_sizes": list(dict.fromkeys(min_cluster_sizes)),
        "hdbscan_min_samples": min_samples,
        "chunk_size": chunk_size,
        "shared_sample_setup_seconds": shared_setup_seconds,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "hdbscan": version("hdbscan"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "results": results,
    }


def _report(summary: dict[str, object]) -> str:
    rows = [
        "| Method | K | Hits | Precision | Recall | Lift | PR-AUC | "
        "Train clusters | Test noise | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        for point in result["capacity_results"]:
            rows.append(
                f"| {result['method']} | {point['capacity']} | {point['hits']} | "
                f"{point['precision']:.2%} | {point['recall']:.2%} | {point['lift']:.2f}× | "
                f"{result['pr_auc']:.5f} | {result['training_clusters']} | "
                f"{result['evaluation_noise_fraction']:.1%} | {result['elapsed_seconds']:.1f} |"
            )
    tie_rows = [
        "| Method | Primary scores at maximum | Ties at K=50 | Ties at K=100 |",
        "|---|---:|---:|---:|",
    ]
    for result in summary["results"]:
        ties = result["primary_score_ties_at_cutoff"]
        tie_rows.append(
            f"| {result['method']} | {result['primary_score_saturation_cases']:,} | "
            f"{ties.get('50', 'n/a')} | {ties.get('100', 'n/a')} |"
        )
    profiles = [
        "| Method | Cluster | Cases | Median transactions | Median value | "
        "Cycle share | Diamond share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        leading = sorted(
            result["training_profiles"], key=lambda profile: profile["cases"], reverse=True
        )[:8]
        noise = next((p for p in result["training_profiles"] if p["cluster"] == -1), None)
        if noise is not None and noise not in leading:
            leading.append(noise)
        for profile in leading:
            profiles.append(
                f"| {result['method']} | {profile['cluster']} | {profile['cases']:,.0f} | "
                f"{profile['median_transactions']:.1f} | ${profile['median_value']:,.2f} | "
                f"{profile['rapid_cycle_share']:.1%} | "
                f"{profile['scatter_gather_share']:.1%} |"
            )
    return "\n".join(
        [
            "# Account-sample clustering comparison",
            "",
            "> Generated by `signalgraph-compare-clusters`; this is an experiment, "
            "not a model switch.",
            "",
            f"- Input: `{summary['input_file']}` (SHA-256: `{summary['input_sha256']}`)",
            f"- Transactions: {summary['transactions']:,}; training account-days: "
            f"{summary['training_cases']:,}",
            f"- Sample: {summary['sample_cases']:,} account-days from "
            f"{summary['sample_accounts']:,} accounts",
            f"- Sample case-ID hash: `{summary['sample_case_ids_sha256']}`; "
            f"seed: {summary['seed']}",
            f"- Sampling: {summary['sample_rule']}",
            f"- Train through {summary['training_date_max']}; "
            f"evaluate from {summary['evaluation_date_min']}",
            f"- Same held-out cases: {summary['evaluation_cases']:,}, "
            f"of which {summary['evaluation_positives']:,} positive",
            f"- HDBSCAN min-cluster-size: {summary['hdbscan_min_cluster_sizes']}; "
            f"min-samples: {summary['hdbscan_min_samples']}",
            f"- Feature engineering: {summary['feature_seconds']:.2f} seconds; "
            f"shared sample setup: {summary['shared_sample_setup_seconds']:.2f} seconds",
            "",
            "## Results",
            "",
            *rows,
            "",
            "## Percentile saturation (before the shared tie breaker)",
            "",
            *tie_rows,
            "",
            "## Training segment profiles (largest eight per method, plus noise)",
            "",
            *profiles,
            "",
            "The full K-Means row uses all early account-days and the production 82/18 score.",
            "The sampled K-Means and HDBSCAN rows share the same training sample, scaler,",
            "Isolation Forest configuration, and anomaly-only percentile score. Equal",
            "percentiles are ordered by the same global Isolation Forest raw anomaly",
            "score, with a tiny weight that preserves different primary percentiles. Noise",
            "gets the global detector and its all-training reference. HDBSCAN assigns later",
            "cases approximately to *existing* clusters; it does not refit on future data.",
            "",
            "Scores and settings were fixed without reference to laundering outcomes.",
            "Labels were used only for the out-of-time metrics above. K is a budget for",
            "the entire held-out queue, not a daily investigation capacity. A zero-cluster",
            "HDBSCAN run scores everything with the global detector and is inconclusive",
            "as a clustering comparison. Demo and partial-data results are smoke tests.",
            "Elapsed time is per method, excluding shared feature engineering and sample setup;",
            "observed RSS is only",
            "a process-memory snapshot at method completion (see the JSON report).",
            "",
        ]
    )


def run_comparison(
    transactions: pd.DataFrame,
    output_dir: str | Path,
    *,
    input_file: str = "deterministic demo",
    input_sha256: str | None = None,
    report_dir: str | Path | None = None,
    **settings: object,
) -> dict[str, object]:
    """Build features once and persist only aggregate comparison results."""

    started = perf_counter()
    features = build_account_day_features(transactions)
    feature_seconds = perf_counter() - started
    summary = compare_account_days(features, **settings)
    summary.update(
        {
            "input_file": input_file,
            "input_sha256": input_sha256,
            "transactions": len(transactions),
            "feature_seconds": round(feature_seconds, 2),
            "total_seconds": round(perf_counter() - started, 2),
        }
    )
    for path in (output_dir, report_dir):
        if path is None:
            continue
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "comparison_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (directory / "COMPARISON_REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cluster-comparison"))
    parser.add_argument("--report-dir", type=Path, help="optional directory for aggregate reports")
    parser.add_argument("--sample-cases", type=int, default=20_000)
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--capacities", type=int, nargs="+", default=DEFAULT_CAPACITIES)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--allow-small-input", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions = generate_demo_transactions() if args.demo else load_transactions(args.input)
    if args.input and not args.allow_small_input and len(transactions) < 5_000_000:
        raise ValueError("HI-Small requires ~5.08M rows; use --allow-small-input for smoke tests")
    summary = run_comparison(
        transactions,
        args.output_dir,
        input_file="deterministic demo" if args.demo else args.input.name,
        input_sha256=None if args.demo else _sha256(args.input),
        report_dir=args.report_dir,
        sample_cases=args.sample_cases,
        min_cluster_sizes=tuple(args.min_cluster_sizes),
        min_samples=args.min_samples,
        n_clusters=args.clusters,
        capacities=args.capacities,
        chunk_size=args.chunk_size,
    )
    print(f"Comparison complete in {summary['total_seconds']:.2f} seconds")
    print(f"Report: {args.output_dir / 'COMPARISON_REPORT.md'}")


if __name__ == "__main__":
    main()
