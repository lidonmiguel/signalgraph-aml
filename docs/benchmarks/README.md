# Benchmark reports

This directory stores small, reproducible benchmark summaries—not raw transactions, trained
models, or full scored case tables.

Run the IBM HI-Small benchmark from the repository root:

```bash
signalgraph-benchmark \
  --input data/raw/HI-Small_Trans.csv \
  --output-dir artifacts/ibm-hi-small \
  --report-dir docs/benchmarks/ibm-hi-small \
  --capacities 50 100 250 500 1000
```

The generated directory contains:

- `BENCHMARK_REPORT.md` — human-readable results
- `benchmark_summary.json` — dataset hash, environment, runtime, and metrics
- `capacity_curve.csv` — precision, recall, lift, and hits by alert capacity
- `cluster_profiles.csv` — behavioral segment names and medians
- `feature_drift.csv` — aggregate training versus out-of-time PSI and missingness

Every benchmark run also stores `experiment.json`, `feature_reference.json`, and
`feature_drift.json` locally under its ignored `artifacts/` output directory and saves
immutable copies in `experiments/<run-id>/`. The JSON reference includes training-only bin
boundaries and counts. The committed benchmark directory gets the aggregate drift CSV, not
the per-case tables or reference. The experiment record captures source hash, Git revision if
available, parameters, split, environment, held-out metrics, and runtime. Each clustering
comparison saves its summary and the same training-versus-later drift outputs in its own
`experiments/<run-id>/` history.

To check later account-days again without rerunning the model, use `signalgraph-drift
--reference artifacts/ibm-hi-small/feature_reference.json --cases
artifacts/ibm-hi-small/investigation_queue.csv --output-dir artifacts/drift-check`.
For an independent later batch, pass a CSV with `date` and every model feature; every date
must be strictly after the training end. The command reads no labels or account IDs.

The reference records each rare or constant feature's observed values plus a separate
previously unseen bucket. Higher-cardinality features use training decile cutpoints and a
separate missingness bucket. PSI compares smoothed training and later bucket frequencies;
the 0.2 alert threshold is a review heuristic. A flag does not prove degradation or imply
automatic retraining. For temporal monitoring, run the command on separate later batches
and compare their reports; keep the reference and feature schema fixed.

Raw IBM data remains under `data/raw/` and is excluded from Git.

The benchmark skips per-case prose generation because the committed report uses only model
scores and aggregate profiles. The normal CLI and dashboard continue to generate alert reasons.

The published full-data result is available in
[`ibm-hi-small/BENCHMARK_REPORT.md`](ibm-hi-small/BENCHMARK_REPORT.md).

## Cluster comparison

Install the optional dependencies with `python -m pip install -e ".[dev,comparison]"`.
Run `signalgraph-compare-clusters --demo --min-cluster-sizes 30 60` to check the
experiment with synthetic demo data. For a full comparison, run:

```bash
signalgraph-compare-clusters \
  --input data/raw/HI-Small_Trans.csv \
  --output-dir artifacts/cluster-comparison \
  --report-dir docs/benchmarks/cluster-comparison \
  --sample-cases 20000 \
  --min-cluster-sizes 50 100 200 \
  --min-samples 15 \
  --capacities 50 100 250 500 1000
```

The sample draws accounts proportionally from training-only activity quintiles and retains
all their training account-days; the exact sample size may differ from the target. Both sampled
models use the same transformed columns, robust scaler, global Isolation Forest, conditional
cluster detectors, and anomaly-percentile ranking. The production K-Means model uses the full
training set and its existing 82/18 risk formula as a separate reference. Later cases and budgets
are identical across all arms. HDBSCAN's `-1` noise uses the global detector and a reference
of all sampled training cases. If it finds zero clusters, its results represent global scoring
only and should not be interpreted as a successful clustering method.

Small HDBSCAN clusters create coarse empirical anomaly percentiles. Many later cases can
exceed a cluster's training maximum and receive the same primary percentile. Both sampled
methods therefore use the *same* global Isolation Forest anomaly score as a tiny secondary
sort key. It resolves tied primary percentiles without changing the order of unequal ones.
The report lists saturation and ties at the review cutoffs before the secondary score is applied.

The settings are listed separately; no settings are selected using held-out labels. The run
records input SHA-256, sample case-ID hash, split dates, environment, noise fractions, cluster
sizes, PR-AUC, capacity metrics, runtime, and process-memory snapshots. Snapshots are not peak
memory measurements. A report generated with `--demo` or `--allow-small-input` is a smoke test,
not evidence for selecting a clustering method. Only the aggregated full-data output belongs in
`docs/benchmarks/cluster-comparison/`.

The reviewed full-data outcome and its limitations are recorded in
[`cluster-comparison/RESULTS.md`](cluster-comparison/RESULTS.md). The original
run's aggregate `COMPARISON_REPORT.md` and `comparison_summary.json` were
generated on the machine holding the CSV; the committed summary contains
the comparison metrics, input hash, sample hash, and split boundary.
