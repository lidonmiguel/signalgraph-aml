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

The settings are listed separately; no settings are selected using held-out labels. The run
records input SHA-256, sample case-ID hash, split dates, environment, noise fractions, cluster
sizes, PR-AUC, capacity metrics, runtime, and process-memory snapshots. Snapshots are not peak
memory measurements. A report generated with `--demo` or `--allow-small-input` is a smoke test,
not evidence for selecting a clustering method. Only the aggregated full-data output belongs in
`docs/benchmarks/cluster-comparison/`.
