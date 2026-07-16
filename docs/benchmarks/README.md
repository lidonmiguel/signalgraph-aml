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
