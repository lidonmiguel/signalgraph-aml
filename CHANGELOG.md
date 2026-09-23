# Changelog

## Unreleased

- Recorded the full HI-Small K-Means/HDBSCAN comparison and retained the current K-Means model;
  the corrected report shows percentile saturation and the shared tie breaker.
- Resolved saturated percentile ties in the sampled clustering comparison with a shared
  global-anomaly secondary score and added tie diagnostics to the report.
- Added an isolated, reproducible account-sample experiment comparing MiniBatch K-Means and
  HDBSCAN on identical out-of-time cases, with explicit noise handling and aggregate reports.
- Added 60-minute fan-in alongside fan-out, ordered three- and four-account cycles,
  and scatter-gather diamonds as account-day model features.
- Added deterministic scatter-gather demo cases and motif tests covering event order,
  cross-midnight windows, and expired edges.
- The committed IBM report remains a historical pre-motif baseline until the full
  dataset is rerun with this model.

## v0.1 — 2026-07-17

- Published the reproducible IBM HI-Small benchmark on 5,078,345 transactions.
- Added a read-only dashboard tab for benchmark metrics, capacity trade-offs, behavioral
  segments, and reproducibility metadata.
- Added the interactive behavior map, investigation queue, network explorer, and methodology
  views.
- Added leakage-aware temporal evaluation, explainable cluster-relative anomaly scoring, tests,
  CI, Docker support, and data/model cards.

The project is a portfolio demonstration and not a production AML decision system.
