<div align="center">

# SignalGraph AML

**Explainable unsupervised detection of suspicious banking activity**

[![CI](https://github.com/lidonmiguel/signalgraph-aml/actions/workflows/ci.yml/badge.svg)](https://github.com/lidonmiguel/signalgraph-aml/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-investigation_console-FF4B4B?logo=streamlit&logoColor=white)
![Learning](https://img.shields.io/badge/ML-unsupervised-39E6B0)

</div>

SignalGraph AML is a portfolio-ready financial-crime analytics project. It learns ordinary
account behavior, discovers customer segments, and ranks unusual account-days for human review.
The model never sees laundering labels during training; ground truth is revealed only after
scoring to measure performance under a fixed investigation budget.

> **Decision problem:** if an AML team can investigate only 100 alerts today, which accounts
> should it review first—and why?

<img width="1701" height="811" alt="image" src="https://github.com/user-attachments/assets/36fdd8fd-43fa-4607-9d7b-36ecd912d8b6" />

## What is already working

- A deterministic synthetic demo, so the complete project runs without restricted bank data.
- IBM AML schema validation and account-day feature engineering.
- Temporal training on the earliest 70% of dates.
- Behavioral segmentation with MiniBatch K-Means.
- Cluster-relative anomaly scoring with Isolation Forest.
- Human-readable alert reasons and a two-dimensional behavior map.
- Precision/recall capacity planning, PR-AUC, lift, and positive-case value selected.
- Data-driven names and median profiles for every behavioral segment.
- Multi-factor explanations that expose small denominators such as “1 of 1 (100%).”
- A dark Streamlit investigation console with a ranked queue and account network explorer.
- Tests, linting, GitHub Actions, a CLI, and Docker support.

Demo metrics are **smoke-test results**, not claims about performance on real banking data.
The repository includes a separate, auditable IBM benchmark command that records the input hash,
software environment, runtime, capacity curve, and segment profiles.

## Dashboard

The application contains five analyst views:

1. **Behavior map** — PCA projection colored by anomaly risk, with an alert-capacity threshold.
2. **Capacity planning** — precision and recall across different analyst workloads.
3. **Investigation queue** — cases ranked by risk with interpretable alert reasons.
4. **Network explorer** — a one-hop view of counterparties and value moved on the alert date.
5. **Methodology** — a concise record of the leakage-aware experimental design.

Ground-truth outcomes are hidden by default and can be revealed for evaluation.

## Quick start

```bash
git clone https://github.com/lidonmiguel/signalgraph-aml.git
cd signalgraph-aml
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

Run the reproducible command-line demo:

```bash
signalgraph-aml --demo --alert-budget 100
```

This writes the trained model, scored cases, investigation queue, and metrics to `artifacts/`.

## Use the IBM AML benchmark

1. Download `HI-Small_Trans.csv` from the
   [IBM Transactions for Anti Money Laundering dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
2. Place it at `data/raw/HI-Small_Trans.csv`. Raw data is git-ignored.
3. Run the auditable benchmark:

```bash
signalgraph-benchmark \
  --input data/raw/HI-Small_Trans.csv \
  --output-dir artifacts/ibm-hi-small \
  --report-dir docs/benchmarks/ibm-hi-small \
  --capacities 50 100 250 500 1000
```

PowerShell users can place the command on one line or replace each `\` with a backtick.

The command keeps large models and scored case files under the ignored `artifacts/` directory and
writes only the small, commit-ready report tables to `docs/benchmarks/ibm-hi-small/`.

The small benchmark contains roughly five million transactions. Use a machine with at least
8 GB of available memory for the current pandas pipeline. For quick plumbing checks, create a
smaller local CSV while retaining every positive row; do not use that biased sample to report
model quality.

The dataset is synthetic because real AML transactions are private and incompletely labeled.
IBM's generator provides complete ground truth and known transaction patterns, making it useful
for controlled benchmarking. See the [data card](docs/DATA_CARD.md) for provenance and caveats.

## Modeling approach

```mermaid
flowchart TD
    A["Transactions"] --> B["Account-day features"]
    B --> C["Early dates: fit"]
    B --> D["Later dates: score"]
    C --> E["Behavioral clusters"]
    E --> F["Cluster-relative anomalies"]
    D --> F
    F --> G["Ranked alert queue"]
    G --> H["Capacity trade-off"]
    H --> I["Reveal labels for evaluation"]
```

Features cover transaction velocity, incoming and outgoing value, counterparties, bank diversity,
active hours, payment format, cross-currency behavior, flow imbalance, and reciprocal relationships.
Skewed features receive `log1p` transforms and robust scaling.

Risk is a transparent operational score:

```text
risk = 0.82 × within-cluster anomaly percentile
     + 0.18 × distance-from-cluster-center percentile
```

See the [model card](docs/MODEL_CARD.md) for assumptions, intended use, and limitations.

## Repository layout

```text
signalgraph-aml/
├── app.py                         # Streamlit investigation console
├── src/signalgraph_aml/
│   ├── data.py                    # ingestion, validation, demo generator
│   ├── features.py                # account-day behavioral features
│   ├── modeling.py                # clustering, anomalies, explanations
│   ├── evaluation.py              # top-K operational metrics
│   ├── profiling.py               # data-driven segment descriptions
│   ├── benchmark.py               # IBM benchmark and report generator
│   └── pipeline.py                # reproducible CLI
├── tests/                         # unit and leakage tests
├── docs/                          # data and model cards
├── .github/workflows/ci.yml
├── Dockerfile
└── pyproject.toml
```

## Quality checks

```bash
make check
```

The tests verify IBM column normalization, deterministic demo generation, unique and complete
account-day features, strict temporal splitting, label exclusion, bounded risk scores, and top-K
metric calculations.

## Roadmap

- [x] Add a reproducible IBM HI-Small benchmark and report workflow.
- [ ] Run it on `HI-Small_Trans.csv` and commit the generated report.
- [ ] Add graph-motif features for fan-in, fan-out, rapid cycles, and scatter-gather behavior.
- [ ] Compare K-Means with HDBSCAN on a representative account sample.
- [ ] Add experiment tracking and feature-drift monitoring.
- [ ] Publish a hosted read-only dashboard with precomputed, non-sensitive artifacts.

## Responsible use

This project prioritizes investigation; it does not determine guilt, freeze accounts, or replace
an AML analyst. An anomaly is unusual, not necessarily illicit. Any real deployment would require
privacy controls, model validation, bias testing, audit trails, drift monitoring, and human review.
