# Data card

## Primary benchmark

**Name:** IBM Transactions for Anti Money Laundering (AML)

**Source:** [IBM AML-Data](https://github.com/IBM/AML-Data) and the
[Kaggle distribution](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

**Research description:** [Realistic Synthetic Financial Transactions for Anti-Money
Laundering Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf)

**Data license:** CDLA-Sharing-1.0. Check the source terms before redistributing derived data.

The benchmark represents synthetic banks, accounts, payment types, currencies, timestamps, values,
and laundering indicators. It is generated rather than anonymized from individual customers.

## Dataset selection

Start with `HI-Small_Trans.csv`: approximately five million transactions across ten days. It is
large enough to demonstrate practical processing while remaining feasible on a personal computer.
The repository never commits or redistributes this file.

## Demo data

`generate_demo_transactions()` creates a small, deterministic dataset containing ordinary retail,
business, and remittance-like behavior plus injected cycles and fan-out patterns. Its purpose is to
test the product workflow and make the dashboard immediately runnable.

Demo results must not be presented as independent evidence of AML detection performance: the
features and injected patterns were created in the same repository.

## Processing

The raw transaction schema is normalized, validated, and aggregated into account-day observations.
Both sender and receiver activity contribute to each case. The binary laundering field is retained
only as an evaluation outcome and is excluded from all model features.

## Limitations

- Synthetic behavior cannot capture every operational, geographic, or regulatory nuance.
- Complete synthetic labels are cleaner than real suspicious-activity outcomes.
- Currency values are not converted to a common exchange-rate basis in the MVP.
- Ten days is too short for production seasonality or long-term drift studies.
- A laundering label indicates generated ground truth, not legal adjudication.
