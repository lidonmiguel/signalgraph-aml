# Public dashboard evidence

The [hosted read-only dashboard](https://lidonmiguel.github.io/signalgraph-aml/)
serves only `website/` and its small `data/public-results.json` artifact. It does not
publish transactions, account identifiers, scored case tables, models, or queues.
The [deployment guard](../../../scripts/verify_public_dashboard.py) checks the exact
website file list and matches reported figures to committed benchmark evidence.

## Current operational reference

The corrected [full HI-Small comparison](../cluster-comparison/RESULTS.md) reports the
full-training K-Means arm on 720,800 later account-days (3,281 positives). Hits at
review capacities 50, 100, 250, 500, and 1,000 are 30, 35, 39, 45, and 61.
Its PR-AUC is printed as 0.01887. A subsequent local full benchmark on the same
SHA-256 input reported PR-AUC 0.018872797180843642 and 35 hits at K=100; these
two exact numbers were supplied from the local `benchmark_summary.json` output.
The locally generated report is not committed, so its remaining aggregate fields
are not presented as an independently archived full benchmark.

The earlier [committed benchmark](../ibm-hi-small/benchmark_summary.json) used a
pre-motif model. It had 20 hits at K=100 and PR-AUC 0.011841437837539888.
Both runs used `HI-Small_Trans.csv` with SHA-256
`b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.
The comparison does not isolate the effect of graph motifs or prove deployment
performance. Investigation capacity refers to the entire held-out set, not a day.

## Feature drift diagnostic

A local check on saved account-day features found one feature above the PSI review
threshold of 0.2. For `reciprocal_counterparties`, PSI was 0.547284:

| Period | Account-days | With at least one reciprocal counterparty | Missing |
|---|---:|---:|---:|
| Training, 1–7 September | 1,759,477 | 391,630 | 0 |
| Evaluation, 8–18 September | 720,800 | 13,754 | 0 |

The first training date, 1 September, had 437,065 account-days, of which
83.62% had reciprocal activity. Rates on the other high-volume days were
approximately 1.5–2.2%, on both sides of the split. The dates after
10 September have very few cases. This identifies a baseline sensitivity to
the first day; the source of that day's activity has not been established.
No threshold, data, or scoring method was changed to suppress the alert.
These aggregate diagnostic figures were supplied from a local drift check on
the same input and can be regenerated from the owner's ignored case files.

## Scope

All outcomes are from synthetic IBM data, and labels are used only for
evaluation. No later cases influence model fitting. The interactive capacity
control selects a row from committed aggregate results; it cannot score a
new case, query an account, or change the model. The hosted page makes no
claims about actual suspicious activity or real-world efficacy.
