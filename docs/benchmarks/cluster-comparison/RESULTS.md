# K-Means versus HDBSCAN: full HI-Small comparison

The corrected full-data comparison was run locally with
`signalgraph-compare-clusters` from commit `d114014` (percentile tie fix).
This is a summary of the generated `COMPARISON_REPORT.md` supplied for review;
the IBM transactions and account-day records are not committed.

## Reproducibility

| Item | Recorded value |
|---|---|
| Source | `HI-Small_Trans.csv`, 5,078,345 transactions |
| Input SHA-256 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| Training account-days | 1,759,477, through 2022-09-07 |
| Sample | 20,067 account-days from 5,835 accounts; seed 42 |
| Sample case-ID SHA-256 | `3d3511e4fda6a4ca55487290aa860bd721a104cc65166d6cdadd470826f3a684` |
| Sample rule | Accounts sampled proportionally across quintiles of training transaction activity, retaining all their training account-days |
| Held-out population | 720,800 account-days from 2022-09-08, including 3,281 positives (0.455% prevalence) |
| HDBSCAN settings | `min_cluster_size` 50, 100, 200; `min_samples` 15 |
| End-to-end run | 895.71 seconds, including 482.90 seconds of feature engineering |

The full K-Means row uses all early training cases and the production score.
The sampled K-Means and HDBSCAN rows use the *same* training account sample,
preprocessing, Isolation Forest rules, global tie breaker, and held-out cases.
Laundering labels are used only after scoring. Capacities refer to the entire
held-out queue, not to a daily budget.

## Out-of-time results

| Method | Hits@50 | Hits@100 | Hits@250 | Hits@500 | Hits@1,000 | PR-AUC | Clusters | Held-out noise | Method seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full K-Means, operational | 30 | 35 | 39 | 45 | 61 | 0.01887 | 5 | 0% | 45.0 |
| Sampled K-Means, common score | 14 | 17 | 37 | 38 | 51 | 0.01839 | 5 | 0% | 10.5 |
| HDBSCAN, size 50 | 21 | 21 | 21 | 21 | 25 | 0.01095 | 47 | 21.2% | 120.5 |
| HDBSCAN, size 100 | 21 | 21 | 21 | 21 | 41 | 0.01006 | 32 | 18.6% | 115.1 |
| HDBSCAN, size 200 | 21 | 21 | 21 | 21 | 41 | 0.00989 | 23 | 16.2% | 110.2 |

At capacity 100, the full operational model has 35% precision and 1.07%
recall. Sampled K-Means has 17% precision; each HDBSCAN setting has 21%.
Within the like-for-like sampled comparison, HDBSCAN selects more positives
at 50 and 100, while K-Means selects more at 250, 500, and 1,000 and has the
higher PR-AUC. The full operational model performs best at every listed
capacity, but its training size and score also differ from the sampled arms.
The method runtimes exclude feature engineering and shared sample setup.

## Primary-score ties

Empirical within-cluster percentiles can saturate at the training maximum.
The corrected experiment orders equal percentiles with the same global
Isolation Forest anomaly score in each sampled arm. It recorded these ties
*before* the secondary score was applied:

| Method | Cases at primary maximum | Tied at K=50 | Tied at K=100 |
|---|---:|---:|---:|
| Full K-Means, operational score | 5 | 12 | 21 |
| Sampled K-Means | 95 | 95 | 36 |
| HDBSCAN, size 50 | 1,824 | 1,824 | 1,824 |
| HDBSCAN, size 100 | 775 | 775 | 775 |
| HDBSCAN, size 200 | 792 | 792 | 792 |

All HDBSCAN top-100 cases therefore come from a large group at the maximum
primary percentile. The common global anomaly score determines their order.
The extra HDBSCAN hits at small capacities cannot be attributed to its
cluster structure alone. The earlier run without a tie breaker is superseded
by this corrected comparison.

## Decision

Keep the full K-Means model in the app. On this synthetic dataset and this
preselected training sample, HDBSCAN did not improve overall ranking quality
or run time enough to justify a model switch. The result completes the
roadmap comparison; it is not a claim that K-Means is universally better than
HDBSCAN. Other datasets, samples, and clustering settings may behave
differently.
