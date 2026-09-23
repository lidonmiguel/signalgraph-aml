# Model card

## Model summary

SignalGraph AML is an unsupervised case-ranking system. It combines behavioral segmentation with
cluster-relative anomaly detection and returns a risk score from 0 to 100 for each account-day.

## Intended use

- Portfolio demonstration of unsupervised learning and financial-crime analytics.
- Controlled experiments on synthetic transaction data.
- Decision support for prioritizing a limited investigation queue.

It is not intended to make autonomous enforcement decisions or identify criminal conduct.

## Training design

MiniBatch K-Means and RobustScaler are fitted on the earliest complete dates closest to 70% of
account-day volume. Using case volume prevents sparse tail dates from consuming most of the split
while preserving a strict chronological boundary. One Isolation Forest is trained per sufficiently
large behavioral segment; small segments fall back to a global detector. Later dates are scored
out of time.

An optional comparison experiment uses a representative sample of early training accounts to
compare K-Means and HDBSCAN on identical later account-days. The experimental sampled arms use
the same anomaly-only score, with HDBSCAN noise referred to a global training distribution.
This experiment does not modify the deployed K-Means scoring pipeline. See
[`benchmarks/README.md`](benchmarks/README.md) for its design and result limitations.

Fan-in and fan-out count distinct senders and recipients in trailing 60-minute windows and record
the daily maximum. Rapid-cycle and scatter-gather flags identify ordered three- or four-edge
cycles (A→B→C→A or A→B→C→D→A) and four-edge diamonds (A→B,C→D) completed within three hours.
Motifs use distinct accounts. Cycle edges are strictly ordered, and both outgoing branches of a
scatter-gather must precede both incoming branches at the collector. The two collector transfers
must have different timestamps; the model does not infer order from simultaneous transfers.
Windows may reach into the preceding day but never beyond the case day's end. Self-transfers do
not count. Motif flags describe bounded transaction structures, not evidence of illicit conduct.

`is_laundering` is not included in the model feature list or used for preprocessing, cluster choice,
model fitting, score construction, or alert explanations.

## Evaluation

The primary metrics reflect a fixed analyst workload:

- Precision@K
- Recall@K
- Lift@K over random alert selection
- PR-AUC
- Positive-associated value among selected alerts

Accuracy is intentionally omitted because laundering cases are extremely rare.

## Explainability

Each alert reports the feature with the largest robust deviation from its behavioral segment, along
with the case value and segment median. The dashboard now shows three non-redundant behavioral
deviations and exposes numerators and denominators for share features. For example, it displays
“1 of 1 outgoing transactions” rather than presenting 100% without context. These deviations are
supporting explanations, not exact Isolation Forest feature attributions. The dashboard also
exposes the account's one-hop transfer network for the selected day.

Cluster names are generated from median behavior without using laundering outcomes. They are
descriptive summaries, not customer identities or risk categories.

## Known limitations

- Isolation Forest scores are relative, not calibrated probabilities of laundering.
- Cluster assignments can change as behavior or preprocessing changes.
- Graph motifs are local structural heuristics, not a full graph model. The one-hop visualization
  is investigative context.
- Model selection on a single synthetic generator may overfit its assumptions.
- Operational labels can be delayed, incomplete, and affected by prior monitoring systems.
- Cluster names are relative to the current population and may change after refitting.

## Required controls for real deployment

Human review, audit logging, access controls, privacy assessment, challenger models, threshold
governance, false-positive analysis, subgroup testing, feature and score drift monitoring, and a
documented process for analyst feedback and model retirement.
