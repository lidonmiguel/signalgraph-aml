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
- The current one-hop visualization is investigative context, not a graph model.
- Model selection on a single synthetic generator may overfit its assumptions.
- Operational labels can be delayed, incomplete, and affected by prior monitoring systems.
- Cluster names are relative to the current population and may change after refitting.

## Required controls for real deployment

Human review, audit logging, access controls, privacy assessment, challenger models, threshold
governance, false-positive analysis, subgroup testing, feature and score drift monitoring, and a
documented process for analyst feedback and model retirement.
