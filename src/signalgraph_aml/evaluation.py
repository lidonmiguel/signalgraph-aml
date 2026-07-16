"""Operational evaluation for rare-event alert ranking."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score


def rank_alerts(scored_cases: pd.DataFrame, alert_budget: int) -> pd.DataFrame:
    """Return the highest-risk cases under a fixed investigation budget."""

    if alert_budget < 1:
        raise ValueError("alert_budget must be positive")
    return scored_cases.sort_values("risk_score", ascending=False).head(alert_budget).copy()


def evaluate_alerts(scored_cases: pd.DataFrame, alert_budget: int = 100) -> dict[str, float | int]:
    """Evaluate rankings only after unsupervised model scoring is complete."""

    if scored_cases.empty:
        raise ValueError("Cannot evaluate an empty case table")
    if "is_laundering" not in scored_cases:
        raise ValueError("Ground-truth labels are required for evaluation")

    budget = min(alert_budget, len(scored_cases))
    alerts = rank_alerts(scored_cases, budget)
    positives = int(scored_cases["is_laundering"].sum())
    hits = int(alerts["is_laundering"].sum())
    prevalence = positives / len(scored_cases)
    precision = hits / budget
    recall = hits / positives if positives else 0.0
    lift = precision / prevalence if prevalence else 0.0
    pr_auc = (
        float(average_precision_score(scored_cases["is_laundering"], scored_cases["risk_score"]))
        if positives
        else 0.0
    )
    detected_value = float(
        alerts.loc[alerts["is_laundering"].eq(1), "total_value"].sum()
    )
    return {
        "cases": len(scored_cases),
        "positives": positives,
        "alert_budget": budget,
        "hits": hits,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "pr_auc": pr_auc,
        "detected_value": detected_value,
    }


def capacity_curve(
    scored_cases: pd.DataFrame,
    capacities: list[int] | range | None = None,
) -> pd.DataFrame:
    """Calculate operational performance across investigation capacities.

    This changes only the cut-off in the existing risk ranking; it never refits the model.
    """

    if scored_cases.empty:
        raise ValueError("Cannot evaluate an empty case table")
    if "is_laundering" not in scored_cases:
        raise ValueError("Ground-truth labels are required for evaluation")

    maximum = len(scored_cases)
    if capacities is None:
        capacities = range(10, min(250, maximum) + 1, 10)
    valid_capacities = sorted(
        {min(int(capacity), maximum) for capacity in capacities if capacity > 0}
    )
    if not valid_capacities:
        raise ValueError("At least one positive capacity is required")

    ranked = scored_cases.sort_values("risk_score", ascending=False).reset_index(drop=True)
    cumulative_hits = ranked["is_laundering"].cumsum()
    cumulative_detected_value = (
        ranked["total_value"] * ranked["is_laundering"]
    ).cumsum()
    total_positives = int(ranked["is_laundering"].sum())
    prevalence = total_positives / maximum

    rows: list[dict[str, float | int]] = []
    for capacity in valid_capacities:
        hits = int(cumulative_hits.iloc[capacity - 1])
        precision = hits / capacity
        recall = hits / total_positives if total_positives else 0.0
        rows.append(
            {
                "capacity": capacity,
                "hits": hits,
                "precision": precision,
                "recall": recall,
                "lift": precision / prevalence if prevalence else 0.0,
                "detected_value": float(cumulative_detected_value.iloc[capacity - 1]),
            }
        )
    return pd.DataFrame(rows)
