"""Human-readable profiles for unsupervised behavioral segments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_cluster_profiles(scored_cases: pd.DataFrame) -> pd.DataFrame:
    """Summarize and name clusters without using laundering outcomes."""

    required = {
        "cluster",
        "total_tx_count",
        "total_value",
        "unique_out_counterparties",
        "unique_in_counterparties",
        "cross_currency_share",
        "cash_share",
        "flow_ratio",
    }
    missing = required.difference(scored_cases.columns)
    if missing:
        raise ValueError(f"Missing cluster profile columns: {sorted(missing)}")

    frame = scored_cases.copy()
    frame["counterparties"] = (
        frame["unique_out_counterparties"] + frame["unique_in_counterparties"]
    )
    profiles = (
        frame.groupby("cluster", observed=True)
        .agg(
            cases=("cluster", "size"),
            median_transactions=("total_tx_count", "median"),
            median_value=("total_value", "median"),
            median_counterparties=("counterparties", "median"),
            median_cross_currency_share=("cross_currency_share", "median"),
            median_cash_share=("cash_share", "median"),
            median_flow_ratio=("flow_ratio", "median"),
        )
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    names = _assign_cluster_names(profiles)
    profiles.insert(1, "cluster_name", profiles["cluster"].map(names))
    return profiles


def cluster_name_map(profiles: pd.DataFrame) -> dict[int, str]:
    """Return a cluster-to-display-name mapping."""

    return {
        int(row.cluster): str(row.cluster_name)
        for row in profiles[["cluster", "cluster_name"]].itertuples(index=False)
    }


def _assign_cluster_names(profiles: pd.DataFrame) -> dict[int, str]:
    metric_candidates = {
        "median_transactions": ("High-frequency activity", "Low-frequency activity"),
        "median_value": ("High-value movers", "Low-value activity"),
        "median_counterparties": ("Broad counterparty network", "Concentrated counterparties"),
        "median_cross_currency_share": ("Cross-currency activity", "Domestic-currency activity"),
        "median_cash_share": ("Cash-oriented activity", "Low-cash activity"),
        "median_flow_ratio": ("Net receivers", "Net senders"),
    }

    candidates: dict[int, list[tuple[float, str]]] = {}
    for row in profiles.itertuples(index=False):
        cluster_candidates: list[tuple[float, str]] = []
        for column, (high_label, low_label) in metric_candidates.items():
            values = profiles[column].astype(float)
            center = float(values.median())
            spread = float(values.quantile(0.75) - values.quantile(0.25))
            if spread == 0 or np.isnan(spread):
                spread = float(values.std()) or 1.0
            deviation = (float(getattr(row, column)) - center) / spread
            cluster_candidates.append(
                (abs(deviation), high_label if deviation >= 0 else low_label)
            )
        candidates[int(row.cluster)] = sorted(cluster_candidates, reverse=True)

    # Prefer distinct descriptions so all five filters remain easy to discuss.
    assigned: dict[int, str] = {}
    used: set[str] = set()
    order = sorted(candidates, key=lambda cluster: candidates[cluster][0][0], reverse=True)
    for cluster in order:
        label = next(
            (name for _, name in candidates[cluster] if name not in used),
            candidates[cluster][0][1],
        )
        assigned[cluster] = label
        used.add(label)
    return assigned
