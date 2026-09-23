"""Shared project configuration."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "out_tx_count",
    "in_tx_count",
    "out_total",
    "in_total",
    "out_mean",
    "in_mean",
    "out_max",
    "in_max",
    "unique_out_counterparties",
    "max_out_recipients_60m",
    "unique_in_counterparties",
    "max_in_senders_60m",
    "unique_out_banks",
    "unique_in_banks",
    "active_out_hours",
    "cash_share",
    "cross_currency_share",
    "reciprocal_counterparties",
    "rapid_cycle_3h",
    "scatter_gather_3h",
    "total_tx_count",
    "total_value",
    "flow_ratio",
]

FEATURE_LABELS = {
    "out_tx_count": "outgoing transaction velocity",
    "in_tx_count": "incoming transaction velocity",
    "out_total": "outgoing value",
    "in_total": "incoming value",
    "out_mean": "average outgoing amount",
    "in_mean": "average incoming amount",
    "out_max": "largest outgoing payment",
    "in_max": "largest incoming payment",
    "unique_out_counterparties": "outgoing counterparties",
    "max_out_recipients_60m": "distinct recipients in the busiest 60 minutes",
    "unique_in_counterparties": "incoming counterparties",
    "max_in_senders_60m": "distinct senders in the busiest 60 minutes",
    "unique_out_banks": "destination-bank diversity",
    "unique_in_banks": "source-bank diversity",
    "active_out_hours": "active-hour spread",
    "cash_share": "cash-payment share",
    "cross_currency_share": "cross-currency activity",
    "reciprocal_counterparties": "reciprocal counterparties",
    "rapid_cycle_3h": "three- or four-account cycle completed within three hours",
    "scatter_gather_3h": "scatter-gather path completed within three hours",
    "total_tx_count": "total transaction velocity",
    "total_value": "total value moved",
    "flow_ratio": "incoming/outgoing flow imbalance",
}

RANDOM_STATE = 42
