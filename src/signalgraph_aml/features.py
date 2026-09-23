"""Account-day feature engineering for transaction monitoring."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from signalgraph_aml.config import FEATURE_COLUMNS


def _max_distinct_counterparties_60m(
    frame: pd.DataFrame,
    account_column: str,
    counterparty_column: str,
    result_column: str,
) -> pd.DataFrame:
    """Maximum distinct counterparties in a trailing hour per account-day.

    Windows may include the preceding day's payments, but each maximum is
    attributed only to the day of the window's final transaction.
    """

    payments = frame[[account_column, counterparty_column, "timestamp", "date"]].sort_values(
        [account_column, "timestamp"], kind="mergesort"
    )
    window = pd.Timedelta(minutes=60)
    records: list[tuple[pd.Timestamp, str, int]] = []

    for account, account_payments in payments.groupby(account_column, sort=False, observed=True):
        recent: deque[tuple[pd.Timestamp, str]] = deque()
        recipient_counts: dict[str, int] = {}
        current_day = None
        maximum = 0
        for timestamp, recipient, day in account_payments[
            ["timestamp", counterparty_column, "date"]
        ].itertuples(index=False, name=None):
            if current_day is not None and day != current_day:
                records.append((current_day, account, maximum))
                maximum = 0
            current_day = day

            cutoff = timestamp - window
            while recent and recent[0][0] < cutoff:
                _, expired = recent.popleft()
                count = recipient_counts[expired] - 1
                if count:
                    recipient_counts[expired] = count
                else:
                    del recipient_counts[expired]

            # A self-transfer is neither a new sender nor a new recipient.
            if recipient != account:
                recent.append((timestamp, recipient))
                recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1
            maximum = max(maximum, len(recipient_counts))

        if current_day is not None:
            records.append((current_day, account, maximum))

    return pd.DataFrame(
        records, columns=["date", "account_id", result_column]
    )


def _graph_motif_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag ordered 3/4-edge cycles and four-edge scatter-gather diamonds.

    Only earlier transactions within three hours are available when an edge is
    evaluated. Equal-timestamp edges are staged together, so an order is never
    inferred from their CSV row order. A completed motif belongs to its closing
    transaction's date, including when earlier edges occurred the previous day.
    """

    ordered = frame[["timestamp", "date", "from_account", "to_account"]].sort_values(
        "timestamp", kind="mergesort"
    )
    window = pd.Timedelta(hours=3)
    recent: deque[tuple[pd.Timestamp, str, str]] = deque()
    outgoing: dict[str, dict[str, deque[pd.Timestamp]]] = {}
    incoming: dict[str, dict[str, deque[pd.Timestamp]]] = {}
    cycles: set[tuple[pd.Timestamp, str]] = set()
    diamonds: set[tuple[pd.Timestamp, str]] = set()
    pending: list[tuple[str, str]] = []
    pending_time: pd.Timestamp | None = None

    for timestamp, day, source, target in ordered.itertuples(index=False, name=None):
        if pending_time is not None and timestamp != pending_time:
            for previous_source, previous_target in pending:
                pair = outgoing.setdefault(previous_source, {}).setdefault(
                    previous_target, deque()
                )
                pair.append(pending_time)
                incoming.setdefault(previous_target, {})[previous_source] = pair
                recent.append((pending_time, previous_source, previous_target))
            pending.clear()

        if pending_time != timestamp:
            pending_time = timestamp
            cutoff = timestamp - window
            while recent and recent[0][0] < cutoff:
                _, old_source, old_target = recent.popleft()
                pair = outgoing[old_source][old_target]
                pair.popleft()
                if not pair:
                    del outgoing[old_source][old_target]
                    del incoming[old_target][old_source]
                    if not outgoing[old_source]:
                        del outgoing[old_source]
                    if not incoming[old_target]:
                        del incoming[old_target]

        if source == target:
            continue

        # A -> B, B -> C, then C -> A (the current transfer).
        first_neighbors = outgoing.get(target, {})
        second_neighbors = incoming.get(source, {})
        candidates = (
            first_neighbors if len(first_neighbors) <= len(second_neighbors)
            else second_neighbors
        )
        for middle in candidates:
            if middle in (source, target) or middle not in first_neighbors:
                continue
            if middle not in second_neighbors:
                continue
            if first_neighbors[middle][0] < second_neighbors[middle][-1]:
                cycles.update(((day, target), (day, middle), (day, source)))

        # A -> B -> C -> D, then D -> A (the demo uses four accounts).
        for first_middle, first_times in first_neighbors.items():
            if first_middle in (source, target):
                continue
            second_hops = outgoing.get(first_middle, {})
            last_hops = incoming.get(source, {})
            for second_middle in second_hops:
                if second_middle in (target, first_middle, source):
                    continue
                if second_middle not in last_hops:
                    continue
                last_time = last_hops[second_middle][-1]
                if any(first_times[0] < time < last_time for time in second_hops[second_middle]):
                    cycles.update(
                        ((day, target), (day, first_middle), (day, second_middle), (day, source))
                    )

        # A -> B and A -> C, followed by B -> D and C -> D.
        # The current source is C and the current target is D.
        for origin in incoming.get(source, {}):
            if origin in (source, target):
                continue
            other_outgoing = outgoing.get(origin, {})
            other_incoming = incoming.get(target, {})
            other_candidates = (
                other_outgoing if len(other_outgoing) <= len(other_incoming)
                else other_incoming
            )
            for other_middle in other_candidates:
                if other_middle in (origin, source, target):
                    continue
                if other_middle not in other_outgoing or other_middle not in other_incoming:
                    continue
                # Both scatter edges must precede either gather edge.
                last_scatter = max(
                    incoming[source][origin][0], other_outgoing[other_middle][0]
                )
                if last_scatter < other_incoming[other_middle][-1]:
                    diamonds.update(
                        ((day, origin), (day, source), (day, other_middle), (day, target))
                    )

        pending.append((source, target))

    cases = cycles | diamonds
    return pd.DataFrame(
        [
            (day, account, int((day, account) in cycles), int((day, account) in diamonds))
            for day, account in cases
        ],
        columns=["date", "account_id", "rapid_cycle_3h", "scatter_gather_3h"],
    )


def build_account_day_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one behavioral observation per account and day."""

    frame = transactions.copy()
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["is_cash"] = frame["payment_format"].str.casefold().eq("cash").astype("int8")
    frame["is_cross_currency"] = (
        frame["payment_currency"].ne(frame["receiving_currency"]).astype("int8")
    )

    outgoing = (
        frame.groupby(["date", "from_account"], observed=True)
        .agg(
            out_tx_count=("from_account", "size"),
            out_total=("amount_paid", "sum"),
            out_mean=("amount_paid", "mean"),
            out_max=("amount_paid", "max"),
            unique_out_counterparties=("to_account", "nunique"),
            unique_out_banks=("to_bank", "nunique"),
            active_out_hours=("hour", "nunique"),
            cash_share=("is_cash", "mean"),
            cross_currency_share=("is_cross_currency", "mean"),
            out_laundering=("is_laundering", "max"),
        )
        .reset_index()
        .rename(columns={"from_account": "account_id"})
    )

    incoming = (
        frame.groupby(["date", "to_account"], observed=True)
        .agg(
            in_tx_count=("to_account", "size"),
            in_total=("amount_received", "sum"),
            in_mean=("amount_received", "mean"),
            in_max=("amount_received", "max"),
            unique_in_counterparties=("from_account", "nunique"),
            unique_in_banks=("from_bank", "nunique"),
            in_laundering=("is_laundering", "max"),
        )
        .reset_index()
        .rename(columns={"to_account": "account_id"})
    )

    pairs = frame[["date", "from_account", "to_account"]].drop_duplicates()
    reverse_pairs = pairs.rename(
        columns={"from_account": "to_account", "to_account": "from_account"}
    )
    reciprocal = (
        pairs.merge(reverse_pairs, on=["date", "from_account", "to_account"])
        .groupby(["date", "from_account"], observed=True)["to_account"]
        .nunique()
        .rename("reciprocal_counterparties")
        .reset_index()
        .rename(columns={"from_account": "account_id"})
    )

    features = outgoing.merge(incoming, how="outer", on=["date", "account_id"])
    features = features.merge(reciprocal, how="left", on=["date", "account_id"])
    features = features.merge(
        _max_distinct_counterparties_60m(
            frame, "from_account", "to_account", "max_out_recipients_60m"
        ),
        how="left",
        on=["date", "account_id"],
    )
    features = features.merge(
        _max_distinct_counterparties_60m(
            frame, "to_account", "from_account", "max_in_senders_60m"
        ),
        how="left",
        on=["date", "account_id"],
    )
    features = features.merge(_graph_motif_flags(frame), how="left", on=["date", "account_id"])
    for column in (
        "max_out_recipients_60m",
        "max_in_senders_60m",
        "rapid_cycle_3h",
        "scatter_gather_3h",
    ):
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0).astype(
            "int32"
        )
    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    features["is_laundering"] = (
        features.pop("out_laundering").fillna(0).astype("int8")
        | features.pop("in_laundering").fillna(0).astype("int8")
    ).astype("int8")
    features["total_tx_count"] = features["out_tx_count"] + features["in_tx_count"]
    features["total_value"] = features["out_total"] + features["in_total"]
    features["flow_ratio"] = np.log1p(features["in_total"]) - np.log1p(features["out_total"])
    features["case_id"] = (
        features["date"].dt.strftime("%Y%m%d") + "-" + features["account_id"]
    )

    ordered = ["case_id", "date", "account_id", *FEATURE_COLUMNS, "is_laundering"]
    return features[ordered].sort_values(["date", "account_id"]).reset_index(drop=True)


def temporal_train_mask(features: pd.DataFrame, train_fraction: float = 0.7) -> pd.Series:
    """Return the earliest complete dates closest to the requested case fraction."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    date_counts = features.groupby("date", observed=True, sort=True).size()
    if len(date_counts) < 2:
        raise ValueError("At least two dates are required for a temporal split")
    cumulative_cases = date_counts.cumsum().iloc[:-1]
    target_cases = len(features) * train_fraction
    cutoff_date = (cumulative_cases - target_cases).abs().idxmin()
    training_dates = set(date_counts.loc[:cutoff_date].index)
    return features["date"].isin(training_dates)
