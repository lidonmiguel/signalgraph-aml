"""Account-day feature engineering for transaction monitoring."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from signalgraph_aml.config import FEATURE_COLUMNS


def _max_out_recipients_60m(frame: pd.DataFrame) -> pd.DataFrame:
    """Maximum distinct recipients in a trailing 60-minute window per sender-day.

    Windows may include the preceding day's payments, but each maximum is
    attributed only to the day of the window's final transaction.
    """

    payments = frame[["from_account", "to_account", "timestamp", "date"]].sort_values(
        ["from_account", "timestamp"], kind="mergesort"
    )
    window = pd.Timedelta(minutes=60)
    records: list[tuple[pd.Timestamp, str, int]] = []

    for account, account_payments in payments.groupby("from_account", sort=False, observed=True):
        recent: deque[tuple[pd.Timestamp, str]] = deque()
        recipient_counts: dict[str, int] = {}
        current_day = None
        maximum = 0
        for timestamp, recipient, day in account_payments[
            ["timestamp", "to_account", "date"]
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

            # A transfer back to the sender is not dispersal to a recipient.
            if recipient != account:
                recent.append((timestamp, recipient))
                recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1
            maximum = max(maximum, len(recipient_counts))

        if current_day is not None:
            records.append((current_day, account, maximum))

    return pd.DataFrame(
        records, columns=["date", "account_id", "max_out_recipients_60m"]
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
        _max_out_recipients_60m(frame), how="left", on=["date", "account_id"]
    )
    features["max_out_recipients_60m"] = (
        features["max_out_recipients_60m"].fillna(0).astype("int32")
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
