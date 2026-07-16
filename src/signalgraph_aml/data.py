"""Transaction ingestion, validation, and deterministic demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signalgraph_aml.config import RANDOM_STATE

REQUIRED_COLUMNS = {
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
}

COLUMN_ALIASES = {
    "account": "from_account",
    "account_1": "to_account",
}


def _snake_case(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the IBM AML schema and validate required fields.

    The original label remains available for evaluation, but downstream model code
    explicitly excludes it from the feature set.
    """

    renamed = {
        column: COLUMN_ALIASES.get(_snake_case(column), _snake_case(column))
        for column in frame
    }
    result = frame.rename(columns=renamed).copy()
    missing = REQUIRED_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    result = result[list(REQUIRED_COLUMNS)].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    for column in ("from_bank", "from_account", "to_bank", "to_account"):
        result[column] = result[column].astype(str)
    for column in ("amount_received", "amount_paid"):
        result[column] = pd.to_numeric(result[column], errors="raise").clip(lower=0)
    result["is_laundering"] = (
        pd.to_numeric(result["is_laundering"], errors="raise").fillna(0).astype("int8")
    )
    return result.sort_values("timestamp").reset_index(drop=True)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load an IBM AML CSV from disk."""

    return normalize_transactions(pd.read_csv(Path(path)))


def generate_demo_transactions(
    n_accounts: int = 320,
    n_transactions: int = 6_000,
    n_days: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate realistic-enough local data with several hidden AML patterns.

    The generator makes the repository runnable without redistributing IBM data. It
    is a product demo and test fixture, not a substitute for the research dataset.
    """

    if n_accounts < 40 or n_transactions < 200 or n_days < 3:
        raise ValueError("Demo generation needs >=40 accounts, >=200 transactions, and >=3 days")

    rng = np.random.default_rng(random_state)
    accounts = np.array([f"ACC-{index:05d}" for index in range(n_accounts)])
    banks = np.array([f"BANK-{index:02d}" for index in range(1, 13)])
    account_bank = dict(zip(accounts, rng.choice(banks, size=n_accounts), strict=True))
    profiles = rng.choice(
        ["retail", "business", "remittance"],
        size=n_accounts,
        p=[0.70, 0.20, 0.10],
    )
    profile_by_account = dict(zip(accounts, profiles, strict=True))

    sources = rng.choice(accounts, size=n_transactions)
    destinations = rng.choice(accounts, size=n_transactions)
    same = sources == destinations
    while same.any():
        destinations[same] = rng.choice(accounts, size=int(same.sum()))
        same = sources == destinations

    amount_parameters = {
        "retail": (3.4, 0.75),
        "business": (6.2, 0.85),
        "remittance": (4.8, 0.95),
    }
    amounts = np.array(
        [rng.lognormal(*amount_parameters[profile_by_account[source]]) for source in sources]
    ).round(2)

    base = pd.Timestamp("2025-01-06")
    seconds = rng.integers(0, n_days * 86_400, size=n_transactions)
    timestamps = base + pd.to_timedelta(seconds, unit="s")
    payment_formats = rng.choice(
        ["ACH", "Credit Card", "Wire", "Cheque", "Cash"],
        size=n_transactions,
        p=[0.37, 0.32, 0.17, 0.08, 0.06],
    )
    currencies = np.array(["US Dollar", "Euro", "Yen", "UK Pound"])
    payment_currency = rng.choice(currencies, size=n_transactions, p=[0.55, 0.28, 0.09, 0.08])
    receiving_currency = payment_currency.copy()
    cross_currency = rng.random(n_transactions) < 0.06
    receiving_currency[cross_currency] = rng.choice(currencies, size=int(cross_currency.sum()))

    records = pd.DataFrame(
        {
            "timestamp": timestamps,
            "from_bank": [account_bank[item] for item in sources],
            "from_account": sources,
            "to_bank": [account_bank[item] for item in destinations],
            "to_account": destinations,
            "amount_received": amounts,
            "receiving_currency": receiving_currency,
            "amount_paid": amounts,
            "payment_currency": payment_currency,
            "payment_format": payment_formats,
            "is_laundering": np.zeros(n_transactions, dtype="int8"),
        }
    )

    illicit_rows: list[dict[str, object]] = []
    suspicious_accounts = rng.choice(accounts, size=36, replace=False)

    # Five rapid cycles: money returns to its origin after passing through intermediaries.
    for pattern in range(5):
        nodes = suspicious_accounts[pattern * 4 : pattern * 4 + 4]
        day = n_days - 3 + pattern % 3
        amount = float(rng.uniform(18_000, 70_000))
        for step, (source, target) in enumerate(zip(nodes, np.roll(nodes, -1), strict=True)):
            illicit_rows.append(
                _transaction_record(
                    base + pd.Timedelta(days=day, hours=2, minutes=step * 7),
                    source,
                    target,
                    account_bank,
                    amount * (1 - step * 0.008),
                    "Wire",
                    "US Dollar",
                    "Euro" if step % 2 else "US Dollar",
                )
            )

    # Fan-out patterns: one account quickly disperses incoming funds.
    for pattern in range(4):
        hub = suspicious_accounts[20 + pattern]
        targets = suspicious_accounts[24 + pattern * 3 : 27 + pattern * 3]
        day = n_days - 2 + pattern % 2
        for step, target in enumerate(targets):
            illicit_rows.append(
                _transaction_record(
                    base + pd.Timedelta(days=day, hours=4, minutes=step * 3),
                    hub,
                    target,
                    account_bank,
                    float(rng.uniform(12_000, 35_000)),
                    "Wire",
                    "Euro",
                    "US Dollar",
                )
            )

    records = pd.concat([records, pd.DataFrame(illicit_rows)], ignore_index=True)
    return normalize_transactions(records)


def _transaction_record(
    timestamp: pd.Timestamp,
    source: str,
    target: str,
    account_bank: dict[str, str],
    amount: float,
    payment_format: str,
    payment_currency: str,
    receiving_currency: str,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "from_bank": account_bank[source],
        "from_account": source,
        "to_bank": account_bank[target],
        "to_account": target,
        "amount_received": round(amount, 2),
        "receiving_currency": receiving_currency,
        "amount_paid": round(amount, 2),
        "payment_currency": payment_currency,
        "payment_format": payment_format,
        "is_laundering": 1,
    }
