"""Fail closed if the Pages artifact gains files or diverges from public evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
ALLOWED = {
    "index.html",
    "styles.css",
    "app.js",
    "favicon.svg",
    "data/public-results.json",
}
FORBIDDEN_KEYS = {"account_id", "case_id", "from_account", "to_account", "is_laundering"}


def verify() -> None:
    entries = list(WEBSITE.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Website contains a symlink")
    actual = {path.relative_to(WEBSITE).as_posix() for path in entries if path.is_file()}
    if actual != ALLOWED:
        raise ValueError(f"Unexpected published files: {sorted(actual ^ ALLOWED)}")
    for relative in ALLOWED:
        path = WEBSITE / relative
        if path.is_symlink() or path.stat().st_size > 80_000:
            raise ValueError(f"Unsafe public asset: {relative}")

    public = json.loads((WEBSITE / "data/public-results.json").read_text(encoding="utf-8"))

    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            if FORBIDDEN_KEYS & value.keys():
                raise ValueError("Public data includes a case-level field")
            for item in value.values():
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)

    check_keys(public)
    if public["schema_version"] != 1:
        raise ValueError("Unsupported public data version")
    dataset = public["dataset"]
    baseline = json.loads(
        (ROOT / "docs/benchmarks/ibm-hi-small/benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    for key in ("input_sha256", "transactions", "training_cases", "evaluation_cases"):
        if dataset[key] != baseline[key]:
            raise ValueError(f"Dataset mismatch: {key}")
    if dataset["evaluation_positives"] != baseline["evaluation_positives"]:
        raise ValueError("Evaluation count mismatch")
    budgets = public["capacities"]
    if budgets != [row["capacity"] for row in baseline["capacity_results"]]:
        raise ValueError("Published capacities differ from the benchmark")
    if public["historical"]["hits"] != [
        row["hits"] for row in baseline["capacity_results"]
    ] or public["historical"]["pr_auc"] != baseline["pr_auc"]:
        raise ValueError("Historical results differ from the committed benchmark")

    report = (ROOT / "docs/benchmarks/cluster-comparison/RESULTS.md").read_text(
        encoding="utf-8"
    )
    if dataset["input_sha256"] not in report:
        raise ValueError("Comparison input hash differs from the benchmark")
    labels = [
        "Full K-Means, operational",
        "Sampled K-Means, common score",
        "HDBSCAN, size 50",
        "HDBSCAN, size 100",
        "HDBSCAN, size 200",
    ]
    for row, label in zip(public["comparison"], labels, strict=True):
        line = next((line for line in report.splitlines() if line.startswith(f"| {label} |")), None)
        if line is None:
            raise ValueError(f"Missing comparison evidence: {label}")
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if row["hits"] != [int(cell) for cell in cells[1:6]]:
            raise ValueError(f"Comparison hits differ: {label}")
        if row["pr_auc"] != float(cells[6]) or row["seconds"] != float(cells[9]):
            raise ValueError(f"Comparison metrics differ: {label}")
    if public["current"]["hits"] != public["comparison"][0]["hits"]:
        raise ValueError("Operational hits differ from the comparison")
    evidence = (ROOT / "docs/benchmarks/public-dashboard/EVIDENCE.md").read_text(
        encoding="utf-8"
    )
    review = public["drift_review"]
    if str(public["current"]["pr_auc"]) not in evidence:
        raise ValueError("Exact operational PR-AUC lacks provenance")
    if f"{review['first_day_share_pct']:.2f}%" not in evidence:
        raise ValueError("First-day drift context lacks provenance")
    for value in (review["psi"], review["training_any"], review["later_any"]):
        if f"{value:,}" not in evidence:
            raise ValueError(f"Missing drift provenance: {value}")
    if (
        review["training_any"] >= dataset["training_cases"]
        or review["later_any"] >= dataset["evaluation_cases"]
    ):
        raise ValueError("Invalid drift cohort counts")


if __name__ == "__main__":
    verify()
    print("Public dashboard guard passed: only five small assets and verified aggregates.")
