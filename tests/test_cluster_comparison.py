import pandas as pd

from signalgraph_aml.cluster_comparison import (
    compare_account_days,
    representative_account_sample,
)
from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.features import build_account_day_features, temporal_train_mask


def _features():
    return build_account_day_features(generate_demo_transactions(80, 900, 6))


def test_account_sample_is_deterministic_and_keeps_training_days_together():
    features = _features()
    training = features.loc[temporal_train_mask(features)]
    sampled = representative_account_sample(training, target_cases=100)
    repeated = representative_account_sample(training, target_cases=100)

    pd.testing.assert_frame_equal(sampled, repeated)
    counts = training.groupby("account_id").size()
    assert (
        sampled.groupby("account_id")
        .size()
        .equals(counts.loc[sampled["account_id"].unique()].sort_index())
    )
    assert sampled["date"].max() <= training["date"].max()
    assert len(sampled) < len(training)


def test_comparison_never_uses_training_labels_to_fit_or_select():
    features = _features()
    options = dict(
        sample_cases=120,
        min_cluster_sizes=(15,),
        min_samples=5,
        n_clusters=3,
        capacities=[10, 20],
        chunk_size=40,
    )
    original = compare_account_days(features, **options)
    changed = features.copy()
    train_mask = temporal_train_mask(changed)
    changed.loc[train_mask, "is_laundering"] = 1 - changed.loc[train_mask, "is_laundering"]
    rerun = compare_account_days(changed, **options)

    assert original["training_date_max"] < original["evaluation_date_min"]
    assert original["sample_case_ids_sha256"] == rerun["sample_case_ids_sha256"]
    assert [row["method"] for row in original["results"]] == [
        "kmeans_full_operational",
        "kmeans_sampled_anomaly",
        "hdbscan_mcs_15_ms_5",
    ]
    for before, after in zip(original["results"], rerun["results"], strict=True):
        assert before["training_cluster_sizes"] == after["training_cluster_sizes"]
        assert before["evaluation_noise_fraction"] == after["evaluation_noise_fraction"]
        assert before["capacity_results"] == after["capacity_results"]
        assert before["pr_auc"] == after["pr_auc"]


def test_hdbscan_without_clusters_scores_every_case_as_noise():
    features = _features()
    training = features.loc[temporal_train_mask(features)]
    sample_size = len(representative_account_sample(training, target_cases=120))
    result = compare_account_days(
        features,
        sample_cases=120,
        min_cluster_sizes=(sample_size // 2 + 1,),
        min_samples=5,
        n_clusters=3,
        capacities=[10],
        chunk_size=40,
    )["results"][-1]

    assert result["training_clusters"] == 0
    assert result["training_noise_fraction"] == 1.0
    assert result["evaluation_noise_fraction"] == 1.0
    assert 0 <= result["pr_auc"] <= 1
