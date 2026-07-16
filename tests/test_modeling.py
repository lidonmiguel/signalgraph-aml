from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.features import build_account_day_features, temporal_train_mask
from signalgraph_aml.modeling import SignalGraphModel
from signalgraph_aml.profiling import build_cluster_profiles, cluster_name_map


def test_model_scores_without_label_leakage():
    transactions = generate_demo_transactions(100, 900, 6)
    features = build_account_day_features(transactions)
    train_mask = temporal_train_mask(features)
    model = SignalGraphModel(n_clusters=3).fit(features.loc[train_mask])
    scored = model.score(features.loc[~train_mask])

    assert "is_laundering" not in model.feature_columns
    assert scored["risk_score"].between(0, 100).all()
    assert scored["cluster"].between(0, 2).all()
    assert scored["alert_reason"].str.len().gt(10).all()
    assert scored["alert_factors"].str.count(" • ").eq(2).all()

    profiles = build_cluster_profiles(model.score(features))
    names = cluster_name_map(profiles)
    assert len(names) == 3
    assert len(set(names.values())) == 3


def test_share_explanation_discloses_small_denominator():
    transactions = generate_demo_transactions(100, 900, 6)
    features = build_account_day_features(transactions)
    train_mask = temporal_train_mask(features)
    model = SignalGraphModel(n_clusters=3).fit(features.loc[train_mask])
    row = features.loc[~train_mask].iloc[0].copy()
    transformed = model.scaler.transform(model._prepare(row.to_frame().T))
    cluster = int(model.clusterer.predict(transformed)[0])
    row["out_tx_count"] = 1
    row["cross_currency_share"] = 1.0
    explanation = model._format_factor(
        "cross_currency_share", row, model.cluster_medians[cluster]
    )
    assert "1 of 1 outgoing transaction (100%)" in explanation
    assert "limited sample" in explanation
