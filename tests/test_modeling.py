from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.features import build_account_day_features, temporal_train_mask
from signalgraph_aml.modeling import SignalGraphModel


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
