"""Explainable clustering and anomaly ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from signalgraph_aml.config import FEATURE_COLUMNS, FEATURE_LABELS, RANDOM_STATE

LOG_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in {"cash_share", "cross_currency_share", "flow_ratio"}
]

MONEY_FEATURES = {
    "out_total",
    "in_total",
    "out_mean",
    "in_mean",
    "out_max",
    "in_max",
    "total_value",
}
COUNT_FEATURES = {
    "out_tx_count",
    "in_tx_count",
    "unique_out_counterparties",
    "unique_in_counterparties",
    "unique_out_banks",
    "unique_in_banks",
    "active_out_hours",
    "reciprocal_counterparties",
    "total_tx_count",
}
EXPLANATION_GROUPS = {
    "out_tx_count": "transaction_count",
    "in_tx_count": "transaction_count",
    "total_tx_count": "transaction_count",
    "out_total": "transaction_value",
    "in_total": "transaction_value",
    "total_value": "transaction_value",
    "out_mean": "typical_amount",
    "in_mean": "typical_amount",
    "out_max": "largest_amount",
    "in_max": "largest_amount",
    "unique_out_counterparties": "counterparties",
    "unique_in_counterparties": "counterparties",
    "reciprocal_counterparties": "counterparties",
    "unique_out_banks": "banks",
    "unique_in_banks": "banks",
}


class SignalGraphModel:
    """Segment behavior, then rank cluster-relative anomalies.

    Labels are deliberately absent from this class. Only engineered behavior is accepted
    by ``fit`` and ``score``; ground truth is joined later by the evaluation module.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1_024,
            n_init=10,
            random_state=random_state,
        )
        self.global_detector = IsolationForest(
            n_estimators=250,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        self.pca = PCA(n_components=2, random_state=random_state)
        self.detectors: dict[int, IsolationForest] = {}
        self.anomaly_reference: dict[int, np.ndarray] = {}
        self.distance_reference: dict[int, np.ndarray] = {}
        self.cluster_medians: dict[int, pd.Series] = {}
        self.cluster_scales: dict[int, pd.Series] = {}
        self.is_fitted = False

    def fit(self, features: pd.DataFrame) -> SignalGraphModel:
        """Fit only on unlabeled engineered features from the training period."""

        self._validate(features)
        if len(features) < self.n_clusters * 10:
            raise ValueError("Not enough observations for the requested number of clusters")

        transformed = self._prepare(features)
        scaled = self.scaler.fit_transform(transformed)
        clusters = self.clusterer.fit_predict(scaled)
        self.global_detector.fit(scaled)
        self.pca.fit(scaled)

        minimum_cluster_size = max(30, len(self.feature_columns) * 3)
        for cluster in range(self.n_clusters):
            mask = clusters == cluster
            cluster_scaled = scaled[mask]
            if len(cluster_scaled) >= minimum_cluster_size:
                detector = IsolationForest(
                    n_estimators=200,
                    max_samples="auto",
                    contamination="auto",
                    random_state=self.random_state + cluster + 1,
                    n_jobs=-1,
                ).fit(cluster_scaled)
            else:
                detector = self.global_detector
            self.detectors[cluster] = detector
            self.anomaly_reference[cluster] = np.sort(-detector.score_samples(cluster_scaled))

            distances = np.linalg.norm(
                cluster_scaled - self.clusterer.cluster_centers_[cluster], axis=1
            )
            self.distance_reference[cluster] = np.sort(distances)

            raw_cluster = features.loc[features.index[mask], self.feature_columns]
            median = raw_cluster.median()
            scale = (raw_cluster.quantile(0.75) - raw_cluster.quantile(0.25)).replace(0, np.nan)
            fallback = raw_cluster.std().replace(0, 1).fillna(1)
            self.cluster_medians[cluster] = median
            self.cluster_scales[cluster] = scale.fillna(fallback).replace(0, 1)

        self.is_fitted = True
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Attach behavioral segment, risk score, embedding, and alert reason."""

        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before scoring")
        self._validate(features)
        scaled = self.scaler.transform(self._prepare(features))
        clusters = self.clusterer.predict(scaled)
        anomaly_percentiles = np.zeros(len(features), dtype=float)
        distance_percentiles = np.zeros(len(features), dtype=float)

        for cluster in range(self.n_clusters):
            mask = clusters == cluster
            if not mask.any():
                continue
            cluster_scaled = scaled[mask]
            raw_anomaly = -self.detectors[cluster].score_samples(cluster_scaled)
            anomaly_percentiles[mask] = _percentile_rank(
                raw_anomaly, self.anomaly_reference[cluster]
            )
            distance = np.linalg.norm(
                cluster_scaled - self.clusterer.cluster_centers_[cluster], axis=1
            )
            distance_percentiles[mask] = _percentile_rank(
                distance, self.distance_reference[cluster]
            )

        embedding = self.pca.transform(scaled)
        result = features.copy()
        result["cluster"] = clusters
        result["anomaly_percentile"] = anomaly_percentiles
        result["cluster_distance_percentile"] = distance_percentiles
        result["risk_score"] = np.clip(
            100 * (0.82 * anomaly_percentiles + 0.18 * distance_percentiles), 0, 100
        ).round(2)
        result["embedding_x"] = embedding[:, 0]
        result["embedding_y"] = embedding[:, 1]
        explanations = [
            self.explain_factors(row, int(cluster), top_n=3)
            for (_, row), cluster in zip(result.iterrows(), clusters, strict=True)
        ]
        result["alert_reason"] = [factors[0] for factors in explanations]
        result["alert_factors"] = [" • ".join(factors) for factors in explanations]
        return result

    def explain_row(self, row: pd.Series, cluster: int) -> str:
        """Describe the feature furthest from its cluster's ordinary behavior."""

        return self.explain_factors(row, cluster, top_n=1)[0]

    def explain_factors(self, row: pd.Series, cluster: int, top_n: int = 3) -> list[str]:
        """Describe the largest robust behavioral deviations from a segment.

        These are interpretable supporting factors, not exact Isolation Forest feature
        contributions.
        """

        median = self.cluster_medians[cluster]
        scale = self.cluster_scales[cluster]
        values = row[self.feature_columns].astype(float)
        deviation = ((values - median) / scale).abs().astype(float)
        if int(row["out_tx_count"]) == 0:
            deviation.loc[["cross_currency_share", "cash_share"]] = -np.inf

        features: list[str] = []
        used_groups: set[str] = set()
        for feature in deviation.sort_values(ascending=False).index:
            feature = str(feature)
            group = EXPLANATION_GROUPS.get(feature, feature)
            if group in used_groups:
                continue
            features.append(feature)
            used_groups.add(group)
            if len(features) == top_n:
                break
        return [self._format_factor(feature, row, median) for feature in features]

    def _format_factor(self, feature: str, row: pd.Series, median: pd.Series) -> str:
        value = float(row[feature])
        baseline = float(median[feature])
        label = FEATURE_LABELS[feature]

        if feature in {"cross_currency_share", "cash_share"}:
            denominator = int(row["out_tx_count"])
            numerator = int(round(value * denominator))
            noun = "transaction" if denominator == 1 else "transactions"
            limited_sample = "; limited sample" if denominator < 3 else ""
            return (
                f"Unusual {label}: {numerator} of {denominator} outgoing {noun} "
                f"({value:.0%}) vs segment median {baseline:.0%}{limited_sample}"
            )
        if feature in MONEY_FEATURES:
            return f"Unusual {label}: ${value:,.2f} vs segment median ${baseline:,.2f}"
        if feature in COUNT_FEATURES:
            return f"Unusual {label}: {value:,.0f} vs segment median {baseline:,.0f}"
        return f"Unusual {label}: {value:,.2f} vs segment median {baseline:,.2f}"

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = features[self.feature_columns].astype(float).copy()
        for column in LOG_COLUMNS:
            prepared[column] = np.log1p(prepared[column].clip(lower=0))
        return prepared.replace([np.inf, -np.inf], 0).fillna(0)

    def _validate(self, features: pd.DataFrame) -> None:
        missing = set(self.feature_columns).difference(features.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")


def _percentile_rank(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    if len(sorted_reference) == 0:
        return np.zeros(len(values), dtype=float)
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks / len(sorted_reference)
