"""K-Means clustering for station segmentation (M4)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.generator import build_all_datasets
from utils.metrics import clustering_metrics


def _build_station_features() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build station-level aggregate features for clustering."""
    ds = build_all_datasets()
    df_daily = ds["df_daily"].copy()
    df_stations = ds["df_stations"]

    # Station-level aggregates
    agg = df_daily.groupby("station_id").agg(
        avg_volume=("volume_litres", "mean"),
        std_volume=("volume_litres", "std"),
        avg_gross_margin=("gross_margin", "mean"),
        avg_gross_profit=("gross_profit_eur", "mean"),
        avg_price=("our_price", "mean"),
        avg_price_gap=("our_price", lambda x: (x - df_daily.loc[x.index, "min_comp_price"]).mean()),
    ).reset_index()

    agg["std_volume"] = agg["std_volume"].fillna(0)
    agg["avg_price_gap"] = agg["avg_price_gap"].fillna(0)

    # Merge station metadata
    meta_cols = ["station_id", "n_competitors"]
    if "capacity_l_day" in df_stations.columns:
        meta_cols.append("capacity_l_day")
    agg = agg.merge(df_stations[meta_cols], on="station_id", how="left")
    agg["n_competitors"] = agg["n_competitors"].fillna(0).astype(int)

    # Elasticity proxy: volume sensitivity to price gap (simplified)
    def _elasticity(g):
        if len(g) < 5 or g["volume_litres"].std() < 1e-8:
            return 0.0
        try:
            r = np.corrcoef(g["our_price"] - g["min_comp_price"], g["volume_litres"])[0, 1]
            return float(r) if not np.isnan(r) else 0.0
        except Exception:
            return 0.0

    elasticity_proxy = df_daily.groupby("station_id").apply(_elasticity).reset_index()
    elasticity_proxy.columns = ["station_id", "elasticity_proxy"]
    elasticity_proxy["elasticity_proxy"] = elasticity_proxy["elasticity_proxy"].fillna(0)
    agg = agg.merge(elasticity_proxy, on="station_id", how="left")

    feature_cols = [
        "avg_volume", "std_volume", "avg_gross_margin", "avg_gross_profit",
        "avg_price", "avg_price_gap", "n_competitors", "elasticity_proxy",
    ]
    if "capacity_l_day" in agg.columns:
        agg["utilization"] = agg["avg_volume"] / agg["capacity_l_day"].clip(lower=1)
        feature_cols.append("utilization")

    X = agg[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return agg, X_scaled, feature_cols


def train_kmeans(k: int, features: list[str] | None = None) -> dict[str, Any]:
    """
    Train K-Means on station-level features.
    Returns cluster assignments, PCA biplot data, elbow data, silhouette per cluster, cluster profiles.
    """
    agg, X_scaled, all_features = _build_station_features()
    feature_cols = features or all_features
    col_indices = [all_features.index(c) for c in feature_cols if c in all_features]
    if not col_indices:
        col_indices = list(range(len(all_features)))
    X = X_scaled[:, col_indices]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    metrics = clustering_metrics(X, labels)
    silhouette = metrics.get("silhouette", 0.0)

    # Elbow data (inertia for k=2..min(10, n_samples-1))
    elbow_data = []
    for ki in range(2, min(11, len(X))):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X)
        elbow_data.append({"k": ki, "inertia": float(km.inertia_)})

    # PCA for biplot (2 components)
    n_components = min(2, X.shape[1], X.shape[0] - 1)
    if n_components >= 2:
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X)
        pca_loadings = pca.components_.T.tolist()
        pca_explained = pca.explained_variance_ratio_.tolist()
    else:
        pca_coords = np.zeros((len(X), 2))
        pca_loadings = []
        pca_explained = [1.0, 0.0]

    # Silhouette per cluster
    from sklearn.metrics import silhouette_samples
    sil_samples = silhouette_samples(X, labels)
    sil_per_cluster = []
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            sil_per_cluster.append({
                "cluster": int(c),
                "silhouette": float(np.mean(sil_samples[mask])),
                "count": int(mask.sum()),
            })

    # Cluster profiles (mean of each feature per cluster)
    agg["cluster"] = labels
    profiles = []
    for c in range(k):
        sub = agg[agg["cluster"] == c]
        profile = {"cluster": int(c), "count": len(sub)}
        for col in all_features:
            if col in sub.columns:
                profile[col] = float(sub[col].mean())
        profiles.append(profile)

    return {
        "labels": labels.tolist(),
        "cluster_ids": agg["station_id"].tolist(),
        "centroids": centroids.tolist(),
        "silhouette": silhouette,
        "pca_coords": pca_coords.tolist(),
        "pca_loadings": pca_loadings,
        "pca_explained": pca_explained,
        "feature_names": feature_cols,
        "elbow_data": elbow_data,
        "silhouette_per_cluster": sil_per_cluster,
        "cluster_profiles": profiles,
        "metrics": metrics,
    }
