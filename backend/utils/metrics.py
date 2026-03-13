"""Model evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100), 2),
    }


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ValueError:
            pass
    return m


def clustering_metrics(X, labels) -> dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    m = {"n_clusters": n_clusters}
    if n_clusters > 1:
        m["silhouette"] = round(float(silhouette_score(X, labels)), 4)
    return m


def forecast_metrics(y_true, y_pred) -> dict:
    base = regression_metrics(y_true, y_pred)
    base["mape"] = round(float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100), 2)
    return base
