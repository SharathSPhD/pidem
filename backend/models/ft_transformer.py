"""FT-Transformer for M14 - volume prediction with feature tokenization and self-attention."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data.generator import build_all_datasets
from utils.chart_helpers import fig_to_response


def _safe_r2(actual: np.ndarray, pred: np.ndarray) -> float:
    """R^2 that handles zero-variance actuals."""
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    ss_res = float(np.sum((pred - actual) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return 1 - ss_res / ss_tot


FEATURE_COLS = [
    "our_price",
    "cogs",
    "gross_margin",
    "min_comp_price",
    "station_type",
    "temperature",
    "highway_index",
]


def _prepare_data() -> tuple[pd.DataFrame, np.ndarray, StandardScaler, dict]:
    """Build daily data for volume prediction."""
    ds = build_all_datasets()
    df = ds["df_daily"].copy()
    df = df.dropna(subset=["volume_litres"] + [c for c in FEATURE_COLS if c in df.columns])

    # Encode station_type
    stype_map = {"motorway": 0, "urban": 1, "rural": 2}
    df["station_type_enc"] = df["station_type"].map(stype_map).fillna(0).astype(int)

    num_cols = ["our_price", "cogs", "gross_margin", "min_comp_price", "temperature", "highway_index"]
    cat_cols = ["station_type_enc"]
    use_cols = [c for c in num_cols + cat_cols if c in df.columns]
    X = df[use_cols].values.astype(np.float32)
    y = np.log1p(df["volume_litres"].values).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = use_cols

    return df, X_scaled, scaler, {"feature_names": feature_names}


if TORCH_AVAILABLE:
    class FeatureTokenizer(nn.Module):
        """Linear projection of each feature to d_model."""

        def __init__(self, n_features: int, d_model: int):
            super().__init__()
            self.proj = nn.Linear(1, d_model)
            self.n_features = n_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x.unsqueeze(-1))

    class FTTransformer(nn.Module):
        """Simplified FT-Transformer: feature tokens + multi-head self-attention + [CLS] head."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            self.tokenizer = FeatureTokenizer(n_features, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(
            self, x: torch.Tensor, return_attention: bool = False
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            B = x.shape[0]
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tokens], dim=1)

            out = self.transformer(seq)
            cls_out = out[:, 0, :]
            pred = self.head(cls_out).squeeze(-1)

            if return_attention:
                attn_weights = torch.softmax(out[:, 0, :] @ out[:, 1:, :].transpose(-2, -1), dim=-1)
                return pred, attn_weights
            return pred


def _get_attention_weights(model, X: np.ndarray) -> np.ndarray:
    """Extract CLS attention to feature tokens."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X[: min(256, len(X))].astype(np.float32))
        _, attn = model(x, return_attention=True)
        return attn.mean(0).numpy()


def _get_feature_attention_heatmap(model, X: np.ndarray) -> np.ndarray:
    """Feature-feature attention (simplified: use token similarity)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X[: min(128, len(X))].astype(np.float32))
        tokens = model.tokenizer(x)
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = model.transformer(seq)
        feat_tokens = out[:, 1:, :]
        sim = torch.bmm(feat_tokens, feat_tokens.transpose(-2, -1))
        return sim.mean(0).numpy()


def train_ft_transformer(
    n_heads: int = 4,
    n_layers: int = 2,
    compare_mode: bool = True,
) -> dict[str, Any]:
    """
    Train FT-Transformer for volume prediction.
    Returns attention heatmap, CLS weights, comparison vs XGBoost/MLP/Linear, calibration.
    """
    if not TORCH_AVAILABLE:
        return {
            "attention_heatmap": [],
            "feature_names": [],
            "cls_attention": [],
            "comparison": {"error": "PyTorch not installed"},
            "calibration": [],
        }

    df, X, scaler, meta = _prepare_data()
    feature_names = meta["feature_names"]
    n_features = X.shape[1]
    y = np.log1p(df["volume_litres"].values).astype(np.float32)

    # Train/val split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    X_tr, X_val = X[idx[:split]], X[idx[split:]]
    y_tr, y_val = y[idx[:split]], y[idx[split:]]

    d_model = 64
    model = FTTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n_epochs = 30
    batch_size = 64
    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            batch_idx = perm[i : i + batch_size]
            x_b = torch.from_numpy(X_tr[batch_idx].astype(np.float32))
            y_b = torch.from_numpy(y_tr[batch_idx].astype(np.float32))
            opt.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val.astype(np.float32))
        pred_val = model(x_val).numpy()

    ft_mae = float(np.mean(np.abs(np.expm1(pred_val) - np.expm1(y_val))))
    ft_r2 = _safe_r2(np.expm1(y_val), np.expm1(pred_val))

    cls_attention = _get_attention_weights(model, X_val)
    feat_heatmap = _get_feature_attention_heatmap(model, X_val)

    comparison = {"FT-Transformer": {"mae": ft_mae, "r2": ft_r2}}

    if compare_mode:
        lm = LinearRegression()
        lm.fit(X_tr, y_tr)
        lm_pred = lm.predict(X_val)
        comparison["Linear"] = {
            "mae": float(np.mean(np.abs(np.expm1(lm_pred) - np.expm1(y_val)))),
            "r2": float(lm.score(X_val, y_val)),
        }

        mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for _ in range(50):
            perm = np.random.permutation(len(X_tr))
            for i in range(0, len(X_tr), batch_size):
                batch_idx = perm[i : i + batch_size]
                x_b = torch.from_numpy(X_tr[batch_idx].astype(np.float32))
                y_b = torch.from_numpy(y_tr[batch_idx].astype(np.float32)).unsqueeze(1)
                opt_mlp.zero_grad()
                p = mlp(x_b)
                criterion(p.squeeze(), y_b.squeeze()).backward()
                opt_mlp.step()
        mlp.eval()
        with torch.no_grad():
            mlp_pred = mlp(torch.from_numpy(X_val.astype(np.float32))).squeeze().numpy()
        comparison["MLP"] = {
            "mae": float(np.mean(np.abs(np.expm1(mlp_pred) - np.expm1(y_val)))),
            "r2": _safe_r2(np.expm1(y_val), np.expm1(mlp_pred)),
        }

        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
            xgb_model.fit(X_tr, y_tr)
            xgb_pred = xgb_model.predict(X_val)
            comparison["XGBoost"] = {
                "mae": float(np.mean(np.abs(np.expm1(xgb_pred) - np.expm1(y_val)))),
                "r2": _safe_r2(np.expm1(y_val), np.expm1(xgb_pred)),
            }
        except ImportError:
            comparison["XGBoost"] = {"mae": None, "r2": None}

    pred_vol = np.expm1(pred_val)
    actual_vol = np.expm1(y_val)
    bins = np.percentile(pred_vol, [0, 25, 50, 75, 100])
    calibration = []
    for i in range(len(bins) - 1):
        mask = (pred_vol >= bins[i]) & (pred_vol < bins[i + 1])
        if mask.sum() > 0:
            calibration.append({
                "bin": i,
                "pred_mean": float(pred_vol[mask].mean()),
                "actual_mean": float(actual_vol[mask].mean()),
                "count": int(mask.sum()),
            })

    return {
        "attention_heatmap": feat_heatmap.tolist(),
        "feature_names": feature_names,
        "cls_attention": cls_attention.tolist(),
        "comparison": comparison,
        "calibration": calibration,
    }


def build_attention_map_figure(attention_heatmap: list, feature_names: list[str]) -> dict:
    """Build feature-feature attention heatmap."""
    arr = np.array(attention_heatmap)
    if arr.size == 0:
        fig = go.Figure()
        fig.update_layout(title="No attention data")
        return fig_to_response(fig)

    fig = go.Figure(data=go.Heatmap(
        z=arr,
        x=feature_names,
        y=feature_names,
        colorscale="Blues",
    ))
    fig.update_layout(
        title="Feature-Feature Attention Heatmap",
        xaxis_title="Feature",
        yaxis_title="Feature",
    )
    return fig_to_response(fig)


def build_cls_attention_figure(cls_attention: list, feature_names: list[str]) -> dict:
    """Build CLS attention weights bar chart."""
    arr = np.array(cls_attention)
    if arr.ndim > 1:
        arr = arr.mean(0)
    if arr.size == 0:
        fig = go.Figure()
        fig.update_layout(title="No CLS attention data")
        return fig_to_response(fig)

    names = feature_names[: len(arr)]
    fig = go.Figure(data=go.Bar(x=names, y=arr.tolist()))
    fig.update_layout(
        title="CLS Token Attention to Features",
        xaxis_title="Feature",
        yaxis_title="Attention weight",
    )
    return fig_to_response(fig)


def build_comparison_figure(comparison: dict) -> dict:
    """Build model comparison bar chart."""
    if "error" in comparison:
        fig = go.Figure()
        fig.add_annotation(text=comparison["error"], xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Model Comparison")
        return fig_to_response(fig)

    models = [m for m in comparison if isinstance(comparison[m], dict) and comparison[m].get("mae") is not None]
    mae_vals = [comparison[m]["mae"] for m in models]
    r2_vals = [comparison[m].get("r2", 0) or 0 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE (L)", x=models, y=mae_vals))
    fig.add_trace(go.Bar(name="R\u00b2", x=models, y=r2_vals))
    fig.update_layout(
        title="Model Comparison: FT-Transformer vs Baselines",
        barmode="group",
        yaxis_title="Metric value",
    )
    return fig_to_response(fig)


def build_calibration_figure(calibration: list[dict]) -> dict:
    """Build calibration plot (predicted vs actual mean per bin)."""
    if not calibration:
        fig = go.Figure()
        fig.update_layout(title="No calibration data")
        return fig_to_response(fig)

    pred_means = [c["pred_mean"] for c in calibration]
    actual_means = [c["actual_mean"] for c in calibration]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_means, y=actual_means, mode="markers+lines", name="Calibration"))
    fig.add_trace(go.Scatter(x=pred_means, y=pred_means, mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.update_layout(
        title="Calibration: Predicted vs Actual Mean Volume",
        xaxis_title="Predicted mean (L)",
        yaxis_title="Actual mean (L)",
    )
    return fig_to_response(fig)
