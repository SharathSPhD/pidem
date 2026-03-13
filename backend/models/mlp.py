"""MLP Architecture Explorer for M13 - binary classification (price_gap > 0 -> competitive_zone)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data.generator import build_all_datasets
from utils.chart_helpers import fig_to_response


def _prepare_data() -> tuple[torch.Tensor, torch.Tensor, np.ndarray, pd.DataFrame, StandardScaler]:
    """Build hourly data with price_gap and hour for binary classification."""
    ds = build_all_datasets()
    df_daily = ds["df_daily"].copy()
    df_hourly = ds["df_hourly"].copy()

    # Merge daily to get min_comp_price for hourly rows
    daily_sub = df_daily[["station_id", "date", "min_comp_price", "station_type"]].drop_duplicates()
    df_hourly["date"] = pd.to_datetime(df_hourly["datetime"]).dt.strftime("%Y-%m-%d")
    df = df_hourly.merge(daily_sub, on=["station_id", "date"], how="left")
    df["min_comp_price"] = df["min_comp_price"].fillna(df["our_price"])
    df["price_gap"] = df["our_price"] - df["min_comp_price"]
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

    # Binary target: price_gap > 0 -> competitive_zone (1)
    df["target"] = (df["price_gap"] > 0).astype(int)

    # Encode station_id for embedding
    le = LabelEncoder()
    df["station_idx"] = le.fit_transform(df["station_id"].astype(str))

    X_num = df[["price_gap", "hour"]].values.astype(np.float32).copy()
    X_station = df["station_idx"].values.copy()
    y = df["target"].values.astype(np.int64)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    return (
        torch.from_numpy(X_num_scaled),
        torch.from_numpy(X_station),
        y,
        df,
        scaler,
    )


class MLPClassifier(nn.Module):
    """MLP with station embedding and configurable architecture."""

    def __init__(
        self,
        n_stations: int,
        embedding_dim: int,
        n_layers: int,
        units_per_layer: list[int],
        activation: str,
        n_features: int = 2,
    ):
        super().__init__()
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_stations, embedding_dim)
        in_dim = n_features + embedding_dim

        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_fn = acts.get(activation, nn.ReLU)

        layers = []
        for i, units in enumerate(units_per_layer[:n_layers]):
            layers.append(nn.Linear(in_dim, units))
            layers.append(act_fn())
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_station: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_station)
        x = torch.cat([x_num, emb], dim=1)
        return self.mlp(x).squeeze(-1)


def train_mlp(
    n_layers: int = 2,
    units_per_layer: list[int] | None = None,
    activation: str = "relu",
    task: str = "binary",
    embedding_dim: int = 8,
) -> dict[str, Any]:
    """
    Train MLP for binary classification (price_gap > 0 -> competitive_zone).
    Returns decision boundary frames, gradient flow, embedding PCA, activation histograms, loss curve.
    """
    if units_per_layer is None:
        units_per_layer = [32, 16]

    X_num, X_station, y, df, scaler = _prepare_data()
    n_stations = int(X_station.max().item()) + 1

    model = MLPClassifier(
        n_stations=n_stations,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        units_per_layer=units_per_layer,
        activation=activation,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Track gradient magnitudes per layer
    gradient_flow: list[dict[str, float]] = []
    loss_curve: list[float] = []
    decision_frames: list[dict] = []
    activation_histograms: list[dict] = []

    n_epochs = 50
    batch_size = 64
    n = len(X_num)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x_n = X_num[idx]
            x_s = X_station[idx]
            target = torch.from_numpy(y[idx]).float().unsqueeze(1)

            opt.zero_grad()
            logits = model(x_n, x_s)
            loss = criterion(logits.unsqueeze(1), target)
            loss.backward()

            # Record gradient magnitudes
            grad_mags = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_mags[name] = float(p.grad.norm().item())
            if epoch % 5 == 0 and i == 0:
                gradient_flow.append({"epoch": epoch, **grad_mags})

            opt.step()
            epoch_loss += loss.item()

        loss_curve.append(epoch_loss / max(1, n // batch_size))

        # Decision boundary frames at selected epochs
        if epoch in [0, 10, 25, 49]:
            model.eval()
            with torch.no_grad():
                pg_min, pg_max = float(X_num[:, 0].min()), float(X_num[:, 0].max())
                hr_min, hr_max = float(X_num[:, 1].min()), float(X_num[:, 1].max())
                pg_grid = np.linspace(pg_min - 0.5, pg_max + 0.5, 40)
                hr_grid = np.linspace(hr_min - 0.5, hr_max + 0.5, 40)
                PG, HR = np.meshgrid(pg_grid, hr_grid)
                flat_pg = PG.ravel()
                flat_hr = HR.ravel()
                # Use station 0 as reference for grid
                x_grid_num = torch.from_numpy(
                    np.column_stack([flat_pg, flat_hr]).astype(np.float32)
                )
                x_grid_station = torch.zeros(len(flat_pg), dtype=torch.long)
                logits_grid = model(x_grid_num, x_grid_station)
                probs = torch.sigmoid(logits_grid).numpy().reshape(PG.shape)
            decision_frames.append(
                {
                    "epoch": epoch,
                    "pg_grid": pg_grid.tolist(),
                    "hr_grid": hr_grid.tolist(),
                    "probs": probs.tolist(),
                }
            )

    # Station embedding PCA
    model.eval()
    with torch.no_grad():
        all_stations = torch.arange(n_stations, dtype=torch.long)
        emb = model.embedding(all_stations).numpy()
    n_components = min(2, n_stations - 1, emb.shape[1])
    if n_components >= 2:
        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(emb)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        emb_pca = np.zeros((n_stations, 2))
        explained = [1.0, 0.0]

    # Activation distribution (sample forward pass)
    model.eval()
    act_hist_data: dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            act_hist_data[name] = out.detach().numpy().ravel()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU)):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        _ = model(X_num[:256], X_station[:256])

    for h in hooks:
        h.remove()

    # Build activation histograms
    for name, vals in act_hist_data.items():
        hist, edges = np.histogram(vals, bins=30)
        activation_histograms.append(
            {"layer": name, "counts": hist.tolist(), "edges": edges.tolist()}
        )

    return {
        "decision_boundary_frames": decision_frames,
        "gradient_flow": gradient_flow,
        "embedding_pca": {
            "coords": emb_pca.tolist(),
            "explained_variance": explained,
        },
        "activation_histograms": activation_histograms,
        "loss_curve": loss_curve,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def build_decision_boundary_figure(frames: list[dict]) -> dict:
    """Build Plotly figure with animation frames for decision boundary."""
    if not frames:
        fig = go.Figure()
        fig.update_layout(title="No decision boundary data")
        return fig_to_response(fig)

    f0 = frames[0]
    pg = np.array(f0["pg_grid"])
    hr = np.array(f0["hr_grid"])
    probs = np.array(f0["probs"])

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=pg,
            y=hr,
            z=probs,
            colorscale="RdYlGn",
            contours=dict(showlabels=True),
            colorbar=dict(title="P(competitive)"),
        )
    )
    fig.update_layout(
        title="Decision Boundary (price_gap vs hour) - Epoch 0",
        xaxis_title="Price gap (scaled)",
        yaxis_title="Hour (scaled)",
    )

    if len(frames) > 1:
        fig.frames = []
        for f in frames:
            fig.frames.append(
                go.Frame(
                    data=[
                        go.Contour(
                            x=f["pg_grid"],
                            y=f["hr_grid"],
                            z=f["probs"],
                            colorscale="RdYlGn",
                            contours=dict(showlabels=True),
                        )
                    ],
                    name=str(f["epoch"]),
                    layout=go.Layout(title=f"Epoch {f['epoch']}"),
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500}}]),
                        dict(label="Pause", method="animate", args=[[None]]),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            label=f"Epoch {f['epoch']}",
                            method="animate",
                            args=[[f["epoch"]], {"frame": {"duration": 0}}],
                        )
                        for f in frames
                    ],
                )
            ],
        )

    return fig_to_response(fig)


def build_gradient_figure(gradient_flow: list[dict]) -> dict:
    """Build gradient magnitude over training figure."""
    if not gradient_flow:
        fig = go.Figure()
        fig.update_layout(title="No gradient data")
        return fig_to_response(fig)

    epochs = [g["epoch"] for g in gradient_flow]
    fig = go.Figure()
    for key in gradient_flow[0]:
        if key == "epoch":
            continue
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[g[key] for g in gradient_flow],
                name=key,
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title="Gradient Magnitudes per Layer",
        xaxis_title="Epoch",
        yaxis_title="Gradient norm",
        legend=dict(orientation="h"),
    )
    return fig_to_response(fig)


def build_embedding_figure(embedding_pca: dict) -> dict:
    """Build station embedding PCA scatter."""
    coords = embedding_pca.get("coords", [])
    if not coords:
        fig = go.Figure()
        fig.update_layout(title="No embedding data")
        return fig_to_response(fig)

    coords = np.array(coords)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(size=8),
            text=[f"Station {i}" for i in range(len(coords))],
        )
    )
    ev = embedding_pca.get("explained_variance", [1, 0])
    fig.update_layout(
        title=f"Station Embedding PCA (explained: {ev[0]:.2%}, {ev[1]:.2%})",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    return fig_to_response(fig)


def build_activation_histogram_figure(activation_histograms: list[dict]) -> dict:
    """Build activation distribution histograms."""
    from plotly.subplots import make_subplots

    if not activation_histograms:
        fig = go.Figure()
        fig.update_layout(title="No activation data")
        return fig_to_response(fig)

    n = len(activation_histograms)
    fig = make_subplots(rows=(n + 1) // 2, cols=min(2, n), subplot_titles=[h["layer"] for h in activation_histograms])
    for i, h in enumerate(activation_histograms):
        edges = np.array(h["edges"])
        counts = np.array(h["counts"])
        mid = (edges[:-1] + edges[1:]) / 2
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Bar(x=mid.tolist(), y=counts.tolist(), name=h["layer"]),
            row=row,
            col=col,
        )
    fig.update_layout(title="Activation Distribution per Layer")
    return fig_to_response(fig)


def build_loss_curve_figure(loss_curve: list[float]) -> dict:
    """Build training loss curve figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_curve, mode="lines", name="Loss"))
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="BCE Loss",
    )
    return fig_to_response(fig)
