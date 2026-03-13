"""Neural network routers for M13 (MLP) and M14 (FT-Transformer)."""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from models.ft_transformer import (
    build_attention_map_figure,
    build_calibration_figure,
    build_cls_attention_figure,
    build_comparison_figure,
    train_ft_transformer,
)
from models.mlp import (
    build_activation_histogram_figure,
    build_decision_boundary_figure,
    build_embedding_figure,
    build_gradient_figure,
    build_loss_curve_figure,
    train_mlp,
)
from services.run_manager import complete_run, create_run, fail_run, get_run

router = APIRouter(prefix="/api/v1/neural", tags=["neural"])


def _run_mlp_training(run_id: str, params: dict[str, Any]) -> None:
    """Synchronous MLP training (runs in thread)."""
    try:
        out = train_mlp(
            n_layers=params.get("n_layers", 2),
            units_per_layer=params.get("units_per_layer", [32, 16]),
            activation=params.get("activation", "relu"),
            task=params.get("task", "binary"),
            embedding_dim=params.get("embedding_dim", 8),
        )
        decision_fig = build_decision_boundary_figure(out["decision_boundary_frames"])
        gradient_fig = build_gradient_figure(out["gradient_flow"])
        embedding_fig = build_embedding_figure(out["embedding_pca"])
        activation_fig = build_activation_histogram_figure(out["activation_histograms"])
        loss_fig = build_loss_curve_figure(out["loss_curve"])

        complete_run(
            run_id,
            metrics={"loss_final": out["loss_curve"][-1] if out["loss_curve"] else None},
            results={
                "model_type": "mlp",
                "decision_boundary": decision_fig,
                "gradients": gradient_fig,
                "embeddings": embedding_fig,
                "activation_histograms": activation_fig,
                "loss_curve": loss_fig,
                "raw": out,
            },
        )
    except Exception as e:
        fail_run(run_id, str(e))


def _run_ft_transformer_training(run_id: str, params: dict[str, Any]) -> None:
    """Synchronous FT-Transformer training (runs in thread)."""
    try:
        out = train_ft_transformer(
            n_heads=params.get("n_heads", 4),
            n_layers=params.get("n_layers", 2),
            compare_mode=params.get("compare_mode", True),
        )
        attention_fig = build_attention_map_figure(
            out["attention_heatmap"], out["feature_names"]
        )
        cls_fig = build_cls_attention_figure(out["cls_attention"], out["feature_names"])
        comparison_fig = build_comparison_figure(out["comparison"])
        calibration_fig = build_calibration_figure(out["calibration"])

        complete_run(
            run_id,
            metrics={
                "ft_mae": out["comparison"].get("FT-Transformer", {}).get("mae"),
                "ft_r2": out["comparison"].get("FT-Transformer", {}).get("r2"),
            },
            results={
                "model_type": "ft_transformer",
                "attention_map": attention_fig,
                "cls_attention": cls_fig,
                "comparison": comparison_fig,
                "calibration": calibration_fig,
                "raw": out,
            },
        )
    except Exception as e:
        fail_run(run_id, str(e))


@router.post("/mlp/train")
async def train_mlp_endpoint(
    background_tasks: BackgroundTasks,
    n_layers: int = 2,
    units_per_layer: str = "[32, 16]",
    activation: str = "relu",
    task: str = "binary",
    embedding_dim: int = 8,
):
    """Start async MLP training. Returns run_id for polling."""
    import json
    try:
        units = json.loads(units_per_layer)
    except (json.JSONDecodeError, TypeError):
        units = [32, 16]

    run = create_run(est_seconds=60)
    params = {
        "n_layers": n_layers,
        "units_per_layer": units,
        "activation": activation,
        "task": task,
        "embedding_dim": embedding_dim,
    }
    background_tasks.add_task(_run_mlp_training, run.run_id, params)
    return {"run_id": run.run_id, "status": "training"}


@router.post("/ft_transformer/train")
async def train_ft_transformer_endpoint(
    background_tasks: BackgroundTasks,
    n_heads: int = 4,
    n_layers: int = 2,
    compare_mode: bool = True,
):
    """Start async FT-Transformer training. Returns run_id for polling."""
    run = create_run(est_seconds=45)
    params = {"n_heads": n_heads, "n_layers": n_layers, "compare_mode": compare_mode}
    background_tasks.add_task(_run_ft_transformer_training, run.run_id, params)
    return {"run_id": run.run_id, "status": "training"}


def _get_run_or_404(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status.value == "failed":
        raise HTTPException(status_code=400, detail=f"Run failed: {run.error}")
    if run.status.value != "complete":
        raise HTTPException(status_code=202, detail=f"Run still {run.status.value}")
    return run


@router.get("/{run_id}/decision_boundary")
def get_decision_boundary(run_id: str):
    """Get decision boundary animation (MLP only)."""
    run = _get_run_or_404(run_id)
    if run.results.get("model_type") != "mlp":
        raise HTTPException(status_code=400, detail="Not an MLP run")
    return run.results.get("decision_boundary", {})


@router.get("/{run_id}/gradients")
def get_gradients(run_id: str):
    """Get gradient flow figure (MLP only)."""
    run = _get_run_or_404(run_id)
    if run.results.get("model_type") != "mlp":
        raise HTTPException(status_code=400, detail="Not an MLP run")
    return run.results.get("gradients", {})


@router.get("/{run_id}/embeddings")
def get_embeddings(run_id: str):
    """Get station embedding PCA figure (MLP only)."""
    run = _get_run_or_404(run_id)
    if run.results.get("model_type") != "mlp":
        raise HTTPException(status_code=400, detail="Not an MLP run")
    return run.results.get("embeddings", {})


@router.get("/{run_id}/attention_map")
def get_attention_map(run_id: str):
    """Get feature-feature attention heatmap (FT-Transformer only)."""
    run = _get_run_or_404(run_id)
    if run.results.get("model_type") != "ft_transformer":
        raise HTTPException(status_code=400, detail="Not an FT-Transformer run")
    return run.results.get("attention_map", {})


@router.get("/{run_id}/comparison")
def get_comparison(run_id: str):
    """Get model comparison figure (FT-Transformer only)."""
    run = _get_run_or_404(run_id)
    if run.results.get("model_type") != "ft_transformer":
        raise HTTPException(status_code=400, detail="Not an FT-Transformer run")
    return run.results.get("comparison", {})


@router.get("/{run_id}/status")
def get_run_status(run_id: str):
    """Get run status and metrics when complete."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "progress": run.progress,
        "metrics": run.metrics,
        "error": run.error,
    }
