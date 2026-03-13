"""Classification API: threat classifier training, tree structure, SHAP waterfall."""

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, Field

from data.generator import build_all_datasets
from models.classification import train_threat_classifier, get_shap_waterfall_for_observation
from services.run_manager import create_run, complete_run, get_run, fail_run
from utils.chart_helpers import fig_to_response

router = APIRouter(prefix="/api/v1/models/classification", tags=["classification"])

MODEL_TYPES = ("logistic", "tree", "xgboost")


class TrainRequest(BaseModel):
    model_type: str = Field("logistic", description="logistic, tree, or xgboost")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    tree_depth: int = Field(5, ge=1, le=20, description="Max depth for tree/xgboost")
    features: list[str] | None = Field(None, description="Override default feature list")


@router.post("/train")
def train_classifier(req: TrainRequest | None = None):
    """Train threat classifier; returns metrics, confusion matrix with costs, ROC, PR, SHAP beeswarm."""
    req = req or TrainRequest()
    if req.model_type.lower() not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {MODEL_TYPES}")

    run = create_run(est_seconds=20)

    try:
        ds = build_all_datasets()
        df = ds["df_daily"]
        if df.empty:
            raise ValueError("df_daily is empty")

        result = train_threat_classifier(
            features=df,
            model_type=req.model_type,
            threshold=req.threshold,
            tree_depth=req.tree_depth,
            feature_list=req.features,
        )

        # Serialize figures for response
        figures_json = {}
        for name, fig in result["figures"].items():
            if fig is not None:
                figures_json[name] = fig_to_response(fig)

        metrics = result["metrics"]

        # Store training result for GET /tree and GET /shap (in-memory; contains non-JSON-serializable objects)
        training_result = {
            "tree_structure": result["tree_structure"],
            "X": result["X"],
            "shap_values": result["shap_values"],
            "feature_names": result["feature_names"],
        }

        results = {
            "metrics": metrics,
            "tree_structure": result["tree_structure"],
            "figures": figures_json,
            "training_result": training_result,
        }

        complete_run(run.run_id, metrics=metrics, results=results)

        return {
            "run_id": run.run_id,
            "status": "complete",
            "metrics": metrics,
            "figures": figures_json,
            "tree_structure": result["tree_structure"],
        }
    except ValueError as e:
        fail_run(run.run_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        fail_run(run.run_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/tree")
def get_tree_structure(run_id: str):
    """Return decision tree structure (text) for a completed tree/xgboost run."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status.value != "complete":
        raise HTTPException(status_code=400, detail=f"Run not complete: {run.status.value}")

    tree_structure = run.results.get("tree_structure")
    if tree_structure is None:
        raise HTTPException(
            status_code=400,
            detail="Tree structure only available for model_type=tree. Use a tree model and retrain.",
        )

    return {"run_id": run_id, "tree_structure": tree_structure}


@router.get("/{run_id}/shap/{observation_idx}")
def get_shap_waterfall(run_id: str, observation_idx: int):
    """Return SHAP waterfall Plotly figure for one observation."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status.value != "complete":
        raise HTTPException(status_code=400, detail=f"Run not complete: {run.status.value}")

    training_result = run.results.get("training_result", {})
    if not training_result:
        raise HTTPException(status_code=404, detail="No training result stored for this run")

    # Reconstruct result dict for get_shap_waterfall_for_observation
    result = {
        "shap_values": training_result.get("shap_values"),
        "X": training_result.get("X"),
        "feature_names": training_result.get("feature_names", []),
    }

    fig = get_shap_waterfall_for_observation(result, observation_idx)
    if fig is None:
        raise HTTPException(
            status_code=404,
            detail=f"SHAP not available or observation_idx {observation_idx} out of range",
        )

    return fig_to_response(fig)
