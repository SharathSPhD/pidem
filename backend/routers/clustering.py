"""Clustering router: K-Means training and cluster profiles."""

from fastapi import APIRouter, HTTPException

from models.clustering import train_kmeans
from services.run_manager import complete_run, create_run, get_run

router = APIRouter(prefix="/api/v1/models/clustering", tags=["clustering"])


@router.post("/train")
def train_clustering(k: int = 4, features: list[str] | None = None):
    """Run K-Means clustering on station-level features."""
    run = create_run(est_seconds=10)
    try:
        result = train_kmeans(k=k, features=features)
        complete_run(run.run_id, metrics=result.get("metrics", {}), results=result)
        return {"run_id": run.run_id, "status": "complete", **result}
    except Exception as e:
        from services.run_manager import fail_run

        fail_run(run.run_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/profiles")
def get_cluster_profiles(run_id: str):
    """Get cluster profiles for a completed K-Means run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status.value != "complete":
        raise HTTPException(status_code=400, detail=f"Run not complete: {run.status.value}")
    profiles = run.results.get("cluster_profiles", [])
    return {"run_id": run_id, "cluster_profiles": profiles}
