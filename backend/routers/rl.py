"""Reinforcement learning API router."""

from typing import Literal

import plotly.graph_objects as go
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from models.rl_tabular import train_qlearning
from models.rl_dqn import train_dqn
from services.run_manager import create_run, complete_run, get_run, fail_run
from utils.chart_helpers import fig_to_response

router = APIRouter(prefix="/api/v1/rl", tags=["rl"])


class QLearningTrainRequest(BaseModel):
    gamma: float = 0.95
    episodes: int = 500
    epsilon_decay: float = 0.995
    competitor_model: Literal["static", "follow", "undercut"] = "static"


class DQNTrainRequest(BaseModel):
    hidden_layers: list[int] | None = None
    units: int = 64
    replay_size: int = 10000
    target_freq: int = 10
    env_difficulty: Literal["easy", "medium", "hard"] = "medium"


def _run_qlearning(run_id: str, request: QLearningTrainRequest):
    try:
        result = train_qlearning(
            gamma=request.gamma,
            episodes=request.episodes,
            epsilon_decay=request.epsilon_decay,
            competitor_model=request.competitor_model,
        )
        complete_run(
            run_id,
            metrics={"final_avg_reward": result.final_avg_reward, "n_episodes": result.n_episodes},
            results={
                "model_type": "qlearning",
                "q_table": result.q_table,
                "policy_heatmap": result.policy_heatmap,
                "reward_trajectory": result.reward_trajectory,
                "value_function_surface": result.value_function_surface,
            },
        )
    except Exception as e:
        fail_run(run_id, str(e))


def _run_dqn(run_id: str, request: DQNTrainRequest):
    try:
        result = train_dqn(
            hidden_layers=request.hidden_layers,
            units=request.units,
            replay_size=request.replay_size,
            target_freq=request.target_freq,
            env_difficulty=request.env_difficulty,
        )
        complete_run(
            run_id,
            metrics={"final_avg_reward": result.final_avg_reward, "n_episodes": result.n_episodes},
            results={
                "model_type": "dqn",
                "training_curve": result.training_curve,
                "q_value_landscape": result.q_value_landscape,
                "policy_comparison": result.policy_comparison,
            },
        )
    except Exception as e:
        fail_run(run_id, str(e))


@router.post("/qlearning/train")
def train_qlearning_endpoint(request: QLearningTrainRequest, background_tasks: BackgroundTasks):
    """Train tabular Q-learning (async)."""
    run = create_run(est_seconds=request.episodes // 10)
    background_tasks.add_task(_run_qlearning, run.run_id, request)
    return {"run_id": run.run_id, "status": "training"}


@router.post("/dqn/train")
def train_dqn_endpoint(request: DQNTrainRequest, background_tasks: BackgroundTasks):
    """Train DQN (async)."""
    run = create_run(est_seconds=120)
    background_tasks.add_task(_run_dqn, run.run_id, request)
    return {"run_id": run.run_id, "status": "training"}


@router.get("/{run_id}/policy")
def get_policy(run_id: str):
    """Get policy heatmap or Q-value landscape as Plotly figure."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run.results
    model_type = results.get("model_type", "")
    if model_type == "qlearning":
        hm = results.get("policy_heatmap", {})
        if not hm:
            return {"figure": None}
        fig = go.Figure(data=go.Heatmap(
            z=hm["z"], x=hm["x"], y=hm["y"], colorscale="Viridis",
        ))
        fig.update_layout(title="Policy Heatmap (State-Action)", xaxis_title="Action", yaxis_title="State")
    elif model_type == "dqn":
        ql = results.get("q_value_landscape", {})
        if not ql:
            return {"figure": None}
        fig = go.Figure(data=go.Heatmap(
            z=ql["z"], x=ql["x"], y=ql["y"], colorscale="Viridis",
        ))
        fig.update_layout(title="Q-Value Landscape", xaxis_title="Action", yaxis_title="State")
    else:
        return {"figure": None}
    return fig_to_response(fig)


@router.get("/{run_id}/convergence")
def get_convergence(run_id: str):
    """Get reward/training curve as Plotly figure."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run.results
    trajectory = results.get("reward_trajectory") or results.get("training_curve")
    if not trajectory:
        return {"figure": None}
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=trajectory, mode="lines", name="Episode Reward"))
    fig.update_layout(
        title="Training Convergence",
        xaxis_title="Episode",
        yaxis_title="Reward",
    )
    return fig_to_response(fig)


@router.get("/{run_id}/comparison")
def get_comparison(run_id: str):
    """Get policy comparison (DQN vs tabular) or value function surface."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run.results
    if results.get("model_type") == "dqn":
        pc = results.get("policy_comparison", {})
        if "error" in pc:
            return {"policy_comparison": pc}
        fig = go.Figure(data=[
            go.Bar(name="Tabular", x=["Final Reward"], y=[pc.get("tabular_final_reward", 0)]),
            go.Bar(name="DQN", x=["Final Reward"], y=[pc.get("dqn_final_reward", 0)]),
        ])
        fig.update_layout(title="Policy Comparison: Tabular vs DQN", barmode="group")
        return {"policy_comparison": pc, **fig_to_response(fig)}
    # Q-learning: return value function surface as figure
    vf = results.get("value_function_surface", {})
    if not vf:
        return {"figure": None}
    fig = go.Figure(data=[go.Surface(z=vf["z"], x=vf["x"], y=vf["y"], colorscale="Plasma")])
    fig.update_layout(title="Value Function Surface", scene=dict(xaxis_title="Volume", yaxis_title="Price Gap"))
    return fig_to_response(fig)
