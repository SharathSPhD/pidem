"""Tabular Q-learning for pricing MDP."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go

# MDP: 5 price_gap x 3 volume_index x 9 time_bucket x 3 competitor_trend = 405 states
N_PRICE_GAP = 5
N_VOLUME_IDX = 3
N_TIME_BUCKET = 9
N_COMP_TREND = 3
N_STATES = N_PRICE_GAP * N_VOLUME_IDX * N_TIME_BUCKET * N_COMP_TREND  # 405

# Actions: keep, -2ct, -1ct, +1ct, +2ct
ACTIONS = np.array([0, -2, -1, 1, 2])  # ct deltas
N_ACTIONS = len(ACTIONS)


@dataclass
class QLearningResult:
    q_table: dict
    policy_heatmap: dict
    reward_trajectory: list[float]
    value_function_surface: dict
    n_episodes: int
    final_avg_reward: float


def _state_to_idx(price_gap: int, vol_idx: int, time_bucket: int, comp_trend: int) -> int:
    return (price_gap * N_VOLUME_IDX * N_TIME_BUCKET * N_COMP_TREND +
            vol_idx * N_TIME_BUCKET * N_COMP_TREND +
            time_bucket * N_COMP_TREND + comp_trend)


def _idx_to_state(idx: int) -> tuple[int, int, int, int]:
    comp_trend = idx % N_COMP_TREND
    idx //= N_COMP_TREND
    time_bucket = idx % N_TIME_BUCKET
    idx //= N_TIME_BUCKET
    vol_idx = idx % N_VOLUME_IDX
    idx //= N_VOLUME_IDX
    price_gap = idx % N_PRICE_GAP
    return price_gap, vol_idx, time_bucket, comp_trend


def _simulate_step(
    price_gap: int, vol_idx: int, time_bucket: int, comp_trend: int,
    action: int, competitor_model: Literal["static", "follow", "undercut"], rng
) -> tuple[int, float]:
    """One step: (s, a) -> (s', r)."""
    # Price gap bins: -4ct, -2ct, 0, +2ct, +4ct
    new_gap = np.clip(price_gap + (action // 2), 0, N_PRICE_GAP - 1)
    # Volume: low/med/high based on price attractiveness
    vol_mult = [0.7, 1.0, 1.3][vol_idx]
    if new_gap < 2:  # we're cheaper
        vol_mult *= 1.1
    elif new_gap > 2:
        vol_mult *= 0.9
    # Competitor trend: 0=stable, 1=increasing, 2=decreasing
    if competitor_model == "follow":
        comp_trend = min(2, comp_trend + (1 if action > 0 else -1 if action < 0 else 0))
    elif competitor_model == "undercut":
        comp_trend = 1 if rng.random() < 0.3 else comp_trend
    comp_trend = np.clip(comp_trend, 0, N_COMP_TREND - 1)
    # Time advances
    time_bucket = (time_bucket + 1) % N_TIME_BUCKET
    # Volume index: depends on price gap
    vol_idx = min(2, max(0, vol_idx + (1 if new_gap < 2 else -1)))
    vol_idx = np.clip(vol_idx, 0, N_VOLUME_IDX - 1)
    # Reward: margin proxy
    margin_per_l = 15  # ct/L
    base_vol = 30000
    reward = base_vol * vol_mult * (margin_per_l / 100) * (0.9 + 0.1 * (N_PRICE_GAP - new_gap) / N_PRICE_GAP)
    reward += rng.normal(0, reward * 0.05)
    return _state_to_idx(new_gap, vol_idx, time_bucket, comp_trend), float(reward)


def train_qlearning(
    gamma: float = 0.95,
    episodes: int = 500,
    epsilon_decay: float = 0.995,
    competitor_model: Literal["static", "follow", "undercut"] = "static",
) -> QLearningResult:
    """Train tabular Q-learning on pricing MDP."""
    rng = np.random.default_rng(42)
    q = np.zeros((N_STATES, N_ACTIONS))
    eps = 1.0
    reward_trajectory = []

    for ep in range(episodes):
        s = rng.integers(0, N_STATES)
        ep_reward = 0.0
        for _ in range(50):  # steps per episode
            if rng.random() < eps:
                a = rng.integers(0, N_ACTIONS)
            else:
                a = int(np.argmax(q[s, :]))
            s_next, r = _simulate_step(*_idx_to_state(s), ACTIONS[a], competitor_model, rng)
            ep_reward += r
            td = r + gamma * np.max(q[s_next, :]) - q[s, a]
            q[s, a] += 0.1 * td
            s = s_next
        eps *= epsilon_decay
        reward_trajectory.append(ep_reward)

    # Policy: argmax Q per state
    policy = np.argmax(q, axis=1)
    # Policy heatmap: state (flattened) vs action
    policy_onehot = np.eye(N_ACTIONS)[policy]
    # Aggregate by (price_gap, vol_idx) for 2D heatmap
    pg_dim = N_PRICE_GAP * N_VOLUME_IDX
    heatmap = np.zeros((pg_dim, N_ACTIONS))
    for idx in range(N_STATES):
        pg, vi, _, _ = _idx_to_state(idx)
        row = pg * N_VOLUME_IDX + vi
        heatmap[row, :] += policy_onehot[idx, :]
    heatmap = heatmap / (heatmap.sum(axis=1, keepdims=True) + 1e-9)

    # Value function: V(s) = max_a Q(s,a)
    v = np.max(q, axis=1)
    # Surface: (price_gap, volume_idx) -> mean V over time/comp
    surface = np.zeros((N_PRICE_GAP, N_VOLUME_IDX))
    for pg in range(N_PRICE_GAP):
        for vi in range(N_VOLUME_IDX):
            indices = [_state_to_idx(pg, vi, tb, ct) for tb in range(N_TIME_BUCKET) for ct in range(N_COMP_TREND)]
            surface[pg, vi] = np.mean(v[indices])

    q_dict = {f"s{i}_a{j}": float(q[i, j]) for i in range(min(50, N_STATES)) for j in range(N_ACTIONS)}

    return QLearningResult(
        q_table=q_dict,
        policy_heatmap={
            "z": heatmap.tolist(),
            "x": [f"{ACTIONS[a]}ct" for a in range(N_ACTIONS)],
            "y": [f"pg{pg}v{vi}" for pg in range(N_PRICE_GAP) for vi in range(N_VOLUME_IDX)],
        },
        reward_trajectory=reward_trajectory,
        value_function_surface={
            "z": surface.tolist(),
            "x": [f"vol{vi}" for vi in range(N_VOLUME_IDX)],
            "y": [f"gap{pg}" for pg in range(N_PRICE_GAP)],
        },
        n_episodes=episodes,
        final_avg_reward=float(np.mean(reward_trajectory[-50:])),
    )
