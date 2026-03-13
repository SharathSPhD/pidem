"""PyTorch DQN for pricing with experience replay and target network."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.rl_tabular import (
    N_ACTIONS,
    N_STATES,
    ACTIONS,
    _idx_to_state,
    _state_to_idx,
    _simulate_step,
)


@dataclass
class DQNResult:
    training_curve: list[float]
    q_value_landscape: dict
    policy_comparison: dict
    n_episodes: int
    final_avg_reward: float


if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, state_dim: int, n_actions: int, hidden: list[int]):
            super().__init__()
            layers = []
            prev = state_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, n_actions))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def _state_to_vec(idx: int) -> np.ndarray:
    pg, vi, tb, ct = _idx_to_state(idx)
    vec = np.zeros(20)
    vec[pg] = 1.0
    vec[5 + vi] = 1.0
    vec[8 + tb] = 1.0
    vec[17 + ct] = 1.0
    return vec.astype(np.float32)


def train_dqn(
    hidden_layers: list[int] | None = None,
    units: int = 64,
    replay_size: int = 10000,
    target_freq: int = 10,
    env_difficulty: Literal["easy", "medium", "hard"] = "medium",
) -> DQNResult:
    """Train DQN with experience replay and target network."""
    if not TORCH_AVAILABLE:
        return DQNResult(
            training_curve=[],
            q_value_landscape={},
            policy_comparison={"error": "PyTorch not installed"},
            n_episodes=0,
            final_avg_reward=0.0,
        )

    hidden = hidden_layers or [units, units]
    state_dim = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, N_ACTIONS, hidden).to(device)
    target_net = DQN(state_dim, N_ACTIONS, hidden).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    competitor = {"easy": "static", "medium": "follow", "hard": "undercut"}[env_difficulty]
    rng = np.random.default_rng(42)
    replay: list[tuple] = []
    gamma = 0.95
    eps = 1.0
    eps_decay = 0.995
    training_curve = []
    batch_size = 64

    for ep in range(300):
        s = rng.integers(0, N_STATES)
        ep_reward = 0.0
        for _ in range(50):
            s_vec = _state_to_vec(s)
            if rng.random() < eps:
                a = rng.integers(0, N_ACTIONS)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).to(device))
                    a = int(q_vals.argmax(dim=1).item())
            s_next, r = _simulate_step(*_idx_to_state(s), ACTIONS[a], competitor, rng)
            replay.append((s_vec, a, r, _state_to_vec(s_next), s_next == s))
            if len(replay) > replay_size:
                replay.pop(0)
            ep_reward += r

            if len(replay) >= batch_size:
                batch = [replay[rng.integers(0, len(replay))] for _ in range(batch_size)]
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(device)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

                q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0]
                    target = rewards + gamma * q_next * (1 - dones)
                loss = nn.functional.mse_loss(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            s = s_next
        eps *= eps_decay
        if (ep + 1) % target_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        training_curve.append(ep_reward)

    # Q-value landscape
    policy_net.train(mode=False)
    q_landscape = []
    for pg in range(5):
        for vi in range(3):
            idx = _state_to_idx(pg, vi, 4, 1)
            vec = _state_to_vec(idx)
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy()[0]
            q_landscape.append(q_vals.tolist())

    # Policy comparison vs tabular
    from models.rl_tabular import train_qlearning

    tabular_result = train_qlearning(gamma=gamma, episodes=100, competitor_model=competitor)
    q_arr = np.zeros((min(45, N_STATES), N_ACTIONS))
    for i in range(min(45, N_STATES)):
        for j in range(N_ACTIONS):
            q_arr[i, j] = tabular_result.q_table.get(f"s{i}_a{j}", 0)
    tabular_policy = np.argmax(q_arr, axis=1)
    dqn_actions = []
    for i in range(min(45, N_STATES)):
        vec = _state_to_vec(i)
        with torch.no_grad():
            a = policy_net(torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)).argmax(1).item()
        dqn_actions.append(a)
    agreement = float(np.mean(np.array(tabular_policy) == np.array(dqn_actions)))

    return DQNResult(
        training_curve=training_curve,
        q_value_landscape={
            "z": q_landscape,
            "x": [f"{ACTIONS[a]}ct" for a in range(N_ACTIONS)],
            "y": [f"pg{pg}v{vi}" for pg in range(5) for vi in range(3)],
        },
        policy_comparison={
            "tabular_final_reward": tabular_result.final_avg_reward,
            "dqn_final_reward": float(np.mean(training_curve[-50:])) if training_curve else 0.0,
            "policy_agreement": agreement,
        },
        n_episodes=300,
        final_avg_reward=float(np.mean(training_curve[-50:])) if training_curve else 0.0,
    )
