"""Multi-armed bandit simulation for price experimentation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go

ARMS = np.array([-4, -2, 0, 2, 4])  # price deltas in ct


@dataclass
class BanditResult:
    cumulative_regret: list[float]
    arm_selection_heatmap: dict
    posterior_evolution: dict
    revenue_comparison: dict
    algorithm: str


def simulate_bandit(
    algorithm: Literal["epsilon_greedy", "ucb1", "thompson_sampling", "fdsw_thompson"],
    n_hours: int = 168,
    true_elasticity: float = -1.2,
    non_stationary: bool = False,
    prior_strength: float = 1.0,
) -> BanditResult:
    """Simulate bandit with arms as price levels {-4ct, -2ct, parity, +2ct, +4ct}."""
    rng = np.random.default_rng(42)
    n_arms = len(ARMS)
    base_price = 1.65
    base_volume = 30000

    # True expected revenue per arm (ct delta -> price -> volume response -> revenue)
    def revenue(arm_idx: int, t: int) -> float:
        delta_ct = ARMS[arm_idx]
        price = base_price + delta_ct / 100
        if non_stationary:
            drift = 0.02 * np.sin(2 * np.pi * t / (n_hours / 2))
            eff_el = true_elasticity + drift
        else:
            eff_el = true_elasticity
        vol = base_volume * (price / base_price) ** eff_el
        margin_ct = (price * 100) - 145  # cogs ~1.45
        return vol * margin_ct / 100

    true_means = np.array([revenue(a, 0) for a in range(n_arms)])
    best_arm = np.argmax(true_means)
    best_revenue = true_means[best_arm]

    if algorithm == "epsilon_greedy":
        cum_regret, arm_counts, rewards = _epsilon_greedy(
            n_hours, n_arms, revenue, best_revenue, rng
        )
    elif algorithm == "ucb1":
        cum_regret, arm_counts, rewards = _ucb1(
            n_hours, n_arms, revenue, best_revenue, rng
        )
    elif algorithm == "thompson_sampling":
        cum_regret, arm_counts, rewards = _thompson_sampling(
            n_hours, n_arms, revenue, best_revenue, prior_strength, rng
        )
    else:  # fdsw_thompson
        cum_regret, arm_counts, rewards = _fdsw_thompson(
            n_hours, n_arms, revenue, best_revenue, prior_strength, rng
        )

    # Arm selection heatmap (arms x time buckets)
    n_buckets = min(24, n_hours // 7)
    bucket_size = max(1, n_hours // n_buckets)
    heatmap_data = np.zeros((n_arms, n_buckets))
    for t in range(min(n_hours, len(arm_counts))):
        b = min(t // bucket_size, n_buckets - 1)
        heatmap_data[arm_counts[t], b] += 1
    heatmap_data = heatmap_data / (heatmap_data.sum(axis=0, keepdims=True) + 1e-9)

    # Posterior evolution (for Thompson: alpha, beta per arm at final)
    posterior = {"arm_labels": [f"{ARMS[a]}ct" for a in range(n_arms)]}
    if algorithm in ("thompson_sampling", "fdsw_thompson"):
        posterior["alpha"] = [float(prior_strength + sum(1 for t in range(n_hours) if arm_counts[t] == a and rewards[t] > np.median(rewards))) for a in range(n_arms)]
        posterior["beta"] = [float(prior_strength + sum(1 for t in range(n_hours) if arm_counts[t] == a and rewards[t] <= np.median(rewards))) for a in range(n_arms)]

    # Revenue comparison
    total_actual = sum(rewards)
    total_optimal = best_revenue * n_hours
    revenue_comparison = {
        "total_actual": total_actual,
        "total_optimal": total_optimal,
        "regret_ratio": cum_regret[-1] / (total_optimal + 1e-9),
        "best_arm": int(best_arm),
        "arm_labels": [f"{ARMS[a]}ct" for a in range(n_arms)],
    }

    return BanditResult(
        cumulative_regret=cum_regret,
        arm_selection_heatmap={
            "z": heatmap_data.tolist(),
            "x": [f"T{b}" for b in range(n_buckets)],
            "y": [f"{ARMS[a]}ct" for a in range(n_arms)],
        },
        posterior_evolution=posterior,
        revenue_comparison=revenue_comparison,
        algorithm=algorithm,
    )


def _epsilon_greedy(
    n_hours: int, n_arms: int, revenue_fn, best_revenue: float, rng
) -> tuple[list[float], list[int], list[float]]:
    eps = 0.1
    counts = np.zeros(n_arms)
    sums = np.zeros(n_arms)
    arm_counts = []
    rewards = []
    cum_regret = []

    for t in range(n_hours):
        if rng.random() < eps:
            arm = rng.integers(0, n_arms)
        else:
            arm = np.argmax(sums / (counts + 1e-9))
        rev = revenue_fn(arm, t) * (1 + rng.normal(0, 0.05))
        counts[arm] += 1
        sums[arm] += rev
        arm_counts.append(arm)
        rewards.append(rev)
        cum_regret.append((t + 1) * best_revenue - sum(rewards))
    return cum_regret, arm_counts, rewards


def _ucb1(
    n_hours: int, n_arms: int, revenue_fn, best_revenue: float, rng
) -> tuple[list[float], list[int], list[float]]:
    counts = np.zeros(n_arms)
    sums = np.zeros(n_arms)
    arm_counts = []
    rewards = []
    cum_regret = []

    for t in range(n_hours):
        if t < n_arms:
            arm = t
        else:
            ucb = sums / (counts + 1e-9) + np.sqrt(2 * np.log(t + 1) / (counts + 1e-9))
            arm = np.argmax(ucb)
        rev = revenue_fn(arm, t) * (1 + rng.normal(0, 0.05))
        counts[arm] += 1
        sums[arm] += rev
        arm_counts.append(arm)
        rewards.append(rev)
        cum_regret.append((t + 1) * best_revenue - sum(rewards))
    return cum_regret, arm_counts, rewards


def _thompson_sampling(
    n_hours: int, n_arms: int, revenue_fn, best_revenue: float,
    prior_strength: float, rng
) -> tuple[list[float], list[int], list[float]]:
    alpha = np.ones(n_arms) * prior_strength
    beta = np.ones(n_arms) * prior_strength
    arm_counts = []
    rewards = []

    for t in range(n_hours):
        samples = rng.beta(alpha, beta)
        arm = np.argmax(samples)
        rev = revenue_fn(arm, t) * (1 + rng.normal(0, 0.05))
        # Bernoulli-style update: success if above median
        median_rev = np.median([revenue_fn(a, t) for a in range(n_arms)])
        if rev > median_rev:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        arm_counts.append(arm)
        rewards.append(rev)
    cum_regret = [(t + 1) * best_revenue - sum(rewards[: t + 1]) for t in range(n_hours)]
    return cum_regret, arm_counts, rewards


def _fdsw_thompson(
    n_hours: int, n_arms: int, revenue_fn, best_revenue: float,
    prior_strength: float, rng
) -> tuple[list[float], list[int], list[float]]:
    """FDSW: Finite-time Discounted Thompson Sampling with sliding window."""
    window = min(50, n_hours // 4)
    alpha = np.ones(n_arms) * prior_strength
    beta = np.ones(n_arms) * prior_strength
    history: list[tuple[int, float]] = []
    arm_counts = []
    rewards = []

    for t in range(n_hours):
        samples = rng.beta(alpha, beta)
        arm = np.argmax(samples)
        rev = revenue_fn(arm, t) * (1 + rng.normal(0, 0.05))
        history.append((arm, rev))
        if len(history) > window:
            old_arm, old_rev = history.pop(0)
            median_old = np.median([revenue_fn(a, t - window) for a in range(n_arms)])
            if old_rev > median_old:
                alpha[old_arm] = max(prior_strength, alpha[old_arm] - 1)
            else:
                beta[old_arm] = max(prior_strength, beta[old_arm] - 1)
        median_rev = np.median([revenue_fn(a, t) for a in range(n_arms)])
        if rev > median_rev:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        arm_counts.append(arm)
        rewards.append(rev)
    cum_regret = [(t + 1) * best_revenue - sum(rewards[: t + 1]) for t in range(n_hours)]
    return cum_regret, arm_counts, rewards
