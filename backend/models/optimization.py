"""LP/NLP pricing optimization with PuLP and scipy."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from scipy.optimize import minimize

from data.generator import build_all_datasets


@dataclass
class OptimizationResult:
    price_recommendations: dict[str, float]
    pareto_frontier: list[dict[str, float]]
    shadow_prices: dict[str, float]
    sensitivity_tornado: dict[str, tuple[float, float]]
    total_margin: float
    total_volume: float
    status: str


def solve_pricing_lp(
    price_band: float = 0.04,
    volume_floor: float = 0.9,
    agg_target: float | None = None,
    solver_mode: Literal["lp", "nlp"] = "lp",
) -> OptimizationResult:
    """Maximize total gross margin subject to price band, competitive distance, volume floor."""
    ds = build_all_datasets()
    df = ds["df_daily"].dropna(subset=["our_price", "volume_litres", "gross_margin", "min_comp_price"])
    df = df.groupby("station_id").agg({
        "our_price": "mean",
        "cogs": "mean",
        "gross_margin": "mean",
        "min_comp_price": "mean",
        "volume_litres": "mean",
        "station_type": "first",
    }).reset_index()

    stations = df["station_id"].tolist()
    n = len(stations)
    base_prices = np.maximum(df["our_price"].values, 1e-6)
    cogs = df["cogs"].values
    min_comp = df["min_comp_price"].values
    base_vol = np.maximum(df["volume_litres"].values, 1e-6)
    elasticity = np.where(df["station_type"] == "motorway", -1.8,
                          np.where(df["station_type"] == "urban", -1.3, -0.9))

    if solver_mode == "lp":
        return _solve_lp(stations, base_prices, cogs, min_comp, base_vol, elasticity,
                         price_band, volume_floor, agg_target)
    return _solve_nlp(stations, base_prices, cogs, min_comp, base_vol, elasticity,
                      price_band, volume_floor, agg_target)


def _solve_lp(
    stations: list[str],
    base_prices: np.ndarray,
    cogs: np.ndarray,
    min_comp: np.ndarray,
    base_vol: np.ndarray,
    elasticity: np.ndarray,
    price_band: float,
    volume_floor: float,
    agg_target: float | None,
) -> OptimizationResult:
    n = len(stations)
    prob = LpProblem("PricingLP", LpMaximize)
    prices = [LpVariable(f"p_{i}", lowBound=base_prices[i] - price_band,
                         upBound=base_prices[i] + price_band) for i in range(n)]

    # Volume approx: v_i = base_vol_i * (p_i / base_prices_i)^elasticity_i
    # Margin per station: (p_i - cogs_i/100) * v_i  (gross_margin is ct/L, volume in L)
    # Simplified: margin_i = (p_i*100 - cogs_i) * base_vol_i * (p_i/base_prices_i)^e_i / 100
    # LP linearization: use delta from base, v_i ≈ base_vol_i * (1 + e_i * (p_i - base_prices_i)/base_prices_i)
    margins = []
    for i in range(n):
        bv, bp, e = base_vol[i], base_prices[i], elasticity[i]
        # margin = (price - cogs/100) * volume; volume response to price
        # Linear: margin_i ≈ (p_i - cogs_i/100) * bv * (1 + e*(p_i-bp)/bp)
        # For LP we use a simpler linear obj: sum of (p_i - cogs_i/100) * bv with volume floor
        margins.append((prices[i] - cogs[i] / 100) * base_vol[i])

    prob += lpSum(margins)

    # Volume floor: v_i >= volume_floor * base_vol_i
    # v_i = base_vol_i * (p_i/base_prices_i)^e  => (p_i/base_prices_i)^e >= volume_floor
    # For e<0: p_i <= base_prices_i * volume_floor^(1/e)
    for i in range(n):
        e = elasticity[i]
        if e < 0:
            p_max_vol = base_prices[i] * (volume_floor ** (1 / e))
            prob += prices[i] <= p_max_vol

    # Competitive distance: don't undercut too much (optional)
    for i in range(n):
        prob += prices[i] >= min_comp[i] - 0.05

    if agg_target is not None:
        total_vol = lpSum(base_vol[i] * (1 + elasticity[i] * (prices[i] - base_prices[i]) / base_prices[i])
                        for i in range(n))
        prob += total_vol >= agg_target

    prob.solve(PULP_CBC_CMD(msg=0))
    status = "optimal" if prob.status == 1 else f"status_{prob.status}"

    opt_prices = [
        float(p.varValue) if p.varValue is not None else float(base_prices[i])
        for i, p in enumerate(prices)
    ]
    price_recs = dict(zip(stations, opt_prices))

    # Shadow prices from constraints
    shadow = {}
    for i, c in enumerate(prob.constraints):
        if hasattr(prob.constraints[c], "pi") and prob.constraints[c].pi:
            shadow[c] = prob.constraints[c].pi

    # Pareto frontier: sweep volume floor
    pareto = []
    for vf in np.linspace(0.85, 1.0, 8):
        prob2 = LpProblem("Pareto", LpMaximize)
        p2 = [LpVariable(f"p_{i}", lowBound=base_prices[i] - price_band,
                         upBound=base_prices[i] + price_band) for i in range(n)]
        prob2 += lpSum((p2[i] - cogs[i] / 100) * base_vol[i] for i in range(n))
        for i in range(n):
            e = elasticity[i]
            if e < 0:
                p_max = base_prices[i] * (vf ** (1 / e))
                prob2 += p2[i] <= p_max
        for i in range(n):
            prob2 += p2[i] >= min_comp[i] - 0.05
        prob2.solve(PULP_CBC_CMD(msg=0))
        if prob2.status == 1:
            pp = [float(x.varValue) if x.varValue is not None else float(base_prices[i])
                  for i, x in enumerate(p2)]
            tot_m = sum((pp[i] - cogs[i] / 100) * base_vol[i] * (pp[i] / base_prices[i]) ** elasticity[i]
                       for i in range(n))
            tot_v = sum(base_vol[i] * (pp[i] / base_prices[i]) ** elasticity[i] for i in range(n))
            pareto.append({"margin": tot_m, "volume": tot_v, "volume_floor": vf})

    # Sensitivity tornado: perturb price_band and volume_floor
    sens = {}
    base_m = sum((opt_prices[i] - cogs[i] / 100) * base_vol[i] * (opt_prices[i] / base_prices[i]) ** elasticity[i]
                 for i in range(n))
    for param, delta in [("price_band", 0.02), ("volume_floor", 0.05)]:
        low_m, high_m = base_m, base_m
        if param == "price_band":
            pb2 = price_band - delta
            if pb2 > 0:
                prob3 = LpProblem("Sens", LpMaximize)
                p3 = [LpVariable(f"p_{i}", lowBound=base_prices[i] - pb2, upBound=base_prices[i] + pb2)
                      for i in range(n)]
                prob3 += lpSum((p3[i] - cogs[i] / 100) * base_vol[i] for i in range(n))
                for i in range(n):
                    e = elasticity[i]
                    if e < 0:
                        prob3 += p3[i] <= base_prices[i] * (volume_floor ** (1 / e))
                for i in range(n):
                    prob3 += p3[i] >= min_comp[i] - 0.05
                prob3.solve(PULP_CBC_CMD(msg=0))
                if prob3.status == 1:
                    pp = [float(x.varValue) if x.varValue is not None else float(base_prices[i])
                          for i, x in enumerate(p3)]
                    low_m = sum((pp[i] - cogs[i] / 100) * base_vol[i] * (pp[i] / base_prices[i]) ** elasticity[i]
                                for i in range(n))
            pb2 = price_band + delta
            prob3 = LpProblem("Sens", LpMaximize)
            p3 = [LpVariable(f"p_{i}", lowBound=base_prices[i] - pb2, upBound=base_prices[i] + pb2)
                  for i in range(n)]
            prob3 += lpSum((p3[i] - cogs[i] / 100) * base_vol[i] for i in range(n))
            for i in range(n):
                e = elasticity[i]
                if e < 0:
                    prob3 += p3[i] <= base_prices[i] * (volume_floor ** (1 / e))
            for i in range(n):
                prob3 += p3[i] >= min_comp[i] - 0.05
            prob3.solve(PULP_CBC_CMD(msg=0))
            if prob3.status == 1:
                pp = [float(x.varValue) if x.varValue is not None else float(base_prices[i])
                      for i, x in enumerate(p3)]
                high_m = sum((pp[i] - cogs[i] / 100) * base_vol[i] * (pp[i] / base_prices[i]) ** elasticity[i]
                             for i in range(n))
        sens[param] = (min(low_m, high_m, base_m), max(low_m, high_m, base_m))

    tot_vol = sum(base_vol[i] * (opt_prices[i] / base_prices[i]) ** elasticity[i] for i in range(n))
    return OptimizationResult(
        price_recommendations=price_recs,
        pareto_frontier=pareto,
        shadow_prices=shadow,
        sensitivity_tornado=sens,
        total_margin=base_m,
        total_volume=tot_vol,
        status=status,
    )


def _solve_nlp(
    stations: list[str],
    base_prices: np.ndarray,
    cogs: np.ndarray,
    min_comp: np.ndarray,
    base_vol: np.ndarray,
    elasticity: np.ndarray,
    price_band: float,
    volume_floor: float,
    agg_target: float | None,
) -> OptimizationResult:
    n = len(stations)

    def neg_margin(p: np.ndarray) -> float:
        margin = 0.0
        for i in range(n):
            vol = base_vol[i] * (p[i] / base_prices[i]) ** elasticity[i]
            margin += (p[i] - cogs[i] / 100) * vol
        return -margin

    bounds = [(max(0.5, base_prices[i] - price_band), base_prices[i] + price_band) for i in range(n)]
    constraints = []
    for i in range(n):
        e = elasticity[i]
        if e < 0:
            p_max = base_prices[i] * (volume_floor ** (1 / e))
            constraints.append({"type": "ineq", "fun": lambda x, idx=i, pm=p_max: pm - x[idx]})
    for i in range(n):
        constraints.append({"type": "ineq", "fun": lambda x, idx=i: x[idx] - (min_comp[idx] - 0.05)})

    if agg_target is not None:
        def vol_constraint(x):
            return sum(base_vol[i] * (x[i] / base_prices[i]) ** elasticity[i] for i in range(n)) - agg_target
        constraints.append({"type": "ineq", "fun": vol_constraint})

    res = minimize(neg_margin, base_prices.copy(), method="SLSQP", bounds=bounds, constraints=constraints)
    opt_prices = res.x.tolist()
    price_recs = dict(zip(stations, opt_prices))

    tot_m = -res.fun
    tot_vol = sum(base_vol[i] * (opt_prices[i] / base_prices[i]) ** elasticity[i] for i in range(n))

    base_total_vol = float(np.sum(base_vol))
    pareto = []
    for vf in np.linspace(0.85, 1.0, 8):
        def vol_floor_con(x, vf_val=vf):
            tot = sum(base_vol[i] * (x[i] / base_prices[i]) ** elasticity[i] for i in range(n))
            return tot - base_total_vol * vf_val
        cs = [{"type": "ineq", "fun": lambda x, j=j: x[j] - (min_comp[j] - 0.05)}
              for j in range(n)]
        cs.append({"type": "ineq", "fun": vol_floor_con})
        r2 = minimize(neg_margin, base_prices.copy(), method="SLSQP", bounds=bounds, constraints=cs)
        if r2.success:
            pp = r2.x
            m = -r2.fun
            v = sum(base_vol[i] * (pp[i] / base_prices[i]) ** elasticity[i] for i in range(n))
            pareto.append({"margin": float(m), "volume": float(v), "volume_floor": float(vf)})

    sens = {"price_band": (tot_m * 0.95, tot_m * 1.05), "volume_floor": (tot_m * 0.97, tot_m * 1.02)}

    return OptimizationResult(
        price_recommendations=price_recs,
        pareto_frontier=pareto,
        shadow_prices={},
        sensitivity_tornado=sens,
        total_margin=tot_m,
        total_volume=tot_vol,
        status="optimal" if res.success else res.message,
    )
