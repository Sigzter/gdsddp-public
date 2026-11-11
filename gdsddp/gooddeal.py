"""The good-deal step: the worst-case successor measure q* at one node (thesis section 4.4).

Idea. A risk-neutral producer values next week's uncertain continuation value V(m')
by its expectation under the physical probabilities delta. A good-deal bound instead
takes the LOWEST expectation over all probability measures q that
  (a) are "not too far" from delta — the chi-squared distance sum_m' q(m')^2/delta(m')
      is bounded by 1 + h_w^2, where h_w = h / sqrt(52) is the weekly Sharpe-ratio bound
      (Cochrane & Saa-Requejo 2000: no measure in the set can turn any payoff into a
      Sharpe ratio above h per year), and
  (b) price every traded forward correctly: the fair forward must stay a martingale,
      sum_m' q(m') dF_j(m') = 0 for every contract j that is still alive.

The optimisation at node (t, m) with successors m' (physical probabilities delta,
continuation values V and increments dF) is the convex QCQP

    min_q  sum_m' q(m') V(m')
    s.t.   sum_m' q(m') = 1,  q >= 0
           sum_m' q(m') dF_j(m') = 0            for every active contract j
           sum_m' q(m')^2 / delta(m') <= 1 + h_w^2

which Gurobi solves directly. The martingale rows mean that the adversary cannot
move probability along directions that the forwards span (price risk is hedgeable
and therefore priced by the market, not by the producer's aversion); the remaining
budget is spent on the directions the contracts do not reach — inflow. This is the
separation principle of the thesis.

The resulting q* replaces delta as the successor weights in the production LP of
the node (hydro.NodeLP), so the risk adjustment is nested into every SDDP cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .hydro import gurobi_env


def weekly_ball_radius(h: float) -> float:
    """Right-hand side of the chi-squared constraint: 1 + (h / sqrt(52))^2."""
    return 1.0 + (h / np.sqrt(52.0)) ** 2


@dataclass
class GoodDealSolution:
    q: np.ndarray           # worst-case measure on the successors
    value: float            # sum q V  (<= sum delta V)
    budget_used: float      # sum q^2/delta - 1, relative to h_w^2 (1.0 = the ball binds)
    fell_back: bool         # True if Gurobi failed and q = delta was returned
    status: str


def worst_case_measure(V: np.ndarray, delta: np.ndarray, dF: np.ndarray, h: float,
                       flat_tol: float = 1e-6) -> GoodDealSolution:
    """Solve the node QCQP above with Gurobi and return q*.

    V      (n,) continuation values of the successors
    delta  (n,) physical probabilities (sum to 1)
    dF     (n_contracts, n) martingale increments; an empty array if no contract is alive
    h      annual Sharpe bound; h = 0 returns delta
    """
    V = np.asarray(V, float); delta = np.asarray(delta, float)
    delta = delta / delta.sum()
    n = len(V)
    dF = np.asarray(dF, float).reshape(-1, n) if np.size(dF) else np.zeros((0, n))
    # rows with no payoff variation across the successors carry no information: drop them
    if len(dF):
        dF = dF[np.abs(dF).max(1) > 1e-9]
    hw2 = (h / np.sqrt(52.0)) ** 2
    if h <= 0.0 or n == 1 or V.max() - V.min() < flat_tol:
        return GoodDealSolution(delta.copy(), float(delta @ V), 0.0, False, "trivial")

    model = gp.Model(env=gurobi_env())
    q = model.addVars(n, lb=0.0, name="q")
    model.addConstr(gp.quicksum(q[i] for i in range(n)) == 1.0, name="probability")
    for j in range(len(dF)):
        model.addConstr(gp.quicksum(dF[j, i] * q[i] for i in range(n)) == 0.0, name=f"martingale_{j}")
    model.addConstr(gp.quicksum(q[i] * q[i] / delta[i] for i in range(n)) <= 1.0 + hw2, name="good_deal_ball")
    model.setObjective(gp.quicksum(V[i] * q[i] for i in range(n)), GRB.MINIMIZE)
    model.Params.BarConvTol = 1e-10
    model.Params.BarQCPConvTol = 1e-9
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        model.dispose()
        return GoodDealSolution(delta.copy(), float(delta @ V), 0.0, True, f"gurobi status {model.Status}")
    qs = np.array([q[i].X for i in range(n)])
    model.dispose()
    qs = np.clip(qs, 0.0, None); qs /= qs.sum()
    used = (float(np.sum(qs * qs / delta)) - 1.0) / hw2
    return GoodDealSolution(qs, float(qs @ V), used, False, "optimal")
