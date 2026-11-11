"""The physical system and the per-node production LP (thesis sections 5.1 and 4.1).

Two reservoirs in cascade: water released (or spilled) from the upper reservoir
flows into the lower one within the week. Each reservoir r has a storage bound
s_max, a weekly release bound x_max and an energy factor xi_r (MWh per Mm3 through
its plant). Revenue is taxed at tau = tau_C + tau_R (corporate + resource rent).

At a node (t, m) with incoming storage s_in = (s_U, s_L), known inflow (I_U, I_L)
and known price P the stage problem is the LP

    max  (1 - tau) P (xi_U x_U + xi_L x_L)  +  beta * sum_{m'} w_{m'} alpha_{m'}
    s.t. s'_U + x_U + sp_U                 = s_U + I_U            (upper water balance)
         s'_L + x_L + sp_L - x_U - sp_U    = s_L + I_L            (lower water balance)
         0 <= x <= x_max,  sp >= 0,  0 <= s' <= s_max
         alpha_{m'} <= theta_c + g_c . s'   for every cut c of successor m'

where alpha_{m'} is the (cut-approximated) value of continuing in successor state m'
with the storage s' we leave behind, and w_{m'} are the weights on the successors:
the physical probabilities delta for the risk-neutral policy, the worst-case
measure q* from gooddeal.py for the good-deal policy. beta is the weekly after-tax
discount factor.

The duals of the two water balances are the marginal values of stored water; they
are the slopes of the new cut that the backward pass adds (see sddp.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

_ENV = None


def gurobi_env() -> gp.Env:
    """One silent, single-threaded Gurobi environment for the whole process."""
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.setParam("Threads", 1)
        _ENV.start()
    return _ENV


@dataclass(frozen=True)
class Reservoir:
    name: str
    s_max: float          # usable storage, Mm3
    s_init: float         # storage at the start of the horizon, Mm3
    x_max: float          # maximum release through the plant, Mm3/week
    energy_factor: float  # MWh per Mm3


@dataclass(frozen=True)
class Cascade:
    upper: Reservoir
    lower: Reservoir

    @property
    def s0(self) -> np.ndarray:
        return np.array([self.upper.s_init, self.lower.s_init])


@dataclass(frozen=True)
class Economics:
    rf_annual: float = 0.03   # risk-free rate
    tau_C: float = 0.22       # corporate tax
    tau_R: float = 0.45       # resource-rent tax

    @property
    def tau(self) -> float:
        return self.tau_C + self.tau_R

    @property
    def beta(self) -> float:
        """Weekly after-tax discount factor 1 / (1 + (1 - tau_C) ((1 + rf)^(1/52) - 1))."""
        r_w = (1.0 - self.tau_C) * ((1.0 + self.rf_annual) ** (1.0 / 52.0) - 1.0)
        return 1.0 / (1.0 + r_w)


@dataclass
class NodeSolution:
    objective: float
    release: np.ndarray        # (2,) x_U, x_L
    spill: np.ndarray          # (2,)
    storage_out: np.ndarray    # (2,)
    water_values: np.ndarray   # (2,) duals of the balances = marginal value of s_in


class NodeLP:
    """The production LP of one node (t, m), kept in memory so that each solve only
    updates the incoming storage, the successor weights and any new cuts."""

    def __init__(self, cascade: Cascade, econ: Economics, price: float, inflow: tuple[float, float],
                 succ_idx: np.ndarray, terminal_value: tuple[float, float] = (0.0, 0.0)):
        up, lo = cascade.upper, cascade.lower
        tau, beta = econ.tau, econ.beta
        self.beta = beta
        self.inflow = inflow
        self.succ_idx = np.asarray(succ_idx, int)
        m = gp.Model(env=gurobi_env())
        m.ModelSense = GRB.MAXIMIZE
        self.xU = m.addVar(ub=up.x_max, obj=(1 - tau) * price * up.energy_factor, name="x_U")
        self.xL = m.addVar(ub=lo.x_max, obj=(1 - tau) * price * lo.energy_factor, name="x_L")
        self.spU = m.addVar(name="spill_U"); self.spL = m.addVar(name="spill_L")
        # end-of-week storage; at the last stage it is credited with a terminal water value
        self.sU = m.addVar(ub=up.s_max, obj=beta * terminal_value[0], name="s_U")
        self.sL = m.addVar(ub=lo.s_max, obj=beta * terminal_value[1], name="s_L")
        # one continuation variable per successor; ub = 0 until its first cut exists
        self.alpha = [m.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{int(s)}") for s in self.succ_idx]
        self.balU = m.addConstr(self.sU + self.xU + self.spU == inflow[0], name="balance_U")
        self.balL = m.addConstr(self.sL + self.xL + self.spL - self.xU - self.spU == inflow[1], name="balance_L")
        self.n_cuts = [0] * len(self.succ_idx)
        self.m = m

    def add_cuts(self, cuts_next: dict[int, list[tuple[float, float, float]]]) -> None:
        """Append the cuts of each successor that are not in the model yet.
        ``cuts_next`` maps successor state -> list of (theta, g_U, g_L)."""
        for i, s in enumerate(self.succ_idx):
            cuts = cuts_next.get(int(s), [])
            if self.n_cuts[i] == 0 and cuts:
                self.alpha[i].ub = GRB.INFINITY
            while self.n_cuts[i] < len(cuts):
                theta, gU, gL = cuts[self.n_cuts[i]]
                self.m.addConstr(self.alpha[i] <= theta + gU * self.sU + gL * self.sL)
                self.n_cuts[i] += 1

    def solve(self, s_in: np.ndarray, weights: np.ndarray, cuts_next: dict | None = None,
              cold: bool = False) -> NodeSolution:
        """Solve with incoming storage ``s_in`` and successor weights ``weights``.
        ``cold`` discards the warm-start basis (used in simulation for reproducibility)."""
        if cuts_next is not None:
            self.add_cuts(cuts_next)
        self.balU.RHS = s_in[0] + self.inflow[0]
        self.balL.RHS = s_in[1] + self.inflow[1]
        for a, w in zip(self.alpha, weights):
            a.Obj = self.beta * float(w)
        if cold:
            self.m.reset()
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"node LP ended with Gurobi status {self.m.Status}")
        return NodeSolution(objective=self.m.ObjVal,
                            release=np.array([self.xU.X, self.xL.X]),
                            spill=np.array([self.spU.X, self.spL.X]),
                            storage_out=np.array([self.sU.X, self.sL.X]),
                            water_values=np.array([self.balU.Pi, self.balL.Pi]))

    def dispose(self) -> None:
        self.m.dispose()


def terminal_water_values(lattice, cascade: Cascade, econ: Economics) -> tuple[float, float]:
    """EUR per Mm3 credited to the storage left at the end of the horizon: the expected
    end-of-horizon after-tax price times the energy the water can still produce (upper
    water also passes the lower plant). Without this the policy empties the reservoirs."""
    marg = lattice.marginals()[lattice.T - 1]
    Pbar = float(sum(marg[m] * lattice.price(lattice.T - 1, m) for m in range(lattice.K)))
    base = (1 - econ.tau) * Pbar
    xiU, xiL = cascade.upper.energy_factor, cascade.lower.energy_factor
    return base * (xiU + xiL), base * xiL
