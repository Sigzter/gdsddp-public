"""Forward simulation of a trained policy and revenue statistics (thesis sections 5.5, 6.5).

Simulation always uses the PHYSICAL transition probabilities: the good-deal
measure is a valuation device inside the cuts, not a forecast. Each path starts
at a uniformly drawn lattice state with the configured initial storage and, week
by week, solves the node LP with the trained cuts (weights = delta) to decide
release and storage, then moves to a successor state drawn from the lattice.
"""

from __future__ import annotations

import numpy as np

from .hydro import Cascade, Economics, NodeLP, terminal_water_values
from .lattice import Lattice
from .sddp import NEG_PROB, Policy


def simulate(policy: Policy, lattice: Lattice, cascade: Cascade, econ: Economics,
             n_paths: int = 500, seed: int = 7, log=print) -> dict[str, np.ndarray]:
    """Simulate ``n_paths`` weeks-long trajectories under the policy. Returns a dict:
        state (N, T) int, price (N, T), inflow (N, T, 2), storage (N, T+1, 2),
        release (N, T, 2), spill (N, T, 2), revenue (N, T) after-tax EUR,
        terminal (N,) value credited to the final storage, total (N,) = discounted sum."""
    T, K = lattice.T, lattice.K
    rng = np.random.default_rng(seed)
    P = lattice.price_matrix()
    term = terminal_water_values(lattice, cascade, econ)
    state = np.zeros((n_paths, T), int); state[:, 0] = rng.integers(K, size=n_paths)
    U = rng.random((n_paths, T))                         # uniforms for the state transitions
    storage = np.zeros((n_paths, T + 1, 2)); storage[:, 0] = cascade.s0
    release = np.zeros((n_paths, T, 2)); spill = np.zeros((n_paths, T, 2)); revenue = np.zeros((n_paths, T))
    inflow = np.zeros((n_paths, T, 2)); price = np.zeros((n_paths, T))
    xi = np.array([cascade.upper.energy_factor, cascade.lower.energy_factor])
    for t in range(T):
        cuts_next = policy.cuts_at(t + 1) if t < T - 1 else None
        for m in np.unique(state[:, t]):
            paths = np.nonzero(state[:, t] == m)[0]
            sidx, delta = lattice.successors(t, m, NEG_PROB) if t < T - 1 else (np.zeros(0, int), np.zeros(0))
            lp = NodeLP(cascade, econ, P[t, m], lattice.inflow(t, m), sidx,
                        terminal_value=term if t == T - 1 else (0.0, 0.0))
            if cuts_next is not None:
                lp.add_cuts(cuts_next)
            cdf = np.cumsum(delta) if len(sidx) else None
            for p in paths:
                sol = lp.solve(storage[p, t], delta, cold=True)
                storage[p, t + 1] = sol.storage_out; release[p, t] = sol.release; spill[p, t] = sol.spill
                revenue[p, t] = (1 - econ.tau) * P[t, m] * (xi @ sol.release)
                price[p, t] = P[t, m]; inflow[p, t] = lattice.inflow(t, m)
                if cdf is not None:
                    state[p, t + 1] = sidx[min(int(np.searchsorted(cdf, U[p, t] * cdf[-1])), len(sidx) - 1)]
            lp.dispose()
        if t % 13 == 0:
            log(f"  simulated week {t}/{T}")
    disc = econ.beta ** np.arange(T)
    terminal = storage[:, T] @ np.array(term) * econ.beta ** T
    total = (revenue * disc[None, :]).sum(1) + terminal
    return {"state": state, "price": price, "inflow": inflow, "storage": storage, "release": release,
            "spill": spill, "revenue": revenue, "terminal": terminal, "total": total}


def revenue_summary(total: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean, standard deviation, 5 % quantile and CVaR (mean of the worst alpha share) of the total."""
    x = np.asarray(total, float)
    q = np.quantile(x, alpha)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "p5": float(q),
            "cvar5": float(x[x <= q].mean()), "p95": float(np.quantile(x, 0.95)), "n_paths": int(x.size)}
