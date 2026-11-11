"""Stochastic dual dynamic programming with the nested good-deal distortion (thesis section 5.6).

The problem is a multistage maximisation of expected after-tax revenue over T weeks.
The state is (week t, lattice state m, storage s). SDDP approximates the value of
water, V_t(m, s), by a set of linear CUTS per (t, m): because V is concave in s
every cut is an upper bound, V_t(m, s) <= theta + g . s, and the min over the cuts is
the approximation used in the production LP (hydro.py).

One iteration:
  FORWARD PASS   sample one path of lattice states and, solving the node LPs with the
                 current cuts, record the storage it leaves behind at every week.
  BACKWARD PASS  for t = T-1 down to 0 and for EVERY state m of week t:
                   1. (good-deal only) compute the continuation values V(m') of the
                      successors at the trial storage from the cuts of week t+1 and
                      find the worst-case measure q* (gooddeal.py);
                   2. solve the node LP with weights q* (or delta) and read the duals of
                      the water balances: the new cut has slope g = duals and intercept
                      theta = objective - g . s_trial.
                 The stage-0 objectives averaged over the (uniform) initial state give
                 the policy value: an UPPER bound on the true value (cuts over-estimate
                 a concave function), and it decreases towards the optimum.
We stop when this bound has moved less than ``tol`` (relative) over ``window``
consecutive iterations, or at ``iterations``.

Before the first iteration the backward pass is run along three constant storage
trajectories (10 %, 50 %, 90 % of capacity) so that every node has cuts covering the
range of storages the forward passes will visit.

Everything is single-threaded on purpose; a backward pass is one Python loop over
(T x K) nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import ContractSystem
from .gooddeal import worst_case_measure
from .hydro import Cascade, Economics, NodeLP, terminal_water_values
from .lattice import Lattice

NEG_PROB = 1e-6


@dataclass
class SDDPConfig:
    iterations: int = 30
    tol: float = 1e-4            # relative change of the bound over ``window`` iterations
    window: int = 10
    seed: int = 42
    seed_levels: tuple = (0.1, 0.5, 0.9)   # fractions of s_max for the initial backward sweeps
    h: float = 0.0               # annual Sharpe bound; 0 = risk neutral
    log_every: int = 5


@dataclass
class Policy:
    """A trained policy: the cuts per (t, m) and what was recorded while training."""
    cuts: list[list[list[tuple]]]              # cuts[t][m] = [(theta, gU, gL), ...]
    history: pd.DataFrame                      # per iteration: bound, forward value, fallbacks, seconds
    config: SDDPConfig
    name: str = ""
    q_star: dict = field(default_factory=dict)  # (t, m) -> (succ_idx, q) from the last backward pass
    budget_used: dict = field(default_factory=dict)

    @property
    def bound(self) -> float:
        return float(self.history["bound"].iloc[-1])

    def cuts_at(self, t: int) -> dict[int, list[tuple]]:
        """Cuts of all states of week t as the map NodeLP.add_cuts expects."""
        return {m: self.cuts[t][m] for m in range(len(self.cuts[t])) if self.cuts[t][m]}


def _envelope(cuts_t: list[list[tuple]], s: np.ndarray) -> np.ndarray:
    """Cut-implied value of every state at week t for storage s: min over its cuts."""
    out = np.zeros(len(cuts_t))
    for m, lst in enumerate(cuts_t):
        if lst:
            out[m] = min(th + gU * s[0] + gL * s[1] for th, gU, gL in lst)
    return out


def train(lattice: Lattice, cascade: Cascade, econ: Economics, contracts: ContractSystem | None,
          config: SDDPConfig, name: str = "", log=print) -> Policy:
    """Run SDDP and return the trained Policy. ``contracts`` may be None (no forwards)."""
    T, K = lattice.T, lattice.K
    h = float(config.h)
    good_deal = h > 0.0
    rng = np.random.default_rng(config.seed)
    P = lattice.price_matrix()
    inflow = {(t, m): lattice.inflow(t, m) for t in range(T) for m in range(K)}
    succ = [[lattice.successors(t, m, NEG_PROB) for m in range(K)] for t in range(T - 1)]
    term = terminal_water_values(lattice, cascade, econ)
    cuts: list[list[list[tuple]]] = [[[] for _ in range(K)] for _ in range(T)]
    models: dict[tuple[int, int], NodeLP] = {}
    s0 = cascade.s0
    s_max = np.array([cascade.upper.s_max, cascade.lower.s_max])
    q_star: dict = {}; budget: dict = {}          # filled by the last (recorded) backward pass

    def node(t: int, m: int) -> NodeLP:
        if (t, m) not in models:
            sidx = succ[t][m][0] if t < T - 1 else np.zeros(0, int)
            models[(t, m)] = NodeLP(cascade, econ, P[t, m], inflow[(t, m)], sidx,
                                    terminal_value=term if t == T - 1 else (0.0, 0.0))
        return models[(t, m)]

    def backward(s_trial: np.ndarray, record: bool = False):
        """One backward sweep along the trial storages (T+1, 2). Returns the stage-0 bound."""
        Q0 = np.zeros(K); n_fb = 0
        for t in range(T - 1, -1, -1):
            V_next = _envelope(cuts[t + 1], s_trial[t + 1]) if (good_deal and t < T - 1) else None
            cuts_next = {m: cuts[t + 1][m] for m in range(K)} if t < T - 1 else None
            for m in range(K):
                if t < T - 1:
                    sidx, delta = succ[t][m]
                    if good_deal:
                        ids, dF = contracts.increments(t, m, sidx) if contracts is not None else ([], np.zeros((0, len(sidx))))
                        gd = worst_case_measure(V_next[sidx], delta, dF, h)
                        weights = gd.q; n_fb += gd.fell_back
                        if record:
                            q_star[(t, m)] = (sidx, gd.q); budget[(t, m)] = gd.budget_used
                    else:
                        weights = delta
                else:
                    weights = np.zeros(0)
                sol = node(t, m).solve(s_trial[t], weights, cuts_next)
                g = sol.water_values
                cuts[t][m].append((sol.objective - g @ s_trial[t], g[0], g[1]))
                if t == 0:
                    Q0[m] = sol.objective
        return float(Q0.mean()), n_fb

    # ---- initial sweeps so every node has cuts over the storage range
    t0 = time.time()
    for frac in config.seed_levels:
        backward(np.tile(frac * s_max, (T + 1, 1)))
    log(f"[{name}] seeded cuts at {[f'{f:.0%}' for f in config.seed_levels]} of capacity ({time.time() - t0:.0f} s)")

    rows = []
    for it in range(1, config.iterations + 1):
        tic = time.time()
        # forward pass: one sampled path, record the storage the current policy leaves behind
        s_trial = np.zeros((T + 1, 2)); s_trial[0] = s0
        m = int(rng.integers(K)); fwd_value = 0.0; disc = 1.0
        for t in range(T):
            sidx, delta = succ[t][m] if t < T - 1 else (np.zeros(0, int), np.zeros(0))
            sol = node(t, m).solve(s_trial[t], delta, {s: cuts[t + 1][s] for s in range(K)} if t < T - 1 else None)
            s_trial[t + 1] = sol.storage_out
            fwd_value += disc * (1 - econ.tau) * P[t, m] * (cascade.upper.energy_factor * sol.release[0]
                                                            + cascade.lower.energy_factor * sol.release[1])
            disc *= econ.beta
            if t < T - 1:
                m = int(rng.choice(sidx, p=delta))
        fwd_value += disc * (term[0] * s_trial[T][0] + term[1] * s_trial[T][1])   # terminal water value, discounted
        # backward pass along it
        last = it == config.iterations
        bound, n_fb = backward(s_trial, record=last)
        rows.append({"iteration": it, "bound": bound, "forward_value": fwd_value, "fallbacks": n_fb,
                     "seconds": time.time() - tic, "n_cuts": sum(len(c) for ct in cuts for c in ct)})
        if it % config.log_every == 0 or it == 1:
            log(f"[{name}] it {it:3d}  bound {bound / 1e6:9.3f} M  forward {fwd_value / 1e6:8.3f} M  "
                f"{rows[-1]['seconds']:.1f} s/it" + (f"  q fallbacks {n_fb}" if good_deal else ""))
        # convergence: relative movement of the bound over the window
        if it >= config.window:
            b = [r["bound"] for r in rows[-config.window:]]
            if (max(b) - min(b)) / abs(b[-1]) < config.tol and not last:
                log(f"[{name}] converged at iteration {it}: running one final recorded pass")
                bound, n_fb = backward(s_trial, record=True)
                rows[-1]["bound"] = bound
                break
    for nm in models.values():
        nm.dispose()
    return Policy(cuts=cuts, history=pd.DataFrame(rows), config=config, name=name, q_star=q_star, budget_used=budget)
