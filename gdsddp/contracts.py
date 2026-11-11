"""Forward contracts on the lattice (thesis sections 5.3 and 4.4.1).

A forward contract j delivers one MWh in every week of its delivery window
[tau1_j, tau2_j] and settles against the weekly spot price. Its fair price at
week t is the expected average spot price over the remaining delivery, F_j(t, m),
which we compute on the lattice by a backward recursion over the transition
matrices.

The quantity the good-deal step needs is the "cash-augmented increment": the change
in the contract's value from week t to t+1, including the cash flow of the week when
the contract is delivering,

    dF_j(m, m') = F_j(t+1, m') - F_j(t, m) + P_t(m) / L_j * [t in delivery],   L_j = tau2 - tau1 + 1.

By construction the increment has expectation zero under the physical transition
probabilities, sum_m' delta(m, m') dF_j(m, m') = 0: the fair forward is a martingale.
The good-deal step keeps this property under the distorted measure q as well
(sum_m' q(m') dF_j(m, m') = 0), which is what prevents the risk adjustment from
pricing hedgeable price risk: see gooddeal.py.

Strikes: a contract sold at week 0 is struck at its fair price K_j = E[average spot
over delivery] under the lattice marginals (zero risk premium, an assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import Lattice


@dataclass(frozen=True)
class Contract:
    id: str
    start: int      # first delivery week, 1-based stage index
    end: int        # last delivery week, inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class ContractSystem:
    """Fair forward values F_j(t, m), strikes and increments for a list of contracts on a lattice."""

    def __init__(self, contracts: list[Contract], lattice: Lattice):
        self.contracts = list(contracts)
        self.lattice = lattice
        self.ids = [c.id for c in contracts]
        self.P = lattice.price_matrix()                              # (T, K)
        self.start0 = {c.id: c.start - 1 for c in contracts}          # 0-based
        self.end0 = {c.id: min(c.end - 1, lattice.T - 1) for c in contracts}
        self.L = {c.id: c.length for c in contracts}
        self.F = {c.id: self._fair_value(c.id) for c in contracts}    # (T+1, K)
        marg = lattice.marginals()
        self.strikes = {c.id: float((marg[self.start0[c.id]:self.end0[c.id] + 1]
                                     * self.P[self.start0[c.id]:self.end0[c.id] + 1]).sum(1).mean())
                        for c in contracts}
        self._cache: dict = {}

    def _fair_value(self, cid: str) -> np.ndarray:
        """F_j(t, m) = E[ sum of remaining delivery-week prices | state m at t ] / L_j."""
        T, K = self.lattice.T, self.lattice.K
        s0, e0, L = self.start0[cid], self.end0[cid], self.L[cid]
        C = np.zeros((T + 1, K))
        for t in range(e0, -1, -1):
            cont = self.lattice.transitions[t] @ C[t + 1] if t < T - 1 else np.zeros(K)
            C[t] = (self.P[t] if s0 <= t <= e0 else 0.0) + cont
        return C / L

    def active(self, t: int) -> list[str]:
        """Contracts not yet fully delivered at week t (0-based), i.e. those with a martingale constraint."""
        return [cid for cid in self.ids if t <= self.end0[cid]]

    def increments(self, t: int, m: int, succ_idx: np.ndarray):
        """(active ids, dF) with dF of shape (n_active, len(succ_idx)): the cash-augmented
        increments from node (t, m) to each successor. Memoised: they never change during SDDP."""
        key = (t, m)
        if key in self._cache:
            return self._cache[key]
        ids = self.active(t)
        dF = np.zeros((len(ids), len(succ_idx)))
        for r, cid in enumerate(ids):
            cash = self.P[t, m] / self.L[cid] if self.start0[cid] <= t <= self.end0[cid] else 0.0
            dF[r] = self.F[cid][t + 1, succ_idx] - self.F[cid][t, m] + cash
        self._cache[key] = (ids, dF)
        return ids, dF

    def check_martingale(self) -> float:
        """Largest |E_delta[dF]| over all nodes (should be ~1e-12): a sanity check."""
        worst = 0.0
        for t in range(self.lattice.T - 1):
            for m in range(self.lattice.K):
                idx, d = self.lattice.successors(t, m, 0.0)
                ids, dF = self.increments(t, m, idx)
                if len(ids):
                    worst = max(worst, float(np.abs(dF @ d).max()))
        self._cache.clear()        # the check used the full support; SDDP uses the thresholded one
        return worst
