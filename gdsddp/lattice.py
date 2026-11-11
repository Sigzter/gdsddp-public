"""Quantisation: from simulated paths to a Markov lattice (thesis section 5.2.3).

SDDP needs the uncertainty in a finite form. We take the simulated paths of all
state coordinates (the two price factors and the two inflows) and, for every
week t, run ONE K-means over all coordinates jointly (scikit-learn), so each of
the K lattice states of that week is a point in the joint space of the
coordinates placed where the simulated paths actually are. The transition
probabilities between the states of week t and week t+1 are the empirical
frequencies of the paths.

Why joint K-means and not one K-means per coordinate? A product of per-coordinate
grids puts states in corners that never occur (very wet AND very expensive); joint
clustering spends the K states where the probability mass is, which matters when
the coordinates are correlated (price and inflow are).

Scaling. Before clustering, every coordinate is standardised (z-score) within the
week so no coordinate dominates because of its units. Inflows are clustered in log
space (they are lognormal); their centroids are mapped back with exp of the mean
log so that the state's inflow is the geometric mean of its members.

States are sorted by received price within each week; this is only a convention
that makes state indices easier to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

LOG_COORDS = ("inflow_upper", "inflow_lower")


@dataclass
class Lattice:
    T: int
    K: int
    coords: dict[str, np.ndarray]         # name -> (T, K) centroid values in original units
    transitions: list[np.ndarray]         # T-1 dense (K, K) row-stochastic matrices
    meta: dict = field(default_factory=dict)

    @property
    def price_coords(self) -> list[str]:
        return [c for c in self.coords if c.startswith("price")]

    def price(self, t: int, m: int) -> float:
        """Received spot price at state m of week t: sum of the price coordinates, floored at 0."""
        return max(0.0, float(sum(self.coords[c][t, m] for c in self.price_coords)))

    def inflow(self, t: int, m: int) -> tuple[float, float]:
        """(upper, lower) inflow in Mm3/week at state m of week t."""
        return float(self.coords["inflow_upper"][t, m]), float(self.coords["inflow_lower"][t, m])

    def price_matrix(self) -> np.ndarray:
        P = sum(self.coords[c] for c in self.price_coords)
        return np.maximum(P, 0.0)

    def successors(self, t: int, m: int, threshold: float = 1e-6):
        """Successor states of (t, m) with probability above ``threshold`` and their
        (renormalised) probabilities delta. Only these states enter the node problems."""
        row = self.transitions[t][m]
        idx = np.nonzero(row > threshold)[0]
        p = row[idx]
        return idx, p / p.sum()

    def marginals(self) -> np.ndarray:
        """(T, K) probability of being in each state, starting uniform in week 0."""
        out = np.zeros((self.T, self.K))
        out[0] = 1.0 / self.K
        for t in range(self.T - 1):
            out[t + 1] = out[t] @ self.transitions[t]
        return out

    def save(self, path: Path) -> None:
        np.savez_compressed(path, T=self.T, K=self.K, names=np.array(list(self.coords)),
                            **{f"coord_{k}": v for k, v in self.coords.items()},
                            transitions=np.stack(self.transitions))

    @classmethod
    def load(cls, path: Path) -> "Lattice":
        d = np.load(path)
        names = [str(n) for n in d["names"]]
        return cls(T=int(d["T"]), K=int(d["K"]), coords={n: d[f"coord_{n}"] for n in names},
                   transitions=list(d["transitions"]))


def build_lattice(paths: dict[str, np.ndarray], K: int, seed: int = 42, n_init: int = 4,
                  coordinates: tuple[str, ...] = ("price_chi", "price_xi", "inflow_upper", "inflow_lower"),
                  verbose: bool = True) -> Lattice:
    """Build a K-state-per-week Markov lattice from simulated paths.

    paths        dict of (n_paths, T) arrays from models.simulate_paths
    K            number of states per week
    coordinates  which columns of ``paths`` become lattice coordinates. The default
                 keeps the two price factors separate; ("price", "inflow_upper",
                 "inflow_lower") collapses the price to one coordinate.
    """
    n_paths, T = paths[coordinates[0]].shape
    is_log = [c in LOG_COORDS for c in coordinates]
    centroids = {c: np.zeros((T, K)) for c in coordinates}
    labels = np.zeros((n_paths, T), dtype=int)
    within = np.zeros(T)
    for t in range(T):
        # 1. coordinates of all paths in week t, logs for the inflows, standardised
        X = np.column_stack([np.log(np.maximum(paths[c][:, t], 1e-9)) if lg else paths[c][:, t]
                             for c, lg in zip(coordinates, is_log)])
        mean, sd = X.mean(0), X.std(0)
        Z = (X - mean) / np.where(sd > 0, sd, 1.0)
        # 2. one K-means over all coordinates jointly
        km = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit(Z)
        lab = km.labels_
        # 3. centroids in original units (mean of the members; exp of the mean log for inflows)
        cent = np.zeros((K, len(coordinates)))
        for k in range(K):
            members = lab == k
            cent[k] = X[members].mean(0) if members.any() else X.mean(0)
        within[t] = km.inertia_ / max(Z.var(0).sum() * n_paths, 1e-12)   # share of variance left inside clusters
        for j, lg in enumerate(is_log):
            if lg:
                cent[:, j] = np.exp(cent[:, j])
        # 4. sort states by received price (a convention only)
        price_cols = [j for j, c in enumerate(coordinates) if c.startswith("price")]
        order = np.argsort(cent[:, price_cols].sum(1), kind="stable")
        remap = np.argsort(order)
        labels[:, t] = remap[lab]
        for j, c in enumerate(coordinates):
            centroids[c][t] = cent[order, j]
        if verbose and (t % 13 == 0 or t == T - 1):
            print(f"  week {t:3d}: K-means done, {within[t]:.3f} of the variance left within clusters")

    # 5. empirical transition probabilities between consecutive weeks
    transitions = []
    for t in range(T - 1):
        M = np.zeros((K, K))
        np.add.at(M, (labels[:, t], labels[:, t + 1]), 1.0)
        rows = M.sum(1)
        marg = np.bincount(labels[:, t + 1], minlength=K).astype(float)
        for m in range(K):
            if rows[m] > 0:
                M[m] /= rows[m]
            else:                                   # a state no path visits: use the marginal of next week
                M[m] = marg / marg.sum()
        transitions.append(M)
    return Lattice(T=T, K=K, coords=centroids, transitions=transitions,
                   meta={"n_paths": n_paths, "seed": seed, "coordinates": list(coordinates),
                         "within_variance_share": float(within.mean())})
