"""Figures for the notebooks. Plain matplotlib, one small palette.

Colour use: a light-to-dark blue ramp for an ordered family of series (lattice
states, the sweep over h), one dark ink for the reference series (risk neutral,
data), green for the good-deal policy where exactly two series are compared."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sddp import NEG_PROB

RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
INK = "#1f2933"
INK2 = "#52606d"
MUTED = "#9aa5b1"
GREEN = "#2e7d32"


def style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
                         "axes.titlelocation": "left", "axes.titlesize": 10, "legend.frameon": False,
                         "legend.fontsize": 8, "font.size": 9})


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """Quantiles of a discrete distribution given by (values, weights), one row per week."""
    out = np.zeros((len(qs), values.shape[0]))
    for t in range(values.shape[0]):
        order = np.argsort(values[t]); v = values[t, order]; w = weights[t, order]
        cum = np.cumsum(w) / w.sum()
        out[:, t] = [v[min(np.searchsorted(cum, q), len(v) - 1)] for q in qs]
    return out


def lattice_fan(lattice, coord: str, title: str | None = None, ax=None, unit: str = "", n_paths: int = 6, seed: int = 0):
    """The lattice as a fan: probability-weighted 5–95 % and 25–75 % bands and the median of one
    coordinate week by week (weights = the lattice marginals), with a few paths sampled from the
    transition matrices on top. (The states are re-sorted by price every week, so joining
    'state m' across weeks would not be a path.)"""
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    C = lattice.coords[coord]; W = lattice.marginals(); x = np.arange(lattice.T)
    q = _weighted_quantiles(C, W, (0.05, 0.25, 0.5, 0.75, 0.95))
    ax.fill_between(x, q[0], q[4], color=RAMP[0], alpha=0.35, lw=0, label="5–95 %")
    ax.fill_between(x, q[1], q[3], color=RAMP[1], alpha=0.5, lw=0, label="25–75 %")
    ax.plot(x, q[2], color=RAMP[3], lw=1.8, label="median")
    rng = np.random.default_rng(seed)
    for i in range(n_paths):
        m = rng.choice(lattice.K, p=W[0]); path = [C[0, m]]
        for t in range(lattice.T - 1):
            m = rng.choice(lattice.K, p=lattice.transitions[t][m]); path.append(C[t + 1, m])
        ax.plot(x, path, color=INK, lw=0.7, alpha=0.6, label="sampled lattice paths" if i == 0 else None)
    ax.set_xlabel("week"); ax.set_ylabel(f"{coord} {unit}".strip())
    ax.set_title(title or f"lattice of {coord} ({lattice.K} states per week)"); ax.legend(fontsize=8)
    return ax


def lattice_scatter(paths: dict, lattice, t: int, x: str, y: str, ax=None, n_show: int = 2000):
    """Simulated paths of week t in the (x, y) plane with the lattice states on top."""
    ax = ax or plt.subplots(figsize=(4.5, 3.8))[1]
    n = min(n_show, paths[x].shape[0])
    ax.scatter(paths[x][:n, t], paths[y][:n, t], s=4, color=MUTED, alpha=0.4, label="simulated paths")
    def coord(name):          # "price" = the received price, the sum of the price coordinates
        return lattice.price_matrix()[t] if name == "price" else lattice.coords[name][t]
    ax.scatter(coord(x), coord(y), s=22, color=INK, zorder=3, label="lattice states")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"week {t}: joint K-means states"); ax.legend()
    return ax


def convergence(policies: dict, ax=None):
    """Upper bound per iteration for several policies."""
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    for i, (name, pol) in enumerate(policies.items()):
        col = INK if pol.config.h == 0 else RAMP[min(3, i)]
        ax.plot(pol.history["iteration"], pol.history["bound"] / 1e6, color=col, lw=1.8, label=name)
    ax.set_xlabel("SDDP iteration"); ax.set_ylabel("first-stage bound, M EUR"); ax.set_title("convergence of the policy bound")
    ax.legend()
    return ax


def storage_fan(sims: dict, which: int = 1, label: str = "lower", ax=None, legend: bool = True):
    """Median storage of a reservoir for each policy; the 10–90 % band is drawn for the first
    policy only (the bands of the policies overlap almost entirely, so one is enough to show
    the spread across paths). The legend sits below the axes so it never covers the curves;
    pass legend=False on the second reservoir panel, which uses the same colours."""
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    for i, (name, sim) in enumerate(sims.items()):
        S = sim["storage"][:, :, which]
        col = INK if i == 0 else RAMP[min(3, i)]
        if i == 0:
            ax.fill_between(np.arange(S.shape[1]), np.quantile(S, 0.1, 0), np.quantile(S, 0.9, 0), color=RAMP[0], alpha=0.35, lw=0,
                            label=f"10–90 % of paths, {name}")
        ax.plot(np.median(S, 0), color=col, lw=1.8, label=name)
    ax.set_xlabel("week"); ax.set_ylabel("storage, Mm³"); ax.set_title(f"{label} reservoir: median per policy")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    return ax


def revenue_hist(sims: dict, ax=None):
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    for i, (name, sim) in enumerate(sims.items()):
        col = INK if i == 0 else RAMP[min(3, i)]
        ax.hist(sim["total"] / 1e6, bins=30, histtype="step", lw=1.6, color=col, label=name)
    ax.set_xlabel("total after-tax revenue, M EUR"); ax.set_ylabel("paths"); ax.set_title("revenue distribution (physical measure)")
    ax.legend()
    return ax


def bound_vs_h(table, ax=None):
    """Policy value as a function of the Sharpe bound h, one line per contract set
    (table with columns h, bound, mean and optionally contract_set; the risk-neutral
    row, h = 0, is the common starting point of every line)."""
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    t = table.copy()
    if "contract_set" not in t.columns:
        t["contract_set"] = ""
    t["contract_set"] = t["contract_set"].fillna("").astype(str)
    rn = t[t["h"] <= 0]
    sets = [c for c in dict.fromkeys(t.loc[t["h"] > 0, "contract_set"])]   # in order of appearance
    if not sets:
        sets = [""]
    for i, cs in enumerate(sets):
        sub = pd.concat([rn, t[(t["h"] > 0) & (t["contract_set"] == cs)]]).sort_values("h")
        lab = "good-deal bound" + (f", contracts {cs}" if cs else "")
        ax.plot(sub["h"], sub["bound"] / 1e6, "o-", color=RAMP[min(3, 3 - i)], label=lab)
    ax.plot(t.sort_values("h")["h"], t.sort_values("h")["mean"] / 1e6, "s--", color=INK2, label="simulated mean revenue")
    ax.set_xlabel("annual Sharpe bound h"); ax.set_ylabel("M EUR"); ax.set_title("value against the good-deal budget"); ax.legend(fontsize=8)
    return ax


def distortion_tilt(policy, lattice, ax=None):
    """How the worst-case measure shifts the expected next-week inflow and price, per week,
    averaged over the states of the week (from the recorded q* of the last backward pass)."""
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    T = lattice.T
    rows = {"inflow_lower": np.full(T - 1, np.nan), "inflow_upper": np.full(T - 1, np.nan), "price": np.full(T - 1, np.nan)}
    P = lattice.price_matrix()
    for t in range(T - 1):
        acc = {k: [] for k in rows}
        for m in range(lattice.K):
            if (t, m) not in policy.q_star:
                continue
            sidx, q = policy.q_star[(t, m)]
            _, delta = lattice.successors(t, m, NEG_PROB)
            for k in ("inflow_lower", "inflow_upper"):
                v = lattice.coords[k][t + 1, sidx]
                acc[k].append((q @ v - delta @ v) / max(np.sqrt(delta @ (v - delta @ v) ** 2), 1e-9))
            v = P[t + 1, sidx]
            acc["price"].append((q @ v - delta @ v) / max(np.sqrt(delta @ (v - delta @ v) ** 2), 1e-9))
        for k in rows:
            if acc[k]:
                rows[k][t] = np.mean(acc[k])
    for k, col in zip(rows, (RAMP[3], RAMP[1], INK)):
        ax.plot(rows[k], color=col, lw=1.6, label=k)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("week"); ax.set_ylabel("E_q − E_δ, in std units")
    ax.set_title("where the worst-case measure spends its budget (mean over states)"); ax.legend()
    return ax

