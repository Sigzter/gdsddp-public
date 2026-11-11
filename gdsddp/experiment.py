"""The experiment of notebook 4 as three reusable steps, so that a large run can be split into
separate jobs: BUILD the lattice once, TRAIN-AND-SIMULATE one policy per job, REPORT at the end.

    python3 scripts/run_experiment.py build  --name full --T 102 --K 200 --paths 20000
    python3 scripts/run_experiment.py train  --name full --h 0            # risk neutral
    python3 scripts/run_experiment.py train  --name full --h 0.5
    python3 scripts/run_experiment.py report --name full

Every step reads and writes results/<name>/. The parameters of the models are read from
data/processed/ (written by notebooks 1-3) when present, otherwise the literature values are used,
exactly as in notebook 4.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Contract, ContractSystem
from .data import PROCESSED, ROOT, SAMPLE, load_json, save_json
from .hydro import Cascade, Economics, Reservoir
from .lattice import Lattice, build_lattice
from .models import InflowModel, PriceModel, simulate_paths
from .sddp import Policy, SDDPConfig, train
from .simulate import revenue_summary, simulate

RESULTS = ROOT / "results"


# ------------------------------------------------------------------ parameters
def load_parameters(processed: Path = PROCESSED):
    """(PriceModel, InflowModel, rho_price_inflow, sources) from data/processed or the defaults."""
    src = {}
    if (processed / "price_params.json").exists():
        pp = load_json(processed / "price_params.json"); src["dynamics"] = "data/processed/price_params.json"
    else:
        pp = {"phi_chi": 0.977, "sigma_chi": 2.292, "sigma_xi": 0.554}; src["dynamics"] = "literature (Kleiven et al. 2023)"
    if (processed / "price_season.json").exists():
        se = load_json(processed / "price_season.json"); src["season"] = "data/processed/price_season.json"
    else:
        se = {"season_amplitude": 4.421, "season_phase": 2.792, "xi0": 30.0}; src["season"] = "thesis table 3"
    price = PriceModel(phi_chi=pp["phi_chi"], sigma_chi=pp["sigma_chi"], sigma_xi=pp["sigma_xi"],
                       season_amplitude=se["season_amplitude"], season_phase=se["season_phase"], xi0=se["xi0"])
    if (processed / "inflow_params.json").exists():
        inflow = InflowModel.from_json(processed / "inflow_params.json"); src["inflow"] = "data/processed/inflow_params.json"
    else:
        from .data import load_station_flows
        inflow = InflowModel.fit(load_station_flows(SAMPLE / "inflow_daily.csv")); src["inflow"] = "fitted on data/sample/inflow_daily.csv"
    if (processed / "price_inflow_coupling.json").exists():
        coupling = float(load_json(processed / "price_inflow_coupling.json")["rho_price_inflow"]); src["coupling"] = "data/processed"
    else:
        coupling = -0.3; src["coupling"] = "default -0.3"
    return price, inflow, coupling, src


def default_cascade() -> Cascade:
    """The thesis' two-reservoir reduction of the Namsen cascade (table 2)."""
    return Cascade(upper=Reservoir("Namsvatn", s_max=458, s_init=261.5, x_max=74.4, energy_factor=221.5),
                   lower=Reservoir("Tunnsjø", s_max=440, s_init=296.5, x_max=70.2, energy_factor=575.7))


def default_contracts(T: int) -> list[Contract]:
    """A quarter and a half-year in year one, a full second year when the horizon has one
    (the thesis' two-contract set is H2 [27, 52] and Y2 [53, 104])."""
    cs = [Contract("Q2", 14, min(26, T)), Contract("H2", 27, min(52, T))] if T >= 27 else [Contract("Q", 1, T)]
    if T > 52:
        cs.append(Contract("Y2", 53, T))
    return [c for c in cs if c.start <= T]


def parse_contracts(spec: str | None, T: int) -> list[Contract]:
    """'H2:27-52,Y2:53-104' -> contracts (clipped to T); None -> default_contracts(T); 'none' -> no contracts.
    Items may be separated by ',', '+', ';' or spaces ('+' is the one to use inside ``qsub -v``, which splits on commas)."""
    if not spec:
        return default_contracts(T)
    if spec.strip().lower() == "none":
        return []
    out = []
    for item in re.split(r"[,+;\s]+", spec.strip()):
        cid, rng = item.split(":"); a, b = rng.split("-")
        if int(a) <= T:
            out.append(Contract(cid.strip(), int(a), min(int(b), T)))
    return out


# ----------------------------------------------------------------------- steps
def build(name: str, T: int, K: int, n_paths: int, seed: int = 42, start_week: int = 1, n_init: int = 4,
          log=print) -> Lattice:
    out = RESULTS / name; out.mkdir(parents=True, exist_ok=True)
    price, inflow, coupling, src = load_parameters()
    log(f"[build] {name}: T={T} K={K} paths={n_paths}; parameters from {src}")
    t0 = time.time()
    paths = simulate_paths(price, inflow, n_paths, T, seed=seed, rho_price_inflow=coupling, start_week=start_week)
    lattice = build_lattice(paths, K, seed=seed, n_init=n_init, verbose=True)
    lattice.save(out / "lattice.npz")
    save_json({"name": name, "T": T, "K": K, "n_paths": n_paths, "seed": seed, "start_week": start_week,
               "price": vars(price), "coupling": coupling, "sources": src, "lattice_meta": lattice.meta,
               "cascade": {"upper": vars(default_cascade().upper), "lower": vars(default_cascade().lower)},
               "economics": vars(Economics()), "build_seconds": time.time() - t0}, out / "config.json")
    log(f"[build] lattice saved to {out / 'lattice.npz'} ({time.time() - t0:.0f} s)")
    return lattice


def policy_tag(h: float, tag: str = "") -> str:
    base = "risk_neutral" if h <= 0 else f"good_deal_h{h:g}"
    return f"{base}_{tag}" if tag else base


def train_and_simulate(name: str, h: float, iterations: int = 30, tol: float = 1e-4, window: int = 10,
                       sim_paths: int = 300, seed: int = 42, contracts: str | None = None, tag: str = "", log=print) -> dict:
    """``tag`` (e.g. the contract set, '2c') is appended to the policy folder name so several contract
    sets can live in one results directory; the risk-neutral policy needs no tag (it uses no contracts)."""
    out = RESULTS / name
    cfg = load_json(out / "config.json")
    lattice = Lattice.load(out / "lattice.npz")
    cascade, econ = default_cascade(), Economics()
    cs = ContractSystem(parse_contracts(contracts, lattice.T), lattice)
    contract_set = tag; tag = policy_tag(h, tag); pdir = out / tag; pdir.mkdir(exist_ok=True)
    log(f"[train] {name}/{tag}: contracts {[(c.id, c.start, c.end) for c in cs.contracts]}, strikes "
        f"{ {k: round(v, 2) for k, v in cs.strikes.items()} }, martingale check {cs.check_martingale():.1e}")
    t0 = time.time()
    pol = train(lattice, cascade, econ, cs, SDDPConfig(iterations=iterations, tol=tol, window=window, seed=seed, h=h),
                name=tag, log=log)
    t_train = time.time() - t0
    log(f"[train] done in {t_train / 60:.1f} min, bound {pol.bound / 1e6:.3f} M EUR; simulating {sim_paths} paths")
    t0 = time.time()
    sim = simulate(pol, lattice, cascade, econ, n_paths=sim_paths, seed=seed + 1, log=log)
    summ = revenue_summary(sim["total"])
    pol.history.to_csv(pdir / "history.csv", index=False)
    np.savez_compressed(pdir / "cuts.npz", cuts=np.array([(t, m, *c) for t in range(lattice.T) for m in range(lattice.K) for c in pol.cuts[t][m]]))
    np.savez_compressed(pdir / "sim.npz", **sim)
    if pol.q_star:
        keys = sorted(pol.q_star)
        np.savez_compressed(pdir / "q_star.npz", t=np.array([k[0] for k in keys]), m=np.array([k[1] for k in keys]),
                            succ=np.array([pol.q_star[k][0] for k in keys], dtype=object),
                            q=np.array([pol.q_star[k][1] for k in keys], dtype=object),
                            budget=np.array([pol.budget_used[k] for k in keys]))
    res = {"name": name, "policy": tag, "h": h, "contract_set": contract_set, "bound": pol.bound, "iterations": int(len(pol.history)),
           "fallbacks": int(pol.history["fallbacks"].sum()), "train_minutes": t_train / 60, "sim_minutes": (time.time() - t0) / 60,
           "end_storage_upper": float(sim["storage"][:, -1, 0].mean()), "end_storage_lower": float(sim["storage"][:, -1, 1].mean()),
           "contracts": [(c.id, c.start, c.end) for c in cs.contracts], "strikes": cs.strikes, **summ}
    save_json(res, pdir / "summary.json")
    log(f"[train] {tag}: bound {pol.bound / 1e6:.3f} M, simulated mean {summ['mean'] / 1e6:.3f} M, "
        f"end storage lower {res['end_storage_lower']:.0f} Mm3 -> {pdir}")
    return res


def report(name: str, log=print) -> pd.DataFrame:
    """Collect the policies of a run into summary.csv and the standard figures (no training)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plots
    plots.style()
    out = RESULTS / name
    lattice = Lattice.load(out / "lattice.npz")
    rows, hist, sims = [], {}, {}
    for pdir in sorted(p for p in out.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        s = load_json(pdir / "summary.json"); rows.append(s)
        hist[s["policy"]] = pd.read_csv(pdir / "history.csv")
        sims[s["policy"]] = dict(np.load(pdir / "sim.npz"))
    if not rows:
        raise FileNotFoundError(f"no finished policies under {out}")
    table = pd.DataFrame(rows).sort_values("h").set_index("policy")
    rn = table.loc["risk_neutral", "bound"] if "risk_neutral" in table.index else np.nan
    table["premium_vs_rn"] = rn - table["bound"]
    cols = ["h", "contract_set", "bound", "mean", "std", "cvar5", "end_storage_lower", "end_storage_upper", "premium_vs_rn", "iterations", "fallbacks", "train_minutes"]
    cols = [c for c in cols if c in table.columns]
    table[cols].to_csv(out / "summary.csv")
    log(table[cols].to_string())
    # figures
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (tag, hh) in enumerate(hist.items()):
        ax.plot(hh["iteration"], hh["bound"] / 1e6, color=plots.INK if tag == "risk_neutral" else plots.RAMP[min(3, i)], lw=1.8, label=tag)
    ax.set_xlabel("SDDP iteration"); ax.set_ylabel("first-stage bound, M EUR"); ax.set_title("convergence"); ax.legend()
    fig.savefig(out / "fig_convergence.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    plots.storage_fan(sims, which=1, label="lower", ax=axes[0]); plots.storage_fan(sims, which=0, label="upper", ax=axes[1], legend=False); plots.revenue_hist(sims, ax=axes[2])
    fig.savefig(out / "fig_storage_revenue.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    if len(table) > 1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        plots.bound_vs_h(table.reset_index(), ax=ax); fig.savefig(out / "fig_bound_vs_h.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    # distortion tilt of the largest-h policy, from the saved q*
    gd = [p for p in out.iterdir() if p.is_dir() and (p / "q_star.npz").exists()]
    if gd:
        best = max(gd, key=lambda p: load_json(p / "summary.json")["h"])
        qs = np.load(best / "q_star.npz", allow_pickle=True)
        pol = Policy(cuts=[], history=pd.DataFrame(), config=SDDPConfig(h=load_json(best / "summary.json")["h"]), name=best.name,
                     q_star={(int(t), int(m)): (s, q) for t, m, s, q in zip(qs["t"], qs["m"], qs["succ"], qs["q"])})
        fig, ax = plt.subplots(figsize=(7, 3.2)); plots.distortion_tilt(pol, lattice, ax=ax); ax.set_title(f"{best.name}: where the worst-case measure spends its budget")
        fig.savefig(out / "fig_distortion.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    log(f"[report] wrote summary.csv and figures to {out}")
    return table
