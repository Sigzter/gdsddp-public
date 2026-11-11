"""End-to-end smoke test on the sample data (about one minute; needs Gurobi).

    python3 -m pytest tests/            or            python3 tests/test_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdsddp.data import SAMPLE, load_station_flows, load_price_file, weekly_mean, fit_season   # noqa: E402
from gdsddp.models import PriceModel, InflowModel, simulate_paths                             # noqa: E402
from gdsddp.lattice import build_lattice                                                       # noqa: E402
from gdsddp.contracts import Contract, ContractSystem                                          # noqa: E402
from gdsddp.hydro import Reservoir, Cascade, Economics                                         # noqa: E402
from gdsddp.gooddeal import worst_case_measure                                                 # noqa: E402
from gdsddp.sddp import SDDPConfig, train                                                      # noqa: E402
from gdsddp.simulate import simulate, revenue_summary                                          # noqa: E402
from gdsddp.forwardcurve import load_panel, build_block_curve, weekly_fixed_delivery_returns, pca, fit_two_factor  # noqa: E402


def test_pipeline():
    weekly = load_station_flows(SAMPLE / "inflow_daily.csv")
    inflow = InflowModel.fit(weekly)
    assert 0.5 < inflow.phi.mean() < 0.95 and 0.3 < inflow.rho.mean() < 0.9
    season = fit_season(weekly_mean(load_price_file(SAMPLE / "system_price_daily.csv")))
    price = PriceModel(xi0=32.0, season_amplitude=season["season_amplitude"], season_phase=season["season_phase"])

    T, K = 20, 15
    paths = simulate_paths(price, inflow, 2000, T, seed=1, rho_price_inflow=-0.3)
    lat = build_lattice(paths, K, verbose=False)
    assert all(np.allclose(M.sum(1), 1.0) for M in lat.transitions)
    cs = ContractSystem([Contract("Q", 10, 20)], lat)
    assert cs.check_martingale() < 1e-9

    idx, delta = lat.successors(5, 3)
    ids, dF = cs.increments(5, 3, idx)
    V = lat.coords["inflow_lower"][6, idx] * 1e4 + lat.price_matrix()[6, idx] * 1e5
    gd = worst_case_measure(V, delta, dF, 0.75)
    assert gd.status == "optimal" and gd.value < delta @ V and abs(dF @ gd.q).max() < 1e-8 and abs(gd.budget_used - 1) < 1e-6

    cascade = Cascade(Reservoir("U", 458, 261.5, 74.4, 221.5), Reservoir("L", 440, 296.5, 70.2, 575.7))
    econ = Economics()
    rn = train(lat, cascade, econ, cs, SDDPConfig(iterations=4, h=0.0), name="RN", log=lambda *a: None)
    gd_pol = train(lat, cascade, econ, cs, SDDPConfig(iterations=4, h=0.75), name="GD", log=lambda *a: None)
    assert gd_pol.bound < rn.bound and gd_pol.history["fallbacks"].sum() == 0
    assert np.all(np.diff(rn.history["bound"]) <= 1e-6)          # the bound never increases
    sim = simulate(rn, lat, cascade, econ, n_paths=20, log=lambda *a: None)
    s = revenue_summary(sim["total"])
    assert s["mean"] > 0 and sim["storage"].max() <= 458 + 1e-6 and sim["storage"].min() >= -1e-6


def test_pca_recovers_sample_parameters():
    curve, ctypes = build_block_curve(load_panel(SAMPLE / "forward_panel.csv"))
    R = weekly_fixed_delivery_returns(curve, ctypes)
    res = pca(R); fit = fit_two_factor(res["cov"], res["k"])
    level = float(np.nanmean(curve.values))
    assert 0.96 < fit["phi_chi"] < 0.985 and 1.8 < fit["sigma_chi"] * level < 3.2 and 0.3 < fit["sigma_xi"] * level < 0.8
    assert res["share"][0] > 0.8


if __name__ == "__main__":
    test_pipeline(); test_pca_recovers_sample_parameters(); print("ok")
