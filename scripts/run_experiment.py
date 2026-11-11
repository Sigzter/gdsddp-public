#!/usr/bin/env python3
"""Command-line version of notebook 4, in three steps so a large run can be split into jobs.

    python3 scripts/run_experiment.py build  --name full --T 102 --K 200 --paths 20000
    python3 scripts/run_experiment.py train  --name full --h 0 --iterations 30        # risk neutral
    python3 scripts/run_experiment.py train  --name full --h 0.5 --iterations 30
    python3 scripts/run_experiment.py train  --name full --h 1.0 --iterations 30
    python3 scripts/run_experiment.py report --name full

Everything lands in results/<name>/: lattice.npz and config.json from build, one folder per policy
from train (history.csv, cuts.npz, sim.npz, q_star.npz, summary.json), summary.csv and figures from
report. The train steps are independent of each other, so they can run on different machines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdsddp import experiment  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build", help="simulate the models and build the lattice")
    b.add_argument("--name", required=True); b.add_argument("--T", type=int, default=52); b.add_argument("--K", type=int, default=50)
    b.add_argument("--paths", type=int, default=5000); b.add_argument("--seed", type=int, default=42)
    b.add_argument("--start-week", type=int, default=1); b.add_argument("--n-init", type=int, default=4, help="K-means restarts per week")
    t = sub.add_parser("train", help="train one policy with SDDP and simulate it")
    t.add_argument("--name", required=True); t.add_argument("--h", type=float, required=True, help="annual Sharpe bound; 0 = risk neutral")
    t.add_argument("--iterations", type=int, default=30); t.add_argument("--tol", type=float, default=1e-4); t.add_argument("--window", type=int, default=10)
    t.add_argument("--sim-paths", type=int, default=300); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--contracts", default=None, help="e.g. 'H2:27-52,Y2:53-104' or 'H2:27-52+Y2:53-104' (delivery weeks, 1-based; '+' is safe inside qsub -v), 'none', or omitted for the default by T")
    t.add_argument("--tag", default="", help="suffix of the policy folder, e.g. the contract set '2c'")
    r = sub.add_parser("report", help="summary table and figures of the finished policies")
    r.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.step == "build":
        experiment.build(a.name, a.T, a.K, a.paths, seed=a.seed, start_week=a.start_week, n_init=a.n_init)
    elif a.step == "train":
        experiment.train_and_simulate(a.name, a.h, iterations=a.iterations, tol=a.tol, window=a.window,
                                      sim_paths=a.sim_paths, seed=a.seed, contracts=a.contracts, tag=a.tag)
    else:
        experiment.report(a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
