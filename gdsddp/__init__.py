"""gdsddp — good-deal bounds inside SDDP for a two-reservoir hydropower cascade.

A deliberately small, single-threaded reference implementation of the method in
Berthet & Tveit (2026), "Good-deal bounds in SDDP for medium-term hydropower
scheduling" (NTNU project thesis). Read the modules in this order:

    models.py     the price model (Schwartz-Smith, two factors) and the inflow model
                  (periodic AR(1) per reservoir with correlated innovations)
    lattice.py    quantisation of simulated paths into a Markov lattice (joint K-means)
    contracts.py  forward contracts on the lattice and their martingale increments
    hydro.py      the reservoir system and the per-node production LP (Gurobi)
    gooddeal.py   the per-node worst-case measure q* (Gurobi QCQP)
    sddp.py       the SDDP loop: forward pass, backward pass, cuts, convergence
    simulate.py   forward simulation of a trained policy and revenue statistics
    plots.py      figures used by the notebooks

Every function is documented where it is defined; the notebooks under notebooks/
walk through the whole pipeline from raw data to results.
"""

from .models import PriceModel, InflowModel, simulate_paths            # noqa: F401
from .lattice import Lattice, build_lattice                             # noqa: F401
from .contracts import Contract, ContractSystem                         # noqa: F401
from .hydro import Reservoir, Cascade, Economics                        # noqa: F401
from .gooddeal import worst_case_measure, weekly_ball_radius            # noqa: F401
from .sddp import SDDPConfig, Policy, train                             # noqa: F401
from .simulate import simulate, revenue_summary                         # noqa: F401

__version__ = "1.0.0"
