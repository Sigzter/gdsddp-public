# gdsddp-public

Reference implementation of the method in

> Berthet, M. and Tveit, S. (2026). *Good-deal bounds in SDDP for medium-term hydropower scheduling.*
> Project thesis, Department of Industrial Economics and Technology Management, NTNU.
> Supervisors: Stein-Erik Fleten and Felipe van de Sande Araujo.

The code schedules a two-reservoir hydropower cascade over a weekly horizon with stochastic dual dynamic programming
(SDDP) and values it under a good-deal bound: at every node of the backward pass the continuation value is taken under
the worst probability measure that keeps every traded forward contract fairly priced and has a Sharpe ratio below a
bound `h`. Setting `h = 0` gives the risk-neutral policy. Everything is single-threaded; the node LP and the good-deal
QCQP are solved with Gurobi.

## What is done

1. **Price model.** Weekly system price `P = xi + chi + season`: `xi` a random walk (long-term level), `chi` an AR(1)
   (short-term deviation), a cosine season. The season and the starting level are fitted on the hourly system price;
   `phi_chi`, `sigma_chi`, `sigma_xi` are fitted to the covariance of weekly fixed-delivery log returns of a
   maximum-smoothness forward curve built from the futures panel, after a PCA of those returns.
2. **Inflow model.** Station flows are aggregated to two equivalent reservoirs and to weekly volumes. Each reservoir
   follows a periodic AR(1) on the log inflow (one parameter set per week of the year); the two innovations are
   correlated with each other and with the price innovation.
3. **Lattice.** The models are simulated jointly and quantised week by week with one K-means over
   `(chi, xi + season, log I_upper, log I_lower)` into `K` states; transitions are empirical frequencies.
4. **Contracts.** Forward contracts settle on the weekly price. Their fair value in every node is computed by backward
   recursion, and their increments to the successors have zero mean under the physical probabilities.
5. **Node LP.** Two reservoirs in cascade; release, spill and end storage maximise after-tax revenue plus the
   discounted continuation value, represented by one cut-bounded variable per successor. The duals of the water
   balances give the next cut. A terminal water value credits the water left at the end of the horizon.
6. **Good-deal step.** The successor weights of the backward pass are the solution of

   ```
   min_q  sum_m' q(m') V(m')
   s.t.   sum q = 1,  q >= 0
          sum_m' q(m') dF_j(m') = 0                  for every live contract j
          sum_m' q(m')^2 / delta(m') <= 1 + h^2/52
   ```

7. **SDDP.** Cuts seeded by backward sweeps at 10, 50 and 90 % of capacity, then one forward pass along a sampled
   lattice path and one backward pass over all states per iteration. Stops at a maximum iteration count or when the
   first-stage bound has moved less than a tolerance over a window. The measure `q*` of the last pass is stored.
8. **Simulation.** Each policy is simulated on fresh paths under the physical measure.

## Contents

```
gdsddp/
  models.py        PriceModel, InflowModel, simulate_paths
  lattice.py       Lattice, build_lattice (scikit-learn KMeans)
  contracts.py     Contract, ContractSystem (fair values, strikes, martingale increments)
  hydro.py         Reservoir, Cascade, Economics, NodeLP (Gurobi), terminal_water_values
  gooddeal.py      worst_case_measure (Gurobi QCQP)
  sddp.py          SDDPConfig, Policy, train
  simulate.py      simulate, revenue_summary
  plots.py         figures
  data.py          price API fetch, station flows to weekly reservoir inflow, season fit, JSON helpers
  forwardcurve.py  futures panel to forward curve (block or maximum smoothness), fixed-delivery returns, PCA, two-factor fit
  experiment.py    load_parameters, build / train_and_simulate / report used by scripts/run_experiment.py
notebooks/
  01_price_data_api.ipynb           system price from Energi Data Service; weekly series, season, xi_0
  02_namsen_inflow_analysis.ipynb   reservoir inflow from station flows; periodic AR(1) fit; price–inflow coupling
  03_forward_curve_pca.ipynb        forward curve, fixed-delivery returns, PCA, phi_chi / sigma_chi / sigma_xi
  04_run_gdsddp.ipynb               lattice, contracts, SDDP training (risk neutral + good deal), simulation, figures
scripts/
  make_sample_data.py      regenerates data/sample/ from the models with known parameters
  run_experiment.py        build | train | report, for runs too large for the notebook
data/
  README.md                data sources, expected raw files, physical constants
  sample/                  synthetic inputs (committed)
  raw/, processed/         real inputs and derived parameters (git-ignored)
tests/
  test_smoke.py            end-to-end check on the sample data (about one minute)
results/                   outputs of notebook 4 and run_experiment.py (git-ignored)
```

Notebooks 1–3 write their parameters to `data/processed/`; notebook 4 reads them and falls back to literature values
if they are missing. The notebooks are committed with outputs: notebook 1 on the public system price, notebooks 2 and 3
on the sample data (notebook 3 is a parameter-recovery test, generated `0.977, 2.3, 0.55`, fitted `0.974, 2.62, 0.51`),
notebook 4 on a run with 102 weeks, 200 states and the eight-contract set.

## Install and run

```bash
pip install -r requirements.txt
python3 tests/test_smoke.py
jupyter lab notebooks/          # 01 -> 02 -> 03 -> 04
```

Python 3.10 or newer and a Gurobi licence (the academic licence is sufficient). Notebook 4 is committed with the
settings of the 102-week, 200-state run (hours, about 50 GB); set `T = 52`, `K = 50`, `N_LATTICE_PATHS = 5000`,
`N_SIM_PATHS = 300` in its parameter cell for a laptop run of about 30 minutes. For real data, place the files
described in `data/README.md` in `data/raw/` and rerun the notebooks.

## Larger runs

Every node LP is kept in memory with one cut row per successor, so a policy with 102 weeks and 200 states grows to
about 50 GB over 50 iterations and takes 1–2 hours on one core. `scripts/run_experiment.py` splits a run into
independent steps:

```bash
python3 scripts/run_experiment.py build  --name full --T 102 --K 200 --paths 30000
python3 scripts/run_experiment.py train  --name full --h 0
python3 scripts/run_experiment.py train  --name full --h 0.5 --contracts H2:27-52+Y2:53-102 --tag 2c
python3 scripts/run_experiment.py report --name full
```

Each `train` is one job and the jobs are independent; results go to `results/<name>/<policy>/`.

## Simplifications relative to the thesis

* One plant and one price-factor pair; no hedging overlays.
* No cut pruning and no parallel backward pass.
* The good-deal step is solved as a QCQP at every node; the thesis uses the closed-form KKT solution.
* One joint K-means over all coordinates; the thesis clusters price and inflow separately.

## References

* Cochrane, J. H. and Saá-Requejo, J. (2000). Beyond arbitrage: good-deal asset price bounds in incomplete markets. *JPE* 108.
* Schwartz, E. and Smith, J. E. (2000). Short-term variations and long-term dynamics in commodity prices. *Management Science* 46.
* Pereira, M. V. F. and Pinto, L. M. V. G. (1991). Multi-stage stochastic optimization applied to energy planning. *Math. Programming* 52.
* Koekebakker, S. and Ollmar, F. (2005). Forward curve dynamics in the Nordic electricity market. *Managerial Finance* 31.
* Kleiven, A., Risanger, S. and Fleten, S.-E. (2023). Co-movements between forward prices and resource availability in hydro-dominated electricity markets. *Energy Systems*.
