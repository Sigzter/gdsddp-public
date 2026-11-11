# Data

Three inputs feed the pipeline. Only the synthetic sample is committed; the real files are proprietary or licensed
and must be placed in `data/raw/` (git-ignored) by whoever has access to them.

| input | real source | file expected in `data/raw/` | sample in `data/sample/` |
|---|---|---|---|
| Nord Pool **system price**, hourly | public: Energinet "Energi Data Service", dataset `Elspotprices`, price area `SYSTEM` (free, no key; fetched by notebook 1 and cached) | `system_price_hourly.csv` (`date, price_eur_mwh`), written by the notebook | `system_price_daily.csv` |
| **inflow** per station of the Namsen watercourse, m³/s hourly 1958–2025 | NTE / SINTEF RenewHydro dataset (Kong et al. 2025), delivered under agreement, not public | `Historical_inflow.csv` (`date, Namsvatn, Vekteren, Limingen, Tunnsjø, Tunnsjøflyene, ...`) | `inflow_daily.csv` (same columns, daily) |
| **futures panel**: daily settlement prices of Nordic power week/month/quarter/year contracts 2014–2024 | Nasdaq end-of-day data (licensed) | `nordic_power_eod_2014_2024.csv.gz` (`date, main_category, instr_type, series, expire, closing_price, ...`) | `forward_panel.csv` (same columns, subset) |

The loaders accept any regular time step for the flow and price files (they aggregate to weeks), so a daily sample and
an hourly real file go through the same code.

## Physical constants (not estimated)

The two-reservoir reduction of the Namsen cascade comes from the plant topology delivered with the inflow dataset and
from the SINTEF report that accompanies it; the values used in notebook 4 are the thesis' table 2:

| reservoir | stations | s_max (Mm³) | x_max (Mm³/week) | energy factor (MWh/Mm³) |
|---|---|---|---|---|
| upper (Namsvatn) | Namsvatn + Vekteren + Limingen | 458 | 74.4 | 221.5 |
| lower (Tunnsjø) | Tunnsjø + Tunnsjøflyene | 440 | 70.2 | 575.7 |

Economics: risk-free rate 3 %, corporate tax 22 %, resource-rent tax 45 % (Norwegian hydropower, 2025).

## The synthetic sample

`scripts/make_sample_data.py` draws every sample series from the models in `gdsddp/models.py` with round,
literature-style parameters (price: `phi_chi 0.977, sigma_chi 2.3, sigma_xi 0.55, season 4.4 / 2.8`; inflow: a
seasonal profile with a spring flood, `phi 0.7, sigma 0.35, rho 0.7`, annual means 1100 / 430 Mm³). The futures panel
prices week, month, quarter and year contracts from the two-factor model on a daily factor path, so notebook 3 can be
checked against known parameters. Nothing in `data/sample/` is derived from the real data.

## Other public sources

* Nord Pool's own data portal API (subscription) for prices after February 2025, when Energinet stopped publishing the SYSTEM area.
* ENTSO-E transparency platform (free token) for any bidding zone, e.g. NO4.
* `hvakosterstrommen.no` for Norwegian zones from December 2021.
* NVE (Norwegian Water Resources and Energy Directorate) publishes weekly reservoir filling and some inflow series that can substitute the NTE data for a different watercourse.
