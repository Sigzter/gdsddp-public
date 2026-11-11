#!/usr/bin/env python3
"""Generate the SYNTHETIC sample dataset in data/sample/ so that every notebook runs
without the proprietary inputs. Nothing here is real data: the series are drawn
from the models of gdsddp/models.py with round, literature-style parameters.

    python3 scripts/make_sample_data.py

Files written (same layout as the real inputs, see data/README.md):
  data/sample/system_price_daily.csv   date, price_eur_mwh         (daily, 2015-2024)
  data/sample/inflow_daily.csv         date + one column per station, m3/s (daily, 1990-2024);
                                       drawn jointly with the price (innovation correlation -0.3)
  data/sample/forward_panel.csv        date, series, closing_price, ...  (daily quotes, 2016-2020)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdsddp.models import PriceModel                       # noqa: E402
from gdsddp.data import UPPER_STATIONS, LOWER_STATIONS     # noqa: E402

OUT = ROOT / "data" / "sample"
PANEL_LEVEL = 55.0
RHO_PRICE_INFLOW = -0.3     # correlation of the inflow innovations with the week's total price innovation (thesis: about -0.3)
PRICE = PriceModel(phi_chi=0.977, sigma_chi=2.3, sigma_xi=0.55, season_amplitude=4.4, season_phase=2.8, xi0=30.0)


def seasonal_profile(woy):
    """Shape of the mean weekly inflow over the year: low winter base, a spring-flood
    peak around week 22 and a smaller autumn bump; normalised to mean 1."""
    w = np.asarray(woy, float)
    shape = 0.3 + 2.8 * np.exp(-((w - 22) / 3.5) ** 2) + 0.6 * np.exp(-((w - 40) / 6.0) ** 2)
    return shape / shape.mean()


def price_and_inflow(rng):
    """Weekly price (2015-2024) and inflow (1990-2024) drawn JOINTLY: in the weeks both exist the
    inflow innovations share a component with the total price innovation (correlation
    RHO_PRICE_INFLOW), the same construction as InflowModel.simulate. Both are then expanded to days."""
    weeks = pd.date_range("1990-01-01", "2024-12-30", freq="W-MON")
    woy = np.clip(weeks.isocalendar().week.to_numpy().astype(int), 1, 52)
    n = len(weeks); first_price = int(np.searchsorted(weeks.values, np.datetime64("2015-01-05")))
    # --- price factors on the price weeks
    nP = n - first_price
    chi, xi, season, eps_tot = PRICE.simulate(1, nP, rng, start_week=int(woy[first_price]))
    weekly_price = np.maximum(chi[0] + xi[0] + season, 1.0)
    # --- inflow: log-AR(1) deviations around the profile, innovations correlated with the price
    prof = seasonal_profile(np.arange(1, 53))
    mean_m3s = {"upper": 1100e6 / (365.25 * 86400), "lower": 430e6 / (365.25 * 86400)}
    phi, sig, rho_ul, rho_p = 0.7, 0.35, 0.7, RHO_PRICE_INFLOW
    dev = np.zeros((2, n)); prev = np.zeros(2); s_own = np.sqrt(1 - rho_p ** 2)
    r_adj = (rho_ul - rho_p ** 2) / (1 - rho_p ** 2)        # keeps corr(upper, lower) = rho_ul after the common component
    for t in range(n):
        z = rng.standard_normal(2)
        e = np.array([z[0], r_adj * z[0] + np.sqrt(1 - r_adj ** 2) * z[1]])
        if t >= first_price:                                  # share a component with this week's price innovation
            e = rho_p * eps_tot[0, t - first_price] + s_own * e
        prev = phi * prev + sig * e; dev[:, t] = prev
    stat_var = sig ** 2 / (1 - phi ** 2)
    flows = {r: mean_m3s[r] * prof[woy - 1] * np.exp(dev[i] - stat_var / 2) for i, r in enumerate(("upper", "lower"))}
    # --- daily files
    days = pd.date_range("2015-01-05", "2025-01-05", freq="D")
    w_idx = np.clip(np.searchsorted(weeks.values[first_price:], days.values, side="right") - 1, 0, nP - 1)
    daily = weekly_price[w_idx] + 0.8 * rng.standard_normal(len(days))
    pd.DataFrame({"date": days, "price_eur_mwh": np.round(np.maximum(daily, 0.5), 2)}).to_csv(OUT / "system_price_daily.csv", index=False)
    print(f"system_price_daily.csv: {len(days)} days, mean {daily.mean():.1f} EUR/MWh")
    days = pd.date_range("1990-01-01", "2024-12-31", freq="D")
    w_idx = np.clip(np.searchsorted(weeks.values, days.values, side="right") - 1, 0, n - 1)
    smooth = lambda x: np.convolve(np.pad(x, 3, mode="edge"), np.ones(7) / 7, mode="valid")   # noqa: E731
    cols = {"date": days}
    for r, stations, shares in (("upper", UPPER_STATIONS, (0.6, 0.15, 0.25)), ("lower", LOWER_STATIONS, (0.9, 0.1))):
        d = smooth(flows[r][w_idx]) * np.exp(0.05 * rng.standard_normal(len(days)))
        for st, sh in zip(stations, shares):
            cols[st] = np.round(sh * d, 3)
    pd.DataFrame(cols).to_csv(OUT / "inflow_daily.csv", index=False)
    up = sum(cols[s] for s in UPPER_STATIONS); lo = sum(cols[s] for s in LOWER_STATIONS)
    print(f"inflow_daily.csv: {len(days)} days; annual mean upper {up.mean() * 365.25 * 86400 / 1e6:.0f}, "
          f"lower {lo.mean() * 365.25 * 86400 / 1e6:.0f} Mm3; price-inflow innovation correlation {rho_p:+.2f}")


def forward_panel(rng):
    """Daily settlement quotes of week, month, quarter and year contracts priced by the
    two-factor model from a daily factor path (trading days 2016-2020)."""
    days = pd.bdate_range("2016-01-04", "2020-12-31")
    phi_d = PRICE.phi_chi ** (1 / 5); sc_d = PRICE.sigma_chi * np.sqrt((1 - phi_d ** 2) / (1 - PRICE.phi_chi ** 2))
    sx_d = PRICE.sigma_xi / np.sqrt(5)
    chi, xi = 0.0, PANEL_LEVEL          # start the panel's path well above zero: log returns need positive prices
    rows = []
    for day in days:
        chi = phi_d * chi + sc_d * rng.standard_normal(); xi = xi + sx_d * rng.standard_normal()
        contracts = []
        y, w, _ = day.isocalendar()
        for j in range(1, 7):                                          # six weeks ahead
            s = pd.Timestamp.fromisocalendar(y, w, 1) + pd.Timedelta(days=7 * j)
            yy, ww, _ = s.isocalendar(); contracts.append((f"ENOW{ww:02d}-{yy % 100:02d}", s, s + pd.Timedelta(days=6)))
        m0 = pd.Timestamp(day.year, day.month, 1)
        for j in range(1, 7):                                          # six months ahead
            s = m0 + pd.offsets.MonthBegin(j); e = s + pd.offsets.MonthEnd(0)
            contracts.append((f"ENOM{s.strftime('%b').upper()}-{s.year % 100:02d}", s, e))
        q0 = pd.Timestamp(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
        for j in range(1, 9):                                          # eight quarters ahead
            s = q0 + pd.offsets.QuarterBegin(j, startingMonth=1); e = s + pd.offsets.QuarterEnd(0)
            contracts.append((f"ENOQ{(s.month - 1) // 3 + 1}-{s.year % 100:02d}", s, e))
        for j in range(1, 4):                                          # three years ahead
            s = pd.Timestamp(day.year + j, 1, 1); contracts.append((f"ENOYR-{s.year % 100:02d}", s, pd.Timestamp(day.year + j, 12, 31)))
        for code, s, e in contracts:
            dd = pd.date_range(s, e, freq="D")
            k = (dd - day).days.to_numpy() / 7.0
            woy = np.clip(dd.isocalendar().week.to_numpy().astype(int), 1, 52)
            F = float(np.mean(xi + PRICE.phi_chi ** k * chi + PRICE.season(woy))) + 0.02 * rng.standard_normal()
            rows.append((day, "Power Base", "Futures", code, e, round(max(F, 0.5), 2)))
    df = pd.DataFrame(rows, columns=["date", "main_category", "instr_type", "series", "expire", "closing_price"])
    df.to_csv(OUT / "forward_panel.csv", index=False)
    print(f"forward_panel.csv: {len(df)} quotes on {len(days)} trading days, {df.series.nunique()} contracts")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    price_and_inflow(rng); forward_panel(rng)
    print("true parameters of the sample:", PRICE)
