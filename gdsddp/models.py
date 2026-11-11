"""Stochastic models: the spot price and the two reservoir inflows.

PRICE — Schwartz & Smith (2000) two-factor model, written additively in EUR/MWh
and in weekly steps (as in the project thesis, section 5.2.1):

    P_t   = xi_t + chi_t + s(w_t)                       spot price in week t
    chi_t = phi_chi * chi_{t-1} + sigma_chi * eps_chi    short-term deviation (mean-reverting)
    xi_t  = xi_{t-1}            + sigma_xi  * eps_xi     long-term level (random walk)
    s(w)  = A * cos(2 pi (w - w0) / 52)                  deterministic season, w = week of year

The two innovations are independent standard normals. The parameters
(phi_chi, sigma_chi, sigma_xi) are estimated from the forward curve in
notebooks/03_forward_curve_pca.ipynb; the season (A, w0) and the starting level
xi_0 from the spot history in notebooks/01_price_data_api.ipynb.

INFLOW — one periodic AR(1) per reservoir on the log of the weekly inflow
(section 5.2.2 of the thesis, extended to two reservoirs):

    log I_{r,t} = mu_{r,w} + phi_{r,w} * log I_{r,t-1} + sigma_{r,w} * e_{r,t},   r in {upper, lower}

with one parameter triple per week of the year w and corr(e_upper, e_lower) = rho_w.
Optionally the inflow innovations are correlated with the price innovation
(rho_price < 0: wet weeks tend to be cheap weeks). Parameters are fitted by
ordinary least squares per week in notebooks/02_namsen_inflow_analysis.ipynb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

WEEKS = 52


def week_of_year(t: int, start_week: int = 1) -> int:
    """Week of the year (1..52) of stage t (0-based) when stage 0 is calendar week ``start_week``."""
    return (start_week - 1 + t) % WEEKS + 1


# ----------------------------------------------------------------------------- price
@dataclass
class PriceModel:
    phi_chi: float = 0.977          # weekly mean reversion of the short-term factor
    sigma_chi: float = 2.292        # EUR/MWh per week
    sigma_xi: float = 0.554         # EUR/MWh per week
    season_amplitude: float = 4.421  # A, EUR/MWh
    season_phase: float = 2.792     # w0, weeks (the season peaks at w = w0)
    xi0: float = 30.0               # long-term level at the start of the horizon, EUR/MWh
    chi0: float = 0.0               # short-term deviation at the start (0 = its mean)

    def season(self, weeks) -> np.ndarray:
        """Deterministic seasonal component s(w) for week-of-year ``weeks`` (1..52)."""
        w = np.asarray(weeks, float)
        return self.season_amplitude * np.cos(2 * np.pi * (w - self.season_phase) / WEEKS)

    def simulate(self, n_paths: int, T: int, rng: np.random.Generator, start_week: int = 1, return_all: bool = False):
        """Simulate the two factors for T weekly stages.

        Returns (chi, xi, season, eps_total): chi and xi are (n_paths, T); season is (T,)
        and depends only on the calendar; eps_total is the standardised innovation of the
        total price change in each week, used to correlate the inflows with the price.
        With ``return_all`` the factor innovations eps_chi and eps_xi are appended.
        The spot price is max(0, xi + chi + season)."""
        eps_chi = rng.standard_normal((n_paths, T))
        eps_xi = rng.standard_normal((n_paths, T))
        chi = np.zeros((n_paths, T))
        xi = np.zeros((n_paths, T))
        chi_prev = np.full(n_paths, self.chi0)
        xi_prev = np.full(n_paths, self.xi0)
        for t in range(T):
            chi_prev = self.phi_chi * chi_prev + self.sigma_chi * eps_chi[:, t]
            xi_prev = xi_prev + self.sigma_xi * eps_xi[:, t]
            chi[:, t] = chi_prev
            xi[:, t] = xi_prev
        weeks = np.array([week_of_year(t, start_week) for t in range(T)])
        sc, sx = self.sigma_chi, self.sigma_xi
        eps_total = (sc * eps_chi + sx * eps_xi) / np.sqrt(sc * sc + sx * sx)
        if return_all:
            return chi, xi, self.season(weeks), eps_total, eps_chi, eps_xi
        return chi, xi, self.season(weeks), eps_total

    def expected_price(self, chi_t: float, xi_t: float, week_t: int, k: int) -> float:
        """E[P_{t+k} | chi_t, xi_t]: the model-implied forward price k weeks ahead.
        chi decays as phi^k, xi stays, the season is known."""
        w = week_of_year(k, start_week=week_t)
        return xi_t + self.phi_chi ** k * chi_t + float(self.season(w))

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "PriceModel":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------------------- inflow
@dataclass
class InflowModel:
    mu: np.ndarray = field(default=None)        # (2, 52) intercepts of the log-AR(1), rows = upper, lower
    phi: np.ndarray = field(default=None)       # (2, 52) autoregressive coefficients
    sigma: np.ndarray = field(default=None)     # (2, 52) residual standard deviations
    rho: np.ndarray = field(default=None)       # (52,)   correlation of the two residuals
    log_mean: np.ndarray = field(default=None)  # (2, 52) mean log inflow per week (start of the simulation)
    n_years: int = 0

    @classmethod
    def fit(cls, weekly: pd.DataFrame, rho_fixed: float | None = None) -> "InflowModel":
        """Fit the periodic AR(1) by OLS per week of the year.

        ``weekly`` must have the columns ``upper``, ``lower`` (Mm3/week) and ``woy``
        (week of the year, 1..52) in chronological order. For every week w the pairs
        (log I_{t-1}, log I_t) with t in week w give one regression; the residual
        correlation between the two reservoirs is computed per week as well."""
        eps = 1e-6
        woy = weekly["woy"].to_numpy(int)
        logs = [np.log(weekly["upper"].to_numpy(float) + eps), np.log(weekly["lower"].to_numpy(float) + eps)]
        mu = np.zeros((2, WEEKS)); phi = np.zeros((2, WEEKS)); sig = np.zeros((2, WEEKS)); lm = np.zeros((2, WEEKS))
        resid = [np.zeros(len(woy)), np.zeros(len(woy))]
        for r, y_all in enumerate(logs):
            x_all = np.concatenate([[y_all[0]], y_all[:-1]])      # last week's log inflow
            for k in range(WEEKS):
                sel = woy == k + 1
                y, x = y_all[sel], x_all[sel]
                lm[r, k] = y.mean()
                if len(y) > 2 and x.var() > 1e-12:
                    phi_k = float(np.clip(np.cov(x, y, bias=True)[0, 1] / x.var(), -0.99, 0.99))
                    mu_k = y.mean() - phi_k * x.mean()
                    res = y - (mu_k + phi_k * x)
                    sig_k = float(np.std(res, ddof=1))
                else:                                            # degenerate week: constant
                    phi_k, mu_k = 0.0, y.mean(); res = y - mu_k; sig_k = float(np.std(y, ddof=1))
                mu[r, k], phi[r, k], sig[r, k] = mu_k, phi_k, max(sig_k, 1e-4)
                resid[r][sel] = res
        rho = np.zeros(WEEKS)
        for k in range(WEEKS):
            sel = woy == k + 1
            a, b = resid[0][sel], resid[1][sel]
            rho[k] = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else 0.0
        if rho_fixed is not None:
            rho[:] = rho_fixed
        model = cls(mu=mu, phi=phi, sigma=sig, rho=np.clip(rho, -1, 1), log_mean=lm, n_years=int(len(woy) // WEEKS))
        model.residuals = pd.DataFrame({"upper": resid[0], "lower": resid[1]}, index=weekly.index)
        return model

    def simulate(self, n_paths: int, T: int, rng: np.random.Generator, start_week: int = 1,
                 eps_price: np.ndarray | None = None, rho_price: float = 0.0) -> np.ndarray:
        """Simulate (n_paths, T, 2) weekly inflows (upper, lower) in Mm3/week.

        Each week draws a correlated pair of standard normals (correlation rho_w). With
        ``eps_price`` (n_paths, T) and rho_price != 0 both inflow innovations get the common
        component rho_price * eps_price; the pair's own correlation is adjusted so that the
        upper/lower correlation stays rho_w and every innovation keeps unit variance."""
        out = np.zeros((n_paths, T, 2))
        k0 = week_of_year(0, start_week) - 1
        prev = np.stack([np.full(n_paths, self.log_mean[0, k0 - 1]), np.full(n_paths, self.log_mean[1, k0 - 1])])
        couple = eps_price is not None and rho_price != 0.0
        rho2 = rho_price * rho_price
        s_own = np.sqrt(1.0 - rho2) if couple else 1.0
        for t in range(T):
            k = week_of_year(t, start_week) - 1
            z = rng.standard_normal((2, n_paths))
            r = float(self.rho[k])
            if couple:
                r = float(np.clip((r - rho2) / (1.0 - rho2), -1.0, 1.0))
            e_up = z[0]
            e_lo = r * z[0] + np.sqrt(max(0.0, 1.0 - r * r)) * z[1]
            if couple:
                e_up = rho_price * eps_price[:, t] + s_own * e_up
                e_lo = rho_price * eps_price[:, t] + s_own * e_lo
            logs = np.stack([self.mu[0, k] + self.phi[0, k] * prev[0] + self.sigma[0, k] * e_up,
                             self.mu[1, k] + self.phi[1, k] * prev[1] + self.sigma[1, k] * e_lo])
            out[:, t, 0] = np.exp(logs[0]); out[:, t, 1] = np.exp(logs[1])
            prev = logs
        return out

    def to_json(self, path: Path) -> None:
        d = {"mu": self.mu.tolist(), "phi": self.phi.tolist(), "sigma": self.sigma.tolist(),
             "rho": self.rho.tolist(), "log_mean": self.log_mean.tolist(), "n_years": self.n_years}
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def from_json(cls, path: Path) -> "InflowModel":
        d = json.loads(Path(path).read_text())
        return cls(mu=np.array(d["mu"]), phi=np.array(d["phi"]), sigma=np.array(d["sigma"]),
                   rho=np.array(d["rho"]), log_mean=np.array(d["log_mean"]), n_years=int(d.get("n_years", 0)))


# ----------------------------------------------------------------------- joint paths
def simulate_paths(price: PriceModel, inflow: InflowModel, n_paths: int, T: int, seed: int = 42,
                   rho_price_inflow: float = 0.0, start_week: int = 1, couple_to: str = "total") -> dict[str, np.ndarray]:
    """Simulate the joint price/inflow paths that the lattice is built from.

    rho_price_inflow is the correlation between the inflow innovations and the price innovation
    chosen by ``couple_to``: "total" (the week's total price change, as measured in the thesis)
    or "xi" (the long-term factor only: a wet week lowers the price LEVEL permanently but does
    not touch the short-term noise; note that under literature parameters a weekly price change
    is only about one fifth long-term factor, so the implied weekly correlation is much smaller).

    Returns a dict of (n_paths, T) arrays:
        price_chi      the short-term price factor
        price_xi       the long-term factor PLUS the season (so that price = chi + xi)
        price          the spot price, floored at 0
        inflow_upper   Mm3/week
        inflow_lower   Mm3/week
    """
    rng = np.random.default_rng(seed)
    chi, xi, season, eps_total, eps_chi, eps_xi = price.simulate(n_paths, T, rng, start_week, return_all=True)
    if couple_to not in ("total", "xi"):
        raise ValueError("couple_to must be 'total' or 'xi'")
    I = inflow.simulate(n_paths, T, rng, start_week, eps_price=eps_total if couple_to == "total" else eps_xi,
                        rho_price=rho_price_inflow)
    return {"price_chi": chi, "price_xi": xi + season[None, :],
            "price": np.maximum(chi + xi + season[None, :], 0.0),
            "inflow_upper": I[:, :, 0], "inflow_lower": I[:, :, 1]}
