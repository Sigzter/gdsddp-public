"""Forward-curve construction and PCA, used to calibrate the price model.

Pipeline (notebooks/03_forward_curve_pca.ipynb):
  1. a panel of daily settlement prices per futures contract (week, month, quarter,
     year products on the Nord Pool system price) is turned into a WEEKLY CURVE per
     trading day, in two ways: the BLOCK curve (for every week k = 1..104 ahead, the
     price of the shortest contract that covers that week) and the SMOOTH curve (the
     least-curvature weekly curve that reprices all quoted contracts, build_smooth_curve);
  2. the curve is sampled on the last trading day of each ISO week and FIXED-DELIVERY
     log returns are formed: the same delivery week compared on two consecutive
     sampling days, r[t, k] = log F[t, k] - log F[t-1, k+1];
  3. the covariance matrix of the returns across maturities is decomposed with
     scikit-learn's PCA and
     a Schwartz-Smith two-factor covariance is fitted to it:
         Cov(r_k, r_j) = sigma_xi^2 + sigma_chi^2 phi^(k+j)
     which gives phi_chi, sigma_chi and sigma_xi (in log units per week; multiplied
     by a reference price level they become the EUR/MWh parameters of models.PriceModel).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

K_MAX = 104
_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
_MONTH_NO = {m: i + 1 for i, m in enumerate(_MONTHS)}
_PAT = re.compile(r"^(?P<prefix>[A-Z]+?)(?:(?P<week>W\d{2})|(?P<quarter>Q\d)|(?P<year>YR)|(?P<month>"
                  + "|".join(_MONTHS) + r"))-(?P<yy>\d{2})$")


def parse_series(code: str) -> dict | None:
    """Delivery window from a Nasdaq-style contract code: ENOW05-16 (week), ENOMSEP-17
    (month), ENOQ1-18 (quarter), ENOYR-20 (year); also the newer ENOFUTBL... family."""
    m = _PAT.match(code)
    if m is None:
        return None
    g = m.groupdict(); yy = 2000 + int(g["yy"])
    if g["week"]:
        start = pd.Timestamp.fromisocalendar(yy, int(g["week"][1:]), 1); end = start + pd.Timedelta(days=6); typ = "W"
    elif g["quarter"]:
        q = int(g["quarter"][1]); start = pd.Timestamp(yy, 3 * q - 2, 1); end = start + pd.offsets.QuarterEnd(0); typ = "Q"
    elif g["year"]:
        start, end, typ = pd.Timestamp(yy, 1, 1), pd.Timestamp(yy, 12, 31), "Y"
    else:
        start = pd.Timestamp(yy, _MONTH_NO[g["month"]], 1); end = start + pd.offsets.MonthEnd(0); typ = "M"
    return {"series": code, "ctype": typ, "delivery_start": start, "delivery_end": end,
            "delivery_days": (end - start).days + 1}


def load_panel(path: Path) -> pd.DataFrame:
    """Load a contract panel (columns date, series, closing_price, optionally
    main_category / instr_type) and attach the parsed delivery windows of the
    system-price base-load futures."""
    df = pd.read_csv(path, parse_dates=["date"])
    if "instr_type" in df:
        df = df[df["instr_type"] != "Options"]
    if "main_category" in df:
        df = df[df["main_category"] == "Power Base"]
    df = df[df["closing_price"] > 0]
    meta = pd.DataFrame([r for r in (parse_series(s) for s in df["series"].unique()) if r is not None])
    d = df.merge(meta, on="series", how="inner")
    # one quote per (day, type, window): prefer the current cash-settled family (FUTBL)
    d["rank"] = (~d["series"].str.contains("FUTBL")).astype(int)
    d = (d.sort_values(["date", "ctype", "delivery_start", "rank"])
           .drop_duplicates(["date", "ctype", "delivery_start", "delivery_end"], keep="first"))
    return d.drop(columns="rank")[["date", "series", "ctype", "delivery_start", "delivery_end", "delivery_days", "closing_price"]]


def build_block_curve(d: pd.DataFrame, k_max: int = K_MAX):
    """Weekly curve per trading day: for week k ahead the shortest contract TYPE whose
    contracts cover the whole week (W > M > Q > Y); a week split between two contracts
    of that type gets their day-weighted average. Returns (prices, ctypes) with the
    trading days as rows and k = 1..k_max as columns."""
    dates = np.sort(d["date"].unique())
    px = np.full((len(dates), k_max), np.nan); tp = np.full((len(dates), k_max), "", dtype=object)
    blocks = dict(tuple(d.sort_values("date").groupby("date", sort=False)))
    monday = pd.DatetimeIndex(dates) - pd.to_timedelta(pd.DatetimeIndex(dates).weekday, unit="D")
    mon_i = monday.values.astype("datetime64[D]").astype(np.int64)
    ks = np.arange(1, k_max + 1)
    for i, day in enumerate(dates):
        b = blocks[pd.Timestamp(day)]
        ds = b["delivery_start"].values.astype("datetime64[D]").astype(np.int64)
        de = b["delivery_end"].values.astype("datetime64[D]").astype(np.int64)
        price = b["closing_price"].values; ctype = b["ctype"].values
        wk_s = mon_i[i] + 7 * ks; wk_e = wk_s + 6
        filled = np.zeros(k_max, bool)
        for typ in ("W", "M", "Q", "Y"):
            sel = ctype == typ
            if not sel.any():
                continue
            ov = np.minimum(de[sel][None, :], wk_e[:, None]) - np.maximum(ds[sel][None, :], wk_s[:, None]) + 1
            np.clip(ov, 0, None, out=ov)
            ok = (ov.sum(1) == 7) & ~filled
            if ok.any():
                rows = np.where(ok)[0]
                px[i, rows] = (ov[rows] / 7.0) @ price[sel]; tp[i, rows] = typ; filled[rows] = True
    idx = pd.DatetimeIndex(dates, name="date"); cols = pd.Index(ks, name="k")
    return pd.DataFrame(px, index=idx, columns=cols), pd.DataFrame(tp, index=idx, columns=cols).replace("", np.nan)


def build_smooth_curve(d: pd.DataFrame, k_max: int = K_MAX, smoothness: float = 1.0, grid_extra: int = 60):
    """Maximum-smoothness weekly curve per trading day (Benth, Koekebakker & Ollmar 2007, discrete form).

    The block curve hands one price to every week of a contract, so it is a staircase. Here the
    curve f(k), k = 1..k_max (+ grid_extra weeks beyond, so that contracts reaching past the horizon
    still pin the end), is the solution of

        min_f  sum_j w_j (A_j f - p_j)^2  +  smoothness * sum_k (f_{k+1} - 2 f_k + f_{k-1})^2

    where A_j f is the day-weighted average of f over the delivery weeks of contract j and p_j its
    price. The first term reprices the quoted contracts (weight w_j = 1 for weeks, months, quarters,
    0.25 for years, whose single price would otherwise pull a whole year flat), the second
    penalises curvature. ``smoothness`` in (EUR/MWh)^-2 units: larger = smoother, less exact
    repricing. Returns (prices, ctypes) like build_block_curve; ctypes is "S" everywhere (no
    instrument changes to mask in the returns)."""
    dates = np.sort(d["date"].unique())
    G = k_max + grid_extra
    D2 = np.zeros((G - 2, G))
    for i in range(G - 2):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    P2 = smoothness * (D2.T @ D2)
    px = np.full((len(dates), k_max), np.nan)
    blocks = dict(tuple(d.sort_values("date").groupby("date", sort=False)))
    monday = pd.DatetimeIndex(dates) - pd.to_timedelta(pd.DatetimeIndex(dates).weekday, unit="D")
    mon_i = monday.values.astype("datetime64[D]").astype(np.int64)
    ks = np.arange(1, G + 1)
    type_w = {"W": 1.0, "M": 1.0, "Q": 1.0, "Y": 0.25}
    for i, day in enumerate(dates):
        b = blocks[pd.Timestamp(day)]
        ds = b["delivery_start"].values.astype("datetime64[D]").astype(np.int64)
        de = b["delivery_end"].values.astype("datetime64[D]").astype(np.int64)
        wk_s = mon_i[i] + 7 * ks; wk_e = wk_s + 6
        ov = np.minimum(de[None, :], wk_e[:, None]) - np.maximum(ds[None, :], wk_s[:, None]) + 1   # (G, n_contracts) days of overlap
        np.clip(ov, 0, None, out=ov)
        cover = ov.sum(0)
        keep = cover > 0
        if keep.sum() < 3:
            continue
        A = (ov[:, keep] / cover[keep]).T                      # (n, G): day weights, rows sum to 1
        w = np.array([type_w[t] for t in b["ctype"].values[keep]])
        p = b["closing_price"].values[keep]
        M = A.T @ (w[:, None] * A) + P2
        f = np.linalg.solve(M, A.T @ (w * p))
        px[i] = f[:k_max]
    idx = pd.DatetimeIndex(dates, name="date"); cols = pd.Index(np.arange(1, k_max + 1), name="k")
    prices = pd.DataFrame(px, index=idx, columns=cols)
    ctypes = pd.DataFrame("S", index=idx, columns=cols).where(prices.notna())
    return prices, ctypes


def repricing_error(d: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per contract type: RMSE between quoted prices and the curve's day-weighted average over the
    delivery weeks (for contracts fully inside the curve's horizon)."""
    k_max = prices.shape[1]; rows = []
    monday = prices.index - pd.to_timedelta(prices.index.weekday, unit="D")
    mon = dict(zip(prices.index, monday.values.astype("datetime64[D]").astype(np.int64)))
    for day, b in d.groupby("date"):
        if day not in mon:
            continue
        f = prices.loc[day].values
        ds = b["delivery_start"].values.astype("datetime64[D]").astype(np.int64)
        de = b["delivery_end"].values.astype("datetime64[D]").astype(np.int64)
        wk_s = mon[day] + 7 * np.arange(1, k_max + 1); wk_e = wk_s + 6
        ov = np.clip(np.minimum(de[None, :], wk_e[:, None]) - np.maximum(ds[None, :], wk_s[:, None]) + 1, 0, None)
        full = ov.sum(0) == b["delivery_days"].values
        for j in np.where(full)[0]:
            w = ov[:, j] / ov[:, j].sum()
            if np.isfinite(f[w > 0]).all():
                rows.append((b["ctype"].values[j], float(w @ f - b["closing_price"].values[j])))
    df = pd.DataFrame(rows, columns=["ctype", "error"])
    return df.groupby("ctype")["error"].agg(rmse=lambda e: float(np.sqrt(np.mean(e ** 2))), n="size")


def weekly_fixed_delivery_returns(prices: pd.DataFrame, ctypes: pd.DataFrame) -> pd.DataFrame:
    """Sample the curve on the last trading day of each ISO week and compute the log
    return of each FIXED delivery week: column k today against column k + d a week ago,
    where d is the number of weeks the grid advanced (1 for consecutive weeks). Cells
    where the contract type behind the two prices differs are masked (a month price
    against a quarter price is a change of instrument, not a price move)."""
    iso = pd.Series([f"{d.isocalendar()[0]}-{d.isocalendar()[1]:02d}" for d in prices.index], index=prices.index)
    last = pd.DatetimeIndex(sorted(iso.groupby(iso).apply(lambda s: s.index.max()).values))
    W = prices.loc[last]; S = ctypes.loc[last].values.astype(object)
    P = np.log(W.values); K = P.shape[1]
    monday = W.index - pd.to_timedelta(W.index.weekday, unit="D")
    step = np.diff(monday.values).astype("timedelta64[D]").astype(int) // 7
    out = np.full_like(P, np.nan)
    for i, d in enumerate(step):
        r = i + 1
        if 0 < d < K:
            out[r, :K - d] = P[r, :K - d] - P[r - 1, d:]
            roll = (S[r, :K - d] != S[r - 1, d:]) & pd.notna(S[r, :K - d]) & pd.notna(S[r - 1, d:])
            out[r, :K - d][roll] = np.nan
    dmax = int(step.max())
    return pd.DataFrame(out[1:, :K - dmax], index=W.index[1:], columns=np.arange(1, K - dmax + 1))


def pca(R: pd.DataFrame, min_obs: int = 100, max_row_missing: float = 0.2):
    """Principal components of the return matrix with scikit-learn.

    PCA needs a complete matrix: maturities observed on fewer than ``min_obs`` weeks are
    dropped, weeks with more than ``max_row_missing`` of the maturities missing are dropped,
    and the remaining gaps are filled with the column mean (a filled cell contributes no
    variance, which slightly understates the leading components). The components are
    computed on the COVARIANCE (not the correlation) so that a volatile front week counts
    more than a quiet far maturity, as in the forward-curve literature.

    Returns dict(k, cov, eigenvalues, share, loadings (K x 3, PC1 positive at the front),
    vol (per week), n_weeks, scores (first 3 components per week))."""
    from sklearn.decomposition import PCA
    R = R.loc[:, R.notna().sum() >= min_obs]
    R = R[R.isna().mean(1) <= max_row_missing]
    X = R.fillna(R.mean()).to_numpy()
    model = PCA(n_components=None).fit(X)          # all components: get_covariance() is the sample covariance
    S = model.get_covariance()
    vec = model.components_[:3].T.copy()
    for j in range(3):
        if vec[0, j] < 0:
            vec[:, j] *= -1
    ev = model.explained_variance_
    return {"k": R.columns.to_numpy(float), "cov": S, "eigenvalues": ev, "share": model.explained_variance_ratio_,
            "loadings": vec, "vol": np.sqrt(np.diag(S)), "n_weeks": int(len(R)),
            "scores": pd.DataFrame(model.transform(X)[:, :3], index=R.index, columns=["PC1", "PC2", "PC3"]), "sklearn": model}


def model_cov(theta, k):
    """Two-factor covariance of the k-week-ahead forward returns:
    sigma_xi^2 + sigma_chi^2 phi^(k_i + k_j);  theta = (log sigma_xi, log sigma_chi, logit phi)."""
    s_xi, s_chi = np.exp(theta[0]), np.exp(theta[1]); phi = 1.0 / (1.0 + np.exp(-theta[2]))
    return s_xi ** 2 + s_chi ** 2 * phi ** (k[:, None] + k[None, :])


def fit_two_factor(S: np.ndarray, k: np.ndarray) -> dict:
    """Weighted least squares of model_cov on the empirical covariance (weights 1/(sd_i sd_j)
    so that correlations and variances are fitted alike). Returns the parameters in log
    units per week plus fit diagnostics."""
    iu = np.triu_indices(len(k))
    w = 1.0 / np.sqrt(np.outer(np.diag(S), np.diag(S)))[iu]
    best = None
    for x0 in ([np.log(0.01), np.log(0.05), 3.0], [np.log(0.02), np.log(0.1), 1.0], [np.log(0.005), np.log(0.03), 4.0]):
        r = least_squares(lambda th: (model_cov(th, k) - S)[iu] * w, np.array(x0), method="trf", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    th = best.x; C = model_cov(th, k)
    phi = 1.0 / (1.0 + np.exp(-th[2]))
    return {"sigma_xi": float(np.exp(th[0])), "sigma_chi": float(np.exp(th[1])), "phi_chi": float(phi),
            "halflife_weeks": float(np.log(0.5) / np.log(phi)), "model_cov": C,
            "trace_share": float(np.trace(C) / np.trace(S)),
            "vol_rmse_annual_pct": float(np.sqrt(np.mean((np.sqrt(np.diag(C)) - np.sqrt(np.diag(S))) ** 2)) * np.sqrt(52) * 100)}
