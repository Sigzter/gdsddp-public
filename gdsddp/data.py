"""Data access: spot prices from a public API, inflow files, weekly aggregation, season fit.

Three inputs feed the pipeline:
  1. the Nord Pool SYSTEM day-ahead price (public; fetched from Energinet's open
     "Energi Data Service" which republishes it hourly, no key needed),
  2. inflow per station of the Namsen watercourse (NTE / SINTEF, not public — a
     synthetic sample with the same layout ships in data/sample/),
  3. the Nasdaq Nordic power futures panel for the forward-curve PCA (licensed —
     a synthetic sample ships in data/sample/; see forwardcurve.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"
PROCESSED = ROOT / "data" / "processed"

# Stations of the Namsen dataset that make up the two equivalent reservoirs of the
# thesis: upper = Namsvatn with the Vekteren and Limingen lakes, lower = Tunnsjø with
# the small Tunnsjøflyene buffer. Column names as in the NTE file.
UPPER_STATIONS = ["Namsvatn", "Vekteren", "Limingen"]
LOWER_STATIONS = ["Tunnsjø", "Tunnsjøflyene"]


# ------------------------------------------------------------------ system price (API)
def fetch_system_price(start: str = "2015-01-01", end: str | None = None,
                       cache: Path | None = None, timeout: int = 180) -> pd.Series:
    """Hourly Nord Pool system price in EUR/MWh from Energi Data Service
    (dataset Elspotprices, PriceArea = SYSTEM). Returns a Series indexed by UTC hour.

    ONE request for the whole range (limit=0 returns every row; ten years of hours is
    below the service's 100 000-row cap). The service throttles repeated calls from the
    same address with a 429 and a Retry-After header, so the download is cached as CSV
    when ``cache`` is given and the request is retried at most twice after waiting.
    """
    import requests
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")["price_eur_mwh"]
    url = ("https://api.energidataservice.dk/dataset/Elspotprices"
           f"?start={pd.Timestamp(start):%Y-%m-%dT%H:%M}&end={pd.Timestamp(end):%Y-%m-%dT%H:%M}"
           '&filter={"PriceArea":["SYSTEM"]}&columns=HourUTC,SpotPriceEUR&sort=HourUTC&limit=0')
    for attempt in range(3):
        r = requests.get(url, timeout=timeout)
        if r.status_code != 429:
            break
        wait = min(int(r.headers.get("retry-after", 60)) + 1, 330)
        print(f"  rate limited by the API: waiting {wait} s before retry {attempt + 1}/2")
        time.sleep(wait)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["records"])
    s = pd.Series(df["SpotPriceEUR"].astype(float).values,
                  index=pd.to_datetime(df["HourUTC"]), name="price_eur_mwh").sort_index()
    s = s[~s.index.duplicated()]
    s.index.name = "date"
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_csv(cache)
    return s


def load_price_file(path: Path) -> pd.Series:
    """A price file with columns date and one price column (hourly or daily)."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    s = df.iloc[:, 0].astype(float)
    if getattr(s.index, "tz", None) is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    return s


def weekly_mean(s: pd.Series) -> pd.DataFrame:
    """Weekly mean of a time series with columns [price, woy]; partial edge weeks dropped.
    Weeks end on Sunday; woy is the ISO week of the year clipped to 1..52."""
    w = s.resample("W").mean().to_frame("price")
    w["woy"] = np.clip(w.index.isocalendar().week.to_numpy().astype(int), 1, 52)
    w.index.name = "week_ending"
    return w.iloc[1:-1].dropna()


def fit_season(weekly: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    """Least-squares fit of price_t = c + A cos(2 pi (w_t - w0) / 52) on a window.
    Returns amplitude A, phase w0 (weeks), level c and the residual series."""
    d = weekly if start is None else weekly[weekly.index >= pd.Timestamp(start)]
    d = d if end is None else d[d.index <= pd.Timestamp(end)]
    w = d["woy"].to_numpy(float); y = d["price"].to_numpy(float)
    X = np.column_stack([np.ones_like(w), np.cos(2 * np.pi * w / 52), np.sin(2 * np.pi * w / 52)])
    c, a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    A = float(np.hypot(a, b)); w0 = float(np.arctan2(b, a) * 52 / (2 * np.pi)) % 52
    resid = pd.Series(y - X @ np.array([c, a, b]), index=d.index)
    return {"season_amplitude": A, "season_phase": w0, "level": float(c), "residual": resid}


# --------------------------------------------------------------------------- inflow
def load_station_flows(path: Path) -> pd.DataFrame:
    """Read a station-flow file (date + one column per station, m3/s at any regular time
    step) and return the weekly inflow volume in Mm3 of the two equivalent reservoirs
    with columns [upper, lower, woy]. Partial first/last weeks are dropped."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if getattr(df.index, "tz", None) is not None:
        df = df.tz_convert("UTC").tz_localize(None)
    step_s = float(pd.Series(df.index).diff().median().total_seconds())   # seconds per record
    vol = pd.DataFrame({"upper": df[UPPER_STATIONS].sum(1) * step_s * 1e-6,
                        "lower": df[LOWER_STATIONS].sum(1) * step_s * 1e-6})
    weekly = vol.resample("W").sum()
    weekly["woy"] = np.clip(weekly.index.isocalendar().week.to_numpy().astype(int), 1, 52)
    weekly.index.name = "week_ending"
    return weekly.iloc[1:-1]


def deseasonalised_ar1_residuals(series: pd.Series, woy: np.ndarray) -> pd.Series:
    """Residuals of a per-week-of-year AR(1) on the series (used for the price–inflow
    innovation correlation, so both sides are 'news' rather than levels)."""
    y = series.to_numpy(float); x = np.concatenate([[y[0]], y[:-1]])
    res = np.zeros_like(y)
    for k in range(1, 53):
        sel = woy == k
        if sel.sum() > 3 and x[sel].var() > 0:
            phi = np.cov(x[sel], y[sel], bias=True)[0, 1] / x[sel].var()
            mu = y[sel].mean() - phi * x[sel].mean()
            res[sel] = y[sel] - mu - phi * x[sel]
        else:
            res[sel] = y[sel] - y[sel].mean()
    return pd.Series(res, index=series.index)


def save_json(obj: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
