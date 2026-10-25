"""DXY (DTWEXBGS, FRED keyless CSV) + yfinance FX fallback, cached.

The P4 study uses one series — the US Dollar Index — daily, 2015-2026,
as a clean single-asset-class forecasting benchmark.  Cached to parquet so
the notebook is deterministic offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
CACHE = HERE / "data" / "cache"
SERIES = "DTWEXBGS"
START = "2015-01-01"


def _fetch_fred() -> pd.Series:
    import urllib.request

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    import io

    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    idx = pd.to_datetime(raw[raw.columns[0]], errors="coerce")
    vals = pd.to_numeric(raw[raw.columns[1]], errors="coerce").to_numpy()
    out = pd.Series(vals, index=idx, name=SERIES)
    return out[~out.index.isna() & ~out.isna()].sort_index()


def load_series(refetch: bool = False) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "dxy.parquet"
    if cache_file.exists() and not refetch:
        s = pd.read_parquet(cache_file)
        s = s[SERIES] if isinstance(s, pd.DataFrame) else s
    else:
        s = _fetch_fred()
        s = s[s.index >= pd.Timestamp(START)]
        s = s.resample("ME").last().dropna()  # monthly closes for the study
        s.to_frame().to_parquet(cache_file)  # pandas 3.x: Series.to_parquet removed
    return s


def returns_of(series: pd.Series) -> pd.Series:
    """Simple daily-ish pct returns of the monthly DXY series (the forecast target)."""
    r = series.pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna()
