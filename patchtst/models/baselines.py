"""Classical baselines for the DXY forecast ablation.

All deterministic (fixed seeds).  The ablation table compares each
baseline against PatchTST on the same split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(hist: pd.Series, horizon: int = 5) -> pd.Series:
    """Naive return forecast: zero per step (the random-walk null for a
    returns target). This is the honest baseline every model must beat."""
    return pd.Series(np.zeros(horizon), index=range(horizon))


def ets_forecast(hist: pd.Series, horizon: int = 5,
                 seasonal_period: int = 1) -> pd.Series:
    """ETS (Holt-Winters) forecast from statsmodels, deterministic."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit_kwargs = dict(trend=None, damped_trend=False)
    if seasonal_period > 1 and len(hist) >= 2 * seasonal_period:
        fit_kwargs.update(seasonal="additive",
                          seasonal_periods=seasonal_period)
    else:
        fit_kwargs.update(seasonal=None)
    model = ExponentialSmoothing(hist, **fit_kwargs)
    fitted = model.fit()
    pred = fitted.forecast(horizon)
    return pd.Series(np.asarray(pred, dtype=float), index=range(horizon))


def lgbm_forecast(hist: pd.Series, horizon: int = 5,
                   lookbacks: tuple[int, ...] = (1, 3, 6, 12)) -> pd.Series:
    """LightGBM regressor on lag features; deterministic seed."""
    import lightgbm as lgb

    X, y = [], []
    series = np.asarray(hist, dtype=float)
    n = len(series)
    for t in range(max(lookbacks), n):
        feats = [series[t - lb] for lb in lookbacks]
        feats.append(t / 252.0)  # time-since-start (mild seasonality cue)
        X.append(feats)
        y.append(series[t])
    X = np.array(X)
    y = np.array(y)
    model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, seed=42,
                              verbose=-1)
    model.fit(X, y)
    # recursive forecast: use own predictions to build next lags
    tail = list(series[-max(lookbacks):])
    out = []
    for h in range(horizon):
        t = max(lookbacks) + len(out)
        feats = [tail[-lb] for lb in lookbacks]
        feats.append(t / 252.0)
        pred = float(np.ravel(model.predict(np.asarray([feats], dtype=float)))[0])
        out.append(pred)
        tail.append(pred)
    return pd.Series(out, index=range(horizon))
