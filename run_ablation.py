"""P4: DXY (DTWEXBGS, FRED) forecasting ablation.

All models are fit on the first 80% of the monthly DXY log-return series and
scored on the first out-of-sample horizon points (walk-forward, no leakage).
PatchTST (torch, SDPA) vs ETS (statsmodels) vs LightGBM (recursive) vs the
naive zero-return random walk.

Usage:  python run_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

from patchtst.data.fx_data import load_series
from patchtst.models.patchtst import PatchTST, train_patchtst
from patchtst.models.baselines import random_walk, ets_forecast, lgbm_forecast
from patchtst.eval.ablation import ablation_table, drift_monitor


def main() -> None:
    series = load_series()
    print(f"DXY (DTWEXBGS, monthly): {len(series)} obs, "
          f"{series.index.min().date()} -> {series.index.max().date()}")

    r = series.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    vals = r.to_numpy()
    split = int(len(vals) * 0.8)
    train, val = vals[:split], vals[split:]
    print(f"in-sample {len(train)}, out-of-sample {len(val)} "
          f"(walk-forward split at {series.index[split].date()})")

    horizons = [5, 10]
    actual_by_h, preds_by_h = {}, {}
    patch_models = {}

    for h in horizons:
        actual_by_h[h] = val[:h]
        preds_by_h[h] = {}
        preds_by_h[h]["random_walk"] = np.asarray(random_walk(pd.Series(train), h)).ravel()
        try:
            preds_by_h[h]["ets"] = np.asarray(
                ets_forecast(pd.Series(train), h)).ravel()
        except Exception as e:
            print(f"  ETS h={h} failed: {e}")
        try:
            preds_by_h[h]["lightgbm"] = np.asarray(
                lgbm_forecast(pd.Series(train), h)).ravel()
        except Exception as e:
            print(f"  LGBM h={h} failed: {e}")

        # PatchTST: one-shot forecast from the last 36 in-sample obs
        lookback = min(36, len(train) - 1)
        origin = train[-lookback:]
        model, hist = train_patchtst(train, origin, val[:h],
                                     horizon=h, lookback=lookback)
        with torch.no_grad():
            pred = model(torch.from_numpy(origin).float().unsqueeze(0))[0].numpy()
        preds_by_h[h]["patchtst"] = np.asarray(pred).ravel()
        patch_models[h] = (model, hist)

    tab = ablation_table(actual_by_h, preds_by_h)
    print("\n=== MAE ablation (DXY monthly returns, walk-forward) ===")
    print(tab.to_string(index=False))

    # PSI drift monitor for the PatchTST error distribution
    m5, _ = patch_models[5]
    origin5 = train[-36:]
    with torch.no_grad():
        p5 = m5(torch.from_numpy(origin5).float().unsqueeze(0))[0].numpy()
    err_new = np.abs(p5 - val[:5])
    err_old = np.abs(p5 - train[-5:])
    drift = drift_monitor(err_new, err_old)
    print(f"\nPSI drift monitor (PatchTST, h=5): psi={drift['psi']:.3f} "
          f"(retrain trigger >= 0.25: {drift['triggered']})")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    tab.to_csv(out / "ablation.csv", index=False)
    (out / "drift.json").write_text(__import__("json").dumps(drift, indent=2))
    print(f"\nSaved to {out}/ablation.csv and drift.json")


if __name__ == "__main__":
    main()
