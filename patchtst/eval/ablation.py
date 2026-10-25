"""Ablation harness: horizon x model MAE table + PSI drift monitor.

PSI (Population Stability Index) compares the distribution of one model's
forecast errors on two windows; PSI >= 0.25 flags a retraining trigger
(per the pytorch-peft-mlops skill).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def psi(actual: np.ndarray, expected: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index in [0, ~1]; 0 = identical, >=0.25 = drift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    a, b = np.digitize(expected, edges[1:-1]), np.digitize(actual, edges[1:-1])
    a = np.clip(a, 0, len(edges) - 2)
    b = np.clip(b, 0, len(edges) - 2)
    p1 = np.bincount(a, minlength=len(edges) - 1) / max(len(a), 1)
    p2 = np.bincount(b, minlength=len(edges) - 1) / max(len(b), 1)
    p1 = np.clip(p1, 1e-4, None)
    p2 = np.clip(p2, 1e-4, None)
    return float(np.sum((p2 - p1) * np.log(p2 / p1)))


def ablation_table(
    actual_by_horizon: dict[int, np.ndarray],
    preds_by_horizon: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """actual_by_horizon: {h: actual[h]}; preds_by_horizon: {h: {model: pred[h]}}."""
    rows = []
    for h, acts in actual_by_horizon.items():
        for model, preds in preds_by_horizon.get(h, {}).items():
            rows.append({"horizon": h, "model": model,
                         "mae": mae(acts, preds),
                         "psi_vs_random_walk": 0.0})
    df = pd.DataFrame(rows)
    # PSI of each model's error distribution vs a random-walk baseline
    for i, row in df.iterrows():
        h, model = row["horizon"], row["model"]
        if model != "random_walk" and "random_walk" in preds_by_horizon.get(h, {}):
            # compare the *level* forecast distributions
            base = preds_by_horizon[h]["random_walk"]
            df.loc[i, "psi_vs_random_walk"] = psi(
                np.asarray(preds_by_horizon[h][model]), base)
    return df


def drift_monitor(errors_new: np.ndarray, errors_old: np.ndarray,
                  threshold: float = 0.25) -> dict:
    """PSI between old and new error windows; flags retraining trigger."""
    p = psi(errors_new, errors_old)
    return {"psi": p, "triggered": bool(p >= threshold),
            "threshold": threshold}
