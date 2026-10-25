import numpy as np
import pandas as pd

from patchtst.eval.ablation import psi, drift_monitor, mae
from patchtst.models.patchtst import PatchTST
import torch


def test_psi_identical_distributions_is_zero():
    x = np.random.default_rng(0).normal(0, 1, 500)
    assert psi(x, x) < 1e-6


def test_psi_shifted_distributions_large():
    old = np.random.default_rng(0).normal(0, 1, 500)
    new = np.random.default_rng(1).normal(2, 1, 500)
    assert psi(new, old) > 0.5


def test_drift_monitor_trigger():
    old = np.random.default_rng(0).normal(0, 1, 200)
    new = np.random.default_rng(1).normal(3, 1, 200)
    res = drift_monitor(new, old)
    assert res["triggered"] is True


def test_mae_positive_and_zero_on_self():
    a = np.array([1.0, 2.0, 3.0])
    assert mae(a, a) == 0.0
    assert mae(a, a + 1) == 1.0


def test_patchtst_forward_shape():
    m = PatchTST(lookback=36, horizon=5)
    out = m(torch.randn(2, 36))
    assert out.shape == (2, 5)


def test_patchtst_deterministic_same_seed():
    """Same seed + same data -> identical forecast (CPU torch is seed-deterministic;
    tolerance is 1e-5, not bitwise, to stay stable across CPU kernel builds)."""
    from patchtst.models.patchtst import train_patchtst

    rng = np.random.default_rng(3)
    train = rng.normal(0, 0.01, 200).astype(np.float32)
    origin, hold = train[-36:].astype(np.float32), np.zeros(5, dtype=np.float32)

    def run(seed):
        m, _ = train_patchtst(train, origin, hold,
                              horizon=5, lookback=36, epochs=2, seed=seed)
        return m(torch.from_numpy(origin).unsqueeze(0))[0].detach().numpy()

    p1, p2 = run(7), run(7)
    assert np.allclose(p1, p2, atol=1e-5)

    p3 = run(8)
    assert not np.allclose(p1, p3, atol=1e-4)  # different seed -> different model
