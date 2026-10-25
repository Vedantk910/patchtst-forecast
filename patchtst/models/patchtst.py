"""Minimal PatchTST for univariate time-series forecasting.

Design rules (from the pytorch-peft-mlops skill):
  - SDPA attention (nn.TransformerEncoder uses F.scaled_dot_product_attention)
  - optimizer.zero_grad(set_to_none=True)
  - AdamW + early stopping on val MAE, deterministic seed
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int = 4, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) with L % patch_len == 0 -> (B, n_patches, d_model)
        p = self.proj.in_features  # == patch_len
        return self.proj(x.unfold(1, p, p))


class PatchTST(nn.Module):
    """Patch tokens -> positional embed -> N TransformerEncoder layers ->
    pool over tokens -> forecast horizon output."""

    def __init__(self, lookback: int = 36, horizon: int = 5,
                 patch_len: int = 4, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2):
        super().__init__()
        assert lookback % patch_len == 0, "lookback must be a multiple of patch_len"
        self.patch_len = patch_len
        self.embed = PatchEmbedding(patch_len, d_model)
        n_patches = lookback // patch_len
        self.pos = nn.Parameter(torch.zeros(n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> (B, horizon)
        tok = self.embed(x) + self.pos
        tok = self.encoder(tok)
        return self.head(tok.mean(dim=1))


def train_patchtst(
    train: np.ndarray,
    origin: np.ndarray,
    holdout: np.ndarray,
    horizon: int = 5,
    lookback: int = 36,
    epochs: int = 80,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[nn.Module, list[float]]:
    """Deterministic walk-forward trainer with one-shot validation.

    train:   history used to build (lookback -> next h) training windows
    origin:  the last `lookback` obs of the in-sample region (the forecast
             origin; shape must equal lookback)
    holdout: the next obs after the origin; the first `horizon` points are
             the one-shot validation target (out-of-sample by construction)

    Returns (model, per-epoch validation MAE history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert len(origin) == lookback, "origin must equal lookback"
    model = PatchTST(lookback=lookback, horizon=horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()

    Xs = torch.from_numpy(train).float()
    tr, tgt = [], []
    for i in range(lookback, len(Xs) - horizon + 1):
        tr.append(Xs[i - lookback:i])
        tgt.append(Xs[i:i + horizon])
    Xtr = torch.stack(tr)
    Ytr = torch.stack(tgt)
    Xv = torch.from_numpy(origin).float()
    Yv = torch.from_numpy(holdout[:horizon]).float()

    best_state, best_mae, patience, wait = None, float("inf"), 15, 0
    history = []
    for _ in range(epochs):
        model.train()
        for i in range(0, len(Xtr), 32):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[i:i + 32]), Ytr[i:i + 32])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vmae = F.l1_loss(model(Xv.unsqueeze(0))[0], Yv).item()
        history.append(vmae)
        if vmae < best_mae - 1e-6:
            best_mae, wait = vmae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
