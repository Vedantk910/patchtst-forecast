# patchtst-forecast

Forecasting the dollar, honestly. Monthly DXY returns split walk-forward at May 2024: a patched transformer (PatchTST) trained with early stopping against ETS, recursive LightGBM, and the random walk, at horizons 5 and 10, with a PSI drift monitor wired in.

## The honest numbers

- Horizon 5: PatchTST wins at MAE 0.0099; LightGBM 0.0131, ETS 0.0143, random walk 0.0140.
- Horizon 10: LightGBM wins at MAE 0.0110; PatchTST slips to 0.0111, random walk 0.0126.
- The crossover between h=5 and h=10 is where the model choice should change. Choosing by horizon rather than by default is the whole lesson.

## What it does

1. **Data** — monthly DXY returns (FRED), cache-first, walk-forward split at May 2024
2. **Models** — PatchTST (scaled dot-product attention, PyTorch), ETS, recursive LightGBM, random walk
3. **Evaluation** — MAE at horizons 5 and 10, PSI drift monitor against the random walk
4. **Output** — `out/ablation.csv`, the model-by-horizon error table the site charts

## Project layout

```
patchtst/
  data/     # DXY loader + cache
  models/   # PatchTST + baselines
  eval/     # MAE, PSI, ablation table
run_ablation.py  # end-to-end CLI, writes out/ablation.csv
out/         # ablation.csv, drift.json
tests/       # pytest suite (offline, deterministic seed)
```

## Quick start

```bash
pip install -r requirements.txt
python run_ablation.py
python -m pytest tests/ -q
```

Training is deterministic (fixed seed). The writeup keeps the h=10 loss — a transformer that beats a gradient-boosted tree at short horizons and loses at long ones is the honest result, and the crossover is the finding.
