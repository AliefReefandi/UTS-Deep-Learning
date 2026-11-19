"""Task 2d - baseline non-deep-learning: gradient boosting pada fitur ringkas window.

Fitur per sensor (hanya dari baris asli dalam window, padding diabaikan lewat mask):
  rata-rata 5 siklus terakhir (level kini), rata-rata window, kemiringan (tren), std
+ umur mesin dan panjang histori dalam window. Total 14*4 + 2 = 58 fitur.
Ini 'feature engineering manual' yang ingin dikalahkan model sequence (yang belajar fiturnya sendiri).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def window_features(Xw: np.ndarray) -> np.ndarray:
    """Xw: (N, W, F+1) dengan kolom -2 = umur, -1 = mask."""
    m = Xw[:, :, -1:]                                  # (N, W, 1)
    S = Xw[:, :, :-2]                                  # sensor
    n = m.sum(1)                                       # (N, 1) jumlah baris asli
    mean = (S * m).sum(1) / n
    m5 = m[:, -5:]
    last5 = (S[:, -5:] * m5).sum(1) / np.maximum(m5.sum(1), 1)
    tt = np.arange(Xw.shape[1], dtype=np.float32)[None, :, None]
    tbar = (tt * m).sum(1, keepdims=True) / n[:, None]
    dt = (tt - tbar) * m
    slope = (dt * (S - mean[:, None])).sum(1) / np.maximum((dt ** 2).sum(1), 1e-6)
    std = np.sqrt(((S - mean[:, None]) ** 2 * m).sum(1) / n)
    age = Xw[:, -1, -2:-1]
    return np.concatenate([last5, mean, slope, std, age, n / Xw.shape[1]], 1).astype(np.float32)


def featurize(bank, idx, chunk: int = 20000) -> np.ndarray:
    import torch
    out = []
    for k in range(0, len(idx), chunk):
        pos = torch.from_numpy(bank.positions(idx[k:k + chunk])).to(bank.device)
        out.append(window_features(bank.get(pos).float().cpu().numpy()))
    return np.concatenate(out)


def fit_baseline(bank, idx_tr, seed: int, max_train: int = 80000, debug: bool = False):
    rng = np.random.default_rng(seed)
    idx_tr = np.asarray(idx_tr)
    if len(idx_tr) > max_train:                         # subsample: window bertetangga sangat mirip
        idx_tr = idx_tr[rng.choice(len(idx_tr), max_train, replace=False)]
    X = featurize(bank, idx_tr)
    y = bank.labels(idx_tr)
    model = HistGradientBoostingRegressor(max_iter=50 if debug else 400, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=seed)
    model.fit(X, y)
    return model


def predict_baseline(model, bank, idx) -> np.ndarray:
    return np.clip(model.predict(featurize(bank, idx)), 0, 150)
