"""Task 2e - metrik: RMSE, skor asimetris, per rentang RUL dan per mode kerusakan."""
from __future__ import annotations

import numpy as np
import pandas as pd

RUL_BINS = [0, 25, 50, 75, 100, 125, 150.1]
RUL_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-125", "125-150"]


def rmse(y_pred, y_true) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def asymmetric_score(y_pred, y_true) -> float:
    """d = pred - aktual. d >= 0 (prediksi terlambat / terlalu optimis) dihukum lebih berat
    (exp(d/10)) daripada d < 0 (terlalu dini, exp(-d/13)), karena mesin bisa rusak sebelum diservis."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def summarize(y_pred, y_true) -> dict:
    y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
    n = len(y_true)
    return {"n": n, "rmse": rmse(y_pred, y_true), "mae": float(np.mean(np.abs(y_pred - y_true))),
            "bias": float(np.mean(y_pred - y_true)),
            "skor": asymmetric_score(y_pred, y_true),
            "skor_per_sampel": asymmetric_score(y_pred, y_true) / max(n, 1),
            # skor jika jumlah sampel = 300 (ukuran test), agar sebanding dengan test tersembunyi
            "skor_per_300": 300 * asymmetric_score(y_pred, y_true) / max(n, 1)}


def by_range(df: pd.DataFrame, pred="pred", true="rul") -> pd.DataFrame:
    b = pd.cut(df[true], RUL_BINS, labels=RUL_LABELS, right=False)
    rows = []
    for lab, g in df.groupby(b, observed=True):
        s = summarize(g[pred], g[true])
        rows.append({"rentang_rul": lab, "n": s["n"], "rmse": s["rmse"], "bias": s["bias"],
                     "skor_per_sampel": s["skor_per_sampel"]})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, col: str, pred="pred", true="rul") -> pd.DataFrame:
    rows = []
    for lab, g in df.groupby(col):
        s = summarize(g[pred], g[true])
        rows.append({col: lab, "n": s["n"], "rmse": s["rmse"], "bias": s["bias"],
                     "skor_per_sampel": s["skor_per_sampel"]})
    return pd.DataFrame(rows)
