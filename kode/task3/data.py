"""Task 3 - pemuatan data clustering dan preprocessing spektrum.

Data (data/kamus_data.md): clustering.npz berisi `gambar` uint8 (N, 64, 64) citra mikrostruktur dan
`spektrum` float16 (N, 128) spektrum komposisi; clustering_meta.csv berisi indeks, id_sampel,
mikroskop (M1/M2). Baris ke-i meta = indeks i di npz. Tidak ada label.

Preprocessing spektrum (alasan: lihat results/task3/eda/RINGKASAN.md):
1. despike  : spike = lonjakan positif 1-2 kanal. Residual terhadap median bergerak 5 kanal > 0.5; puncak asli
              lebarnya >= ~6 kanal sehingga median mengikutinya, spike tidak. Diganti nilai median.
2. baseline : baseline lambat (offset + kemiringan + lengkung) diestimasi dengan polinom derajat 2
              yang di-fit iteratif hanya pada titik "di bawah" kurva (modpoly): puncak tidak ikut
              menarik baseline. Lalu dikurangkan.
3. normalisasi: dibagi norma L2 per spektrum -> intensitas total (konsentrasi/gain detektor) tidak
              memengaruhi bentuk spektrum.
Semua fungsi per-spektrum (tidak ada statistik lintas sampel), jadi tidak ada kebocoran informasi
antar-sampel dan hasilnya sama untuk subset maupun data penuh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from kode.common.config import data_path

SPIKE_THR = 0.5     # dari celah histogram residual (EDA 07)
SPIKE_WIN = 5
BASE_DEG = 2
BASE_ITER = 30


def load(args, debug_n: int = 3000):
    """Kembalikan (gambar uint8 (N,64,64), spektrum float32 (N,128), meta DataFrame).
    Mode --debug: subset acak `debug_n` sampel (seeded), indeks asli disimpan di meta.indeks."""
    with np.load(data_path(args, "task3_data")) as d:
        img, spec = d["gambar"], d["spektrum"].astype(np.float32)
    meta = pd.read_csv(data_path(args, "task3_meta"))
    assert (meta.indeks.to_numpy() == np.arange(len(meta))).all(), "urutan meta != npz"
    if getattr(args, "debug", False):
        idx = np.sort(np.random.default_rng(args.seed).choice(len(meta), debug_n, replace=False))
        img, spec, meta = img[idx], spec[idx], meta.iloc[idx].reset_index(drop=True)
    return img, spec, meta


def despike(spec: np.ndarray, thr: float = SPIKE_THR, win: int = SPIKE_WIN, return_mask: bool = False):
    """Spike = residual POSITIF terhadap median bergerak > thr. Ambang absolut, bukan k*sigma per
    spektrum: sigma robust (MAD) dari residual terhadap median 5 kanal hampir nol (median sering
    sama dengan titik itu sendiri), sehingga k*sigma menandai noise biasa dan puncak sempit. Residual maks per spektrum bimodal (<=0.4 vs 0.7-4,
    celah kosong 0.4-0.7; grafik EDA 07) -> thr = 0.5. mode="mirror" agar spike di kanal tepi
    juga terdeteksi (padding "nearest" membuat median = nilai spike itu sendiri)."""
    med = median_filter(spec, size=(1, win), mode="mirror")
    mask = (spec - med) > thr
    out = np.where(mask, med, spec)
    return (out, mask) if return_mask else out


def baseline(spec: np.ndarray, deg: int = BASE_DEG, n_iter: int = BASE_ITER) -> np.ndarray:
    """Polinom modpoly, divektorisasi untuk semua spektrum sekaligus (satu matriks Vandermonde)."""
    x = np.linspace(-1, 1, spec.shape[1])
    V = np.vander(x, deg + 1)                       # (L, deg+1)
    P = np.linalg.pinv(V)                           # (deg+1, L)
    y = spec.copy()
    for _ in range(n_iter):
        fit = (P @ y.T).T @ V.T                     # (N, L)
        y = np.minimum(y, fit)                      # titik puncak dipotong ke kurva -> tidak menarik fit
    return (P @ y.T).T @ V.T


def baseline_coef(spec: np.ndarray, deg: int = BASE_DEG, n_iter: int = BASE_ITER) -> np.ndarray:
    """Koefisien baseline [a2, a1, a0] di x in [-1, 1] (untuk EDA: lengkung, kemiringan, offset)."""
    x = np.linspace(-1, 1, spec.shape[1])
    V = np.vander(x, deg + 1)
    return (np.linalg.pinv(V) @ baseline(spec, deg, n_iter).T).T


def preprocess_spectrum(spec: np.ndarray) -> np.ndarray:
    s = despike(spec)
    s = s - baseline(s)
    return (s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def standardize_images(img: np.ndarray) -> np.ndarray:
    """Per-gambar: kurangi rata-rata, bagi std -> menghapus beda brightness/kontras (termasuk M1 vs M2)."""
    x = img.astype(np.float32)
    m = x.mean((1, 2), keepdims=True)
    s = x.std((1, 2), keepdims=True)
    return (x - m) / np.maximum(s, 1e-3)
