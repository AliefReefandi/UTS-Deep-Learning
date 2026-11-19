"""Task 3e - menentukan jumlah cluster k dengan beberapa kriteria tanpa label.

Untuk k = 2..k_max, KMeans pada representasi fusion (latent FusionAE):
  silhouette       (↑) rapat di dalam cluster & terpisah antar-cluster; dihitung pada 10.000 sampel
  Davies-Bouldin   (↓) rasio sebaran dalam cluster terhadap jarak antar-pusat
  Calinski-Harabasz(↑) rasio varians antar / dalam cluster
  stabilitas       (↑) bootstrap: dua subsampel 80% di-cluster terpisah, lalu ARI kedua labelnya pada
                       semua titik. k yang benar = struktur nyata, jadi hasilnya sama walau datanya
                       sedikit berubah. k yang salah memotong cluster di tempat acak, jadi tidak stabil.
  BIC GMM          (↓) dilaporkan sebagai pembanding (model probabilistik)

Aturan pilih (transparan, ditetapkan sebelum melihat hasil):
  1. kandidat = k dengan stabilitas >= 0,80 (cluster yang tidak stabil tidak layak dilaporkan);
  2. di antara kandidat, k* = jumlah peringkat terbaik dari silhouette, Davies-Bouldin, dan
     Calinski-Harabasz (tiap kriteria punya bias berbeda; jumlah peringkat meredam bias satu kriteria);
     seri -> k lebih kecil (lebih sederhana).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAB_MIN = 0.80


def stability(X, k, seed, n_boot=5, frac=0.8) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_boot):
        labs = []
        for _ in range(2):
            idx = rng.choice(len(X), int(frac * len(X)), replace=False)
            km = KMeans(k, n_init=3, random_state=int(rng.integers(1 << 30))).fit(X[idx])
            labs.append(km.predict(X))
        out.append(adjusted_rand_score(*labs))
    return float(np.mean(out))


def scan(X: np.ndarray, ks, seed: int, n_sil: int = 10000, log=print) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(len(X), min(n_sil, len(X)), replace=False)
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        gmm = GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(X)
        rows.append({"k": k,
                     "silhouette": silhouette_score(X[sil_idx], lab[sil_idx]),
                     "davies_bouldin": davies_bouldin_score(X, lab),
                     "calinski_harabasz": calinski_harabasz_score(X, lab),
                     "stabilitas_ARI": stability(X, k, seed + k),
                     "bic_gmm": gmm.bic(X),
                     "ukuran_cluster_min_pct": 100 * np.bincount(lab).min() / len(X)})
        r = rows[-1]
        log(f"    k={k:2d} silhouette {r['silhouette']:.3f} DB {r['davies_bouldin']:.3f} "
            f"CH {r['calinski_harabasz']:.0f} stabilitas {r['stabilitas_ARI']:.3f} BIC {r['bic_gmm']:.0f}")
    return pd.DataFrame(rows)


def choose(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    df = df.copy()
    cand = df.stabilitas_ARI >= STAB_MIN
    if not cand.any():                       # tidak ada yang stabil: pakai yang paling stabil
        cand = df.stabilitas_ARI == df.stabilitas_ARI.max()
    df["kandidat"] = cand
    sub = df[cand]
    rank = (sub.silhouette.rank(ascending=False) + sub.davies_bouldin.rank(ascending=True)
            + sub.calinski_harabasz.rank(ascending=False))
    df.loc[cand, "jumlah_peringkat"] = rank
    best = rank.min()
    k = int(sub.k[rank == best].min())
    df["dipilih"] = df.k == k
    return k, df


def plot(df: pd.DataFrame, k: int, path):
    from kode.common.plot import MUTED, SERIES, plt, savefig
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.9), gridspec_kw={"wspace": .38})
    spec = [("silhouette", "Silhouette (↑)"), ("davies_bouldin", "Davies-Bouldin (↓)"),
            ("calinski_harabasz", "Calinski-Harabasz (↑)"), ("stabilitas_ARI", "Stabilitas bootstrap ARI (↑)"),
            ("bic_gmm", "BIC GMM (↓)")]
    for ax, (col, title) in zip(axes, spec):
        ax.plot(df.k, df[col], "o-", color=SERIES[0], ms=4)
        ax.axvline(k, color=SERIES[1], ls="--", lw=1.2)
        if col == "stabilitas_ARI":
            ax.axhline(STAB_MIN, color=MUTED, ls=":", lw=1)
        ax.set(title=title, xlabel="k")
        ax.set_xticks(df.k[::2] if len(df) > 8 else df.k)
    fig.suptitle(f"Pemilihan k pada latent FusionAE: k* = {k} (garis putus-putus)", fontsize=10, y=1.06)
    savefig(fig, path)
