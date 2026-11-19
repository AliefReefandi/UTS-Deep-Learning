"""Task 3f - visualisasi (UMAP & t-SNE), profil cluster, dan cluster yang sulit dipisahkan.

UMAP pada SEMUA sampel (menjaga struktur global lebih baik: jarak antar-kelompok lebih bermakna);
t-SNE pada 8.000 sampel (bagus untuk struktur lokal, tetapi O(N log N) dan jarak antar-kelompok
tidak bermakna). Keduanya hanya untuk visualisasi: clustering TIDAK dilakukan di ruang 2D.

Cluster sulit dipisahkan, dua ukuran yang saling menguatkan:
  - tumpang tindih kNN: untuk tiap titik di cluster a, berapa persen dari 15 tetangga terdekatnya
    (di latent) yang berada di cluster b; disimetrikan (a,b)+(b,a).
  - ketidakpastian IDEC: rata-rata q terbesar kedua; pasangan (cluster 1, cluster 2 teratas) yang
    sering muncul bersama = batas yang kabur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def embed_2d(Z, seed, n_tsne=8000, log=print):
    out = {}
    try:
        import umap
        out["UMAP"] = (np.arange(len(Z)), umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed)
                       .fit_transform(Z))
    except Exception as e:                       # mis. numba tidak tersedia
        log(f"  UMAP dilewati: {type(e).__name__}: {e}")
    from sklearn.manifold import TSNE
    idx = np.sort(np.random.default_rng(seed).choice(len(Z), min(n_tsne, len(Z)), replace=False))
    out["t-SNE"] = (idx, TSNE(2, perplexity=40, init="pca", random_state=seed).fit_transform(Z[idx]))
    return out


def knn_overlap(Z, lab, k_nn=15, n=10000, seed=0) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(Z)
    nb = nn.kneighbors(Z[idx], return_distance=False)[:, 1:]
    K = lab.max() + 1
    M = np.zeros((K, K))
    for a in range(K):
        sel = lab[idx] == a
        if sel.any():
            M[a] = np.bincount(lab[nb[sel]].ravel(), minlength=K) / (sel.sum() * k_nn)
    return pd.DataFrame(M)


def hard_pairs(M: pd.DataFrame, q: np.ndarray | None, lab, top=3, sep: dict | None = None) -> pd.DataFrame:
    """Urutan: tumpang tindih kNN (ruang gabungan) menurun, lalu akurasi pasangan dari satu modalitas
    (sep: {nama: tabel pairwise_separability}) menaik -> pasangan yang hanya bisa dibedakan oleh satu
    modalitas ikut terangkat walaupun di ruang gabungan sudah terpisah."""
    K = len(M)
    rows = []
    second = None
    if q is not None:
        srt = np.argsort(-q, 1)
        second = pd.crosstab(srt[:, 0], srt[:, 1])
    for a in range(K):
        for b in range(a + 1, K):
            r = {"cluster_a": a, "cluster_b": b, "overlap_kNN_pct": 100 * (M.iat[a, b] + M.iat[b, a]) / 2}
            if second is not None:
                cnt = second.get(b, pd.Series(dtype=float)).get(a, 0) + second.get(a, pd.Series(dtype=float)).get(b, 0)
                r["n_runner_up_IDEC"] = int(cnt)
            for nm, df in (sep or {}).items():
                m = df[(df.cluster_a == a) & (df.cluster_b == b)]
                r[f"akurasi_pasangan_{nm}"] = float(m.akurasi.iloc[0]) if len(m) else np.nan
            rows.append(r)
    out = pd.DataFrame(rows)
    acc_cols = [c for c in out if c.startswith("akurasi_pasangan_")]
    out["akurasi_satu_modalitas_min"] = out[acc_cols].min(1) if acc_cols else np.nan
    out = out.sort_values(["overlap_kNN_pct", "akurasi_satu_modalitas_min"], ascending=[False, True])
    return out.head(top).reset_index(drop=True)


# ------------------------------------------------------------------ grafik
def _scatter(ax, P, c, title, cmap=None, labels_at_center=True, s=1.5):
    from kode.common.plot import SERIES
    cats = np.unique(c)
    for j, v in enumerate(cats):
        sel = c == v
        col = SERIES[j % len(SERIES)] if cmap is None else cmap[j]
        ax.scatter(P[sel, 0], P[sel, 1], s=s, color=col, alpha=.5, linewidths=0, rasterized=True, label=str(v))
        if labels_at_center:
            m = np.median(P[sel], 0)
            ax.text(m[0], m[1], str(v), fontsize=8, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1, alpha=.85))
    ax.set(title=title, xticks=[], yticks=[])
    ax.grid(False)


def plot_2d(proj: dict, lab, mic, single_labs: dict, path):
    from kode.common.plot import SERIES, plt, savefig
    names = list(proj)
    extra = list(single_labs)
    fig, axes = plt.subplots(len(names), 2 + len(extra), figsize=(4.0 * (2 + len(extra)), 3.9 * len(names)),
                             squeeze=False)
    for r, nm in enumerate(names):
        idx, P = proj[nm]
        _scatter(axes[r, 0], P, lab[idx], f"{nm}: cluster final")
        _scatter(axes[r, 1], P, mic[idx], f"{nm}: mikroskop", cmap=[SERIES[0], SERIES[1]], labels_at_center=False)
        axes[r, 1].legend(markerscale=6, fontsize=7, loc="best")
        for j, e in enumerate(extra):
            _scatter(axes[r, 2 + j], P, single_labs[e][idx], f"{nm}: cluster {e}", labels_at_center=False)
    savefig(fig, path)


def plot_profiles(img, spec_pp, lab, mic, path, n_img=6, seed=0):
    from kode.common.plot import SERIES, plt, savefig
    K = lab.max() + 1
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(1.15 * n_img + 3.6, 1.2 * K + 0.6))
    gs = fig.add_gridspec(K, n_img + 1, width_ratios=[1] * n_img + [3.0], wspace=.06, hspace=.12)
    for c in range(K):
        sel = np.flatnonzero(lab == c)
        pick = rng.choice(sel, min(n_img, len(sel)), replace=False)
        for j, i in enumerate(pick):
            a = fig.add_subplot(gs[c, j])
            a.imshow(img[i], cmap="gray")
            a.set_xticks([]), a.set_yticks([]), a.grid(False)
            if j == 0:
                a.set_ylabel(f"c{c}\nn={len(sel)}\nM2 {100 * (mic[sel] == 'M2').mean():.0f}%", fontsize=7,
                             rotation=0, ha="right", va="center")
        a = fig.add_subplot(gs[c, n_img])
        m, s = spec_pp[sel].mean(0), spec_pp[sel].std(0)
        a.fill_between(np.arange(len(m)), m - s, m + s, color=SERIES[0], alpha=.25, lw=0)
        a.plot(m, color=SERIES[0], lw=1.2)
        a.set_yticks([])
        if c < K - 1:
            a.set_xticklabels([])
        else:
            a.set_xlabel("kanal spektrum (rata-rata ± std)")
    fig.suptitle("Profil cluster final: contoh gambar acak & spektrum terproses", fontsize=10)
    savefig(fig, path)


def plot_hard(M: pd.DataFrame, pairs: pd.DataFrame, img, spec_pp, lab, path, n_img=4, seed=0):
    from kode.common.plot import SERIES, TEXT2, plt, savefig
    rng = np.random.default_rng(seed)
    P = len(pairs)
    fig = plt.figure(figsize=(14, 2.6 * max(P, 2)))
    gs = fig.add_gridspec(max(P, 2), 1 + 2 * n_img + 1, width_ratios=[3.2] + [1] * (2 * n_img) + [2.6],
                          wspace=.08, hspace=.6)
    ax = fig.add_subplot(gs[:, 0])
    K = len(M)
    Ms = (M.to_numpy() + M.to_numpy().T) / 2 * 100
    np.fill_diagonal(Ms, np.nan)
    ax.imshow(Ms, cmap="Reds")
    for i in range(K):
        for j in range(K):
            if i != j and Ms[i, j] >= 0.5:
                ax.text(j, i, f"{Ms[i, j]:.0f}", ha="center", va="center", fontsize=6, color=TEXT2)
    ax.set_xticks(range(K)), ax.set_yticks(range(K))
    ax.set(title="Tumpang tindih kNN antar-cluster (%)", xlabel="cluster", ylabel="cluster")
    ax.grid(False)
    for r, row in pairs.iterrows():
        a_, b_ = int(row.cluster_a), int(row.cluster_b)
        for half, c in enumerate((a_, b_)):
            sel = np.flatnonzero(lab == c)
            for j, i in enumerate(rng.choice(sel, min(n_img, len(sel)), replace=False)):
                axi = fig.add_subplot(gs[r, 1 + half * n_img + j])
                axi.imshow(img[i], cmap="gray")
                axi.set_xticks([]), axi.set_yticks([]), axi.grid(False)
                if j == 0:
                    axi.set_title(f"cluster {c}", fontsize=8, loc="left")
        axs = fig.add_subplot(gs[r, -1])
        for half, c in enumerate((a_, b_)):
            axs.plot(spec_pp[lab == c].mean(0), color=SERIES[half], lw=1.1, ls=("-", "--")[half], label=f"c{c}")
        axs.set_yticks([])
        axs.legend(fontsize=7)
        extra = "".join(f"\nakurasi {col.replace('akurasi_pasangan_', '')} saja: {row[col]:.2f}"
                        for col in pairs if col.startswith("akurasi_pasangan_"))
        axs.set_title(f"overlap kNN {row.overlap_kNN_pct:.1f}%{extra}", fontsize=7)
    fig.suptitle("Pasangan cluster yang paling sulit dipisahkan", fontsize=10)
    savefig(fig, path)


def plot_ablation(abl: pd.DataFrame, final, single_labs: dict, path):
    from kode.common.plot import SERIES, TEXT2, plt, savefig
    fig, axes = plt.subplots(1, 1 + len(single_labs), figsize=(4.6 + 4.2 * len(single_labs), 3.6),
                             gridspec_kw={"wspace": .45})
    ax = axes[0]
    y = np.arange(len(abl))
    for j, (col, lab) in enumerate([("AMI_vs_final", "AMI vs final"),
                                    ("akurasi_kNN_cluster_final", "akurasi kNN cluster final"),
                                    ("cramers_v_mikroskop", "Cramér's V mikroskop")]):
        ax.barh(y + (j - 1) * .27, abl[col], .26, color=SERIES[j], label=lab)
    ax.set_yticks(y, abl.representasi)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.05), title="Ablasi modalitas")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    for ax, (name, lab) in zip(axes[1:], single_labs.items()):
        ct = pd.crosstab(final, lab, normalize="index")
        ax.imshow(ct.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = ct.iat[i, j]
                if v >= .1:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if v > .6 else TEXT2)
        ax.set(xlabel=f"cluster ({name})", ylabel="cluster final", title=f"Final vs {name}")
        ax.set_xticks(range(ct.shape[1])), ax.set_yticks(range(ct.shape[0]))
        ax.grid(False)
    savefig(fig, path)
