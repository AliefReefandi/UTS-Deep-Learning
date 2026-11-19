"""Task 3f - pemeriksaan heterogenitas di DALAM cluster final.

Kriteria k global (silhouette, DB, CH, stabilitas) mengukur keterpisahan antar-cluster secara rata-rata,
sehingga struktur yang lebih halus di dalam satu cluster bisa tidak terlihat. Modul ini memeriksa
dua faktor, masing-masing dari satu modalitas, dan apakah keduanya saling terkait:

  faktor tekstur   : pemecahan cluster c oleh refinement blok gambar (cluster.refine_by_block, langkah 1)
  faktor spektrum  : jarak puncak 1-2 (kanal) per spektrum; diuji bimodal per cluster dengan
                     GMM 1D 1 vs 2 komponen. Bimodal jika ΔBIC > 10 (bukti kuat, Kass & Raftery 1995)
                     DAN kedua mode terpisah >= 2 std (campuran dua Gauss baru punya dua puncak jika
                     jarak mean > 2σ) DAN komponen kecil >= 10% (bukan sekadar ekor/outlier).
                     Dengan N ribuan, ΔBIC saja hampir selalu > 10 untuk distribusi yang sedikit miring.
Keluaran: tabel silang 2x2, Cramér's V antar-faktor dan terhadap mikroskop, serta seberapa baik
tiap faktor bisa ditebak dari tiap blok (regresi logistik, CV 5-fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kode.task3.fusion import cramers_v

DBIC_MIN, SEP_MIN, W_MIN = 10.0, 2.0, 0.10


def peak_positions(spec_pp: np.ndarray, n: int = 3, prom: float = 0.05) -> np.ndarray:
    """Posisi (kanal) n puncak paling menonjol, diurutkan; NaN jika puncaknya < n."""
    from scipy.signal import find_peaks
    pos = np.full((len(spec_pp), n), np.nan)
    for i, y in enumerate(spec_pp):
        p, pr = find_peaks(y, prominence=prom)
        if len(p) >= n:
            pos[i] = np.sort(p[np.argsort(-pr["prominences"])[:n]])
    return pos


def bimodality(x: np.ndarray, seed: int):
    from sklearn.mixture import GaussianMixture
    x = x[~np.isnan(x)].reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=seed).fit(x)
    g2 = GaussianMixture(2, n_init=3, random_state=seed).fit(x)
    order = np.argsort(g2.means_.ravel())
    mu = g2.means_.ravel()[order]
    sd = np.sqrt(g2.covariances_.ravel()[order])
    r = {"n_valid": int(len(x)), "median": float(np.median(x)), "delta_bic": float(g1.bic(x) - g2.bic(x)),
         "mode_1": float(mu[0]), "mode_2": float(mu[1]), "jarak_mode_per_std": float((mu[1] - mu[0]) / sd.max()),
         "bobot_min": float(g2.weights_.min())}
    r["bimodal"] = bool(r["delta_bic"] > DBIC_MIN and r["jarak_mode_per_std"] >= SEP_MIN and r["bobot_min"] >= W_MIN)
    return r, g2, order


def _predictability(B: np.ndarray, y: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(max_iter=2000), B, y, cv=cv).mean())


def analyze(final, alt, c: int, blocks: dict, spec_pp, mic, seed: int, log=print):
    """final: label k cluster; alt: label setelah refinement (pecahan tekstur cluster c).
    Return (tabel bimodal per cluster, ringkasan dict, label faktor untuk cluster c)."""
    pos = peak_positions(spec_pp)
    gap = pos[:, 1] - pos[:, 0]
    rows, models = [], {}
    for k in range(final.max() + 1):
        r, g2, order = bimodality(gap[final == k], seed)
        rows.append({"cluster": k, "n": int((final == k).sum()), **r})
        models[k] = (g2, order)
    bim = pd.DataFrame(rows)
    log(bim.round(3).to_string(index=False))

    s = np.flatnonzero(final == c)
    sub_alt = alt[s]
    vals = np.unique(sub_alt)
    tex = (sub_alt == vals[-1]).astype(int) if len(vals) == 2 else np.zeros(len(s), int)
    ok = ~np.isnan(gap[s])
    g2, order = models[c]
    spc = np.full(len(s), -1)
    rank = np.argsort(order)                       # komponen GMM -> 0 (jarak kecil) / 1 (jarak besar)
    spc[ok] = rank[g2.predict(gap[s][ok].reshape(-1, 1))]
    m = spc >= 0
    ct = pd.crosstab(pd.Series(tex[m], name="tekstur"), pd.Series(spc[m], name="spektrum"))
    summ = {
        "cluster": int(c), "n": int(len(s)), "n_spektrum_valid": int(m.sum()),
        "label_tekstur_alternatif": [int(v) for v in vals],
        "tabel_silang_tekstur_x_spektrum": ct.to_dict(),
        "cramers_v": {"tekstur_vs_spektrum": cramers_v(tex[m], spc[m]),
                      "tekstur_vs_mikroskop": cramers_v(tex, mic[s]),
                      "spektrum_vs_mikroskop": cramers_v(spc[m], mic[s][m])},
        "pct_M2": {"tekstur_0": float(100 * np.mean(mic[s][tex == 0] == "M2")),
                   "tekstur_1": float(100 * np.mean(mic[s][tex == 1] == "M2")),
                   "spektrum_0": float(100 * np.mean(mic[s][spc == 0] == "M2")),
                   "spektrum_1": float(100 * np.mean(mic[s][spc == 1] == "M2"))},
        "akurasi_tebak_faktor": {
            f"{f}_dari_{b}": _predictability(blocks[b][s][m], y[m], seed)
            for f, y in (("tekstur", tex), ("spektrum", spc)) for b in blocks},
        "jarak_puncak_median": {"spektrum_0": float(np.median(gap[s][spc == 0])),
                                "spektrum_1": float(np.median(gap[s][spc == 1]))},
        "posisi_puncak_median": {f"spektrum_{j}": np.nanmedian(pos[s][spc == j], 0).tolist() for j in (0, 1)},
    }
    log(f"  cluster {c}: tabel silang tekstur x spektrum\n{ct}")
    log(f"  Cramér's V {summ['cramers_v']}; akurasi tebak faktor {summ['akurasi_tebak_faktor']}")
    return bim, summ, {"idx": s, "tekstur": tex, "spektrum": spc, "gap": gap}


# ------------------------------------------------------------------ grafik
def plot(bim, summ, fac, final, proj, img, spec_pp, path_a, path_b, seed=0):
    from kode.common.plot import SERIES, TEXT2, plt, savefig
    c, s, tex, spc, gap = summ["cluster"], fac["idx"], fac["tekstur"], fac["spektrum"], fac["gap"]

    # (a) histogram jarak puncak per cluster, proyeksi 2D cluster c diwarnai tiap faktor, tabel silang
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), gridspec_kw={"wspace": .3, "width_ratios": [1.5, 1, 1, .8]})
    ax = axes[0]
    lo, hi = np.nanpercentile(gap, [0.5, 99.5])
    bins = np.arange(np.floor(lo), np.ceil(hi) + 2) - .5
    for k in range(final.max() + 1):
        h, _ = np.histogram(gap[final == k], bins=bins, density=True)
        r = bim.set_index("cluster").loc[k]
        ax.step(bins[:-1] + .5, h, where="mid", lw=2.2 if k == c else 1,
                color=SERIES[k % len(SERIES)], alpha=1 if k == c else .55,
                label=f"c{k}{' (bimodal)' if r.bimodal else ''}")
    ax.set(xlabel="jarak puncak 1-2 (kanal)", ylabel="kepadatan", title="Jarak puncak 1-2 per cluster")
    ax.legend(fontsize=7, ncol=2)
    if "UMAP" in proj:
        idx, P = proj["UMAP"]
        pos_in = np.full(len(final), -1)
        pos_in[idx] = np.arange(len(idx))
        j = pos_in[s]
        keep = j >= 0
        for ax, lab, title, names in ((axes[1], tex, "tekstur (blok gambar)", ("tekstur 0", "tekstur 1")),
                                      (axes[2], spc, "jarak puncak 1-2", (
                                          f"spektrum 0 (~{summ['jarak_puncak_median']['spektrum_0']:.0f} kanal)",
                                          f"spektrum 1 (~{summ['jarak_puncak_median']['spektrum_1']:.0f} kanal)"))):
            for v in (0, 1):
                m = keep & (lab == v)
                ax.scatter(P[j[m], 0], P[j[m], 1], s=1.5, color=SERIES[v], alpha=.5, linewidths=0,
                           rasterized=True, label=names[v])
            ax.set(title=f"UMAP cluster {c}: {title}", xticks=[], yticks=[])
            ax.grid(False)
            ax.legend(markerscale=6, fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.25), ncol=1)
    ax = axes[3]
    ct = pd.DataFrame(summ["tabel_silang_tekstur_x_spektrum"])
    ax.imshow(ct.to_numpy(), cmap="Blues", vmin=0)
    for i in range(ct.shape[0]):
        for jj in range(ct.shape[1]):
            ax.text(jj, i, f"{ct.iat[i, jj]}", ha="center", va="center", fontsize=9,
                    color="white" if ct.iat[i, jj] > .5 * ct.to_numpy().max() else TEXT2)
    ax.set_xticks(range(ct.shape[1]), [f"spektrum {v}" for v in ct.columns], fontsize=8)
    ax.set_yticks(range(ct.shape[0]), [f"tekstur {v}" for v in ct.index], fontsize=8)
    v = summ["cramers_v"]
    ax.set_title(f"Tabel silang\nV tekstur-spektrum {v['tekstur_vs_spektrum']:.2f}", fontsize=9)
    ax.grid(False)
    fig.suptitle(f"Cluster {c} heterogen: dua faktor independen (V mikroskop: tekstur "
                 f"{v['tekstur_vs_mikroskop']:.2f}, spektrum {v['spektrum_vs_mikroskop']:.2f})", fontsize=10, y=1.03)
    savefig(fig, path_a)

    # (b) grid 2x2 contoh gambar + spektrum rata-rata per sel
    rng = np.random.default_rng(seed)
    n_img = 4
    fig = plt.figure(figsize=(2 * (n_img * 1.15 + 2.6), 2 * 1.5 + .6))
    gs = fig.add_gridspec(2, 2 * (n_img + 1), width_ratios=([1] * n_img + [2.4]) * 2, wspace=.08, hspace=.3)
    for t in (0, 1):
        for sp in (0, 1):
            cell = np.flatnonzero((tex == t) & (spc == sp))
            off = sp * (n_img + 1)
            for jj, i in enumerate(rng.choice(cell, min(n_img, len(cell)), replace=False)):
                a = fig.add_subplot(gs[t, off + jj])
                a.imshow(img[s[i]], cmap="gray")
                a.set_xticks([]), a.set_yticks([]), a.grid(False)
                if jj == 0:
                    a.set_title(f"tekstur {t} x spektrum {sp} (n={len(cell)})", fontsize=8, loc="left")
            a = fig.add_subplot(gs[t, off + n_img])
            mu = spec_pp[s[cell]].mean(0)
            a.plot(mu, color=SERIES[sp], lw=1.2)
            p2 = [summ["posisi_puncak_median"][f"spektrum_{j}"][1] for j in (0, 1)]
            a.axvspan(min(p2) - 4, max(p2) + 4, color=SERIES[2], alpha=.12, lw=0)
            a.set_yticks([])
            a.set_title("spektrum rata-rata (area: puncak 2)", fontsize=7)
    fig.suptitle(f"Cluster {c}: empat kombinasi tekstur x spektrum", fontsize=10)
    savefig(fig, path_b)
