"""Task 3a - EDA distorsi gambar (rotasi, skala, brightness, beda mikroskop) dan spektrum
(shift, baseline, spike). Grafik/tabel ke <out_dir>/task3/eda/ dan (bukan --debug) disalin ke
<results_dir>/task3/eda/.

  python -m kode.task3.eda [--debug] [--data_dir ...] [--out_dir ...]

Ukuran per gambar (semua TANPA label):
- brightness = rata-rata piksel, kontras = std piksel.
- orientasi = arah dominan dari structure tensor (rata-rata gx^2, gy^2, gx*gy) di seluruh gambar;
  koherensi in [0, 1]: 1 = satu arah dominan (garis/lamela), 0 = isotropik (butir, bintik).
- skala = frekuensi sentroid PSD radial (siklus per 64 px) dari gambar terstandardisasi;
  panjang gelombang = 64 / frekuensi (px). Makin besar = struktur makin kasar.
- rasio frekuensi tinggi = energi PSD di r >= 16 / total -> ketajaman vs blur/noise.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from kode.common import add_common_args, resolve_paths, set_seed
from kode.common.plot import MUTED, SERIES, TEXT2, plt, savefig
from kode.common.utils import save_json
from kode.task3.data import (SPIKE_THR, SPIKE_WIN, baseline, baseline_coef, despike, load,
                             preprocess_spectrum, standardize_images)

R_MAX = 32
HF_R = 16
MIC_COLOR = {"M1": SERIES[0], "M2": SERIES[1]}


# ---------------------------------------------------------------- ukuran gambar
def _radial_index(n=64):
    k = np.fft.fftfreq(n) * n
    r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2).round().astype(int).ravel()
    onehot = np.zeros((n * n, R_MAX + 1), np.float32)
    ok = r <= R_MAX
    onehot[np.where(ok)[0], r[ok]] = 1
    return onehot / np.maximum(onehot.sum(0, keepdims=True), 1)   # cincin kosong (gambar < 64 px) -> 0


def image_stats(img: np.ndarray, chunk: int = 4000):
    """Kembalikan (DataFrame ukuran per gambar, PSD radial (N, R_MAX+1))."""
    onehot = _radial_index(img.shape[1])
    rows, psds = [], []
    r = np.arange(R_MAX + 1)
    for k in range(0, len(img), chunk):
        raw = img[k:k + chunk].astype(np.float32)
        x = standardize_images(img[k:k + chunk])
        gy, gx = np.gradient(x, axis=(1, 2))
        jxx, jyy, jxy = (gx * gx).mean((1, 2)), (gy * gy).mean((1, 2)), (gx * gy).mean((1, 2))
        ang_grad = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(jxx + jyy, 1e-6)
        P = np.abs(np.fft.fft2(x)) ** 2
        psd = P.reshape(len(x), -1) @ onehot                    # (n, R_MAX+1), rata-rata per cincin
        band = psd[:, 1:]
        freq = (band * r[1:]).sum(1) / band.sum(1)
        hf = psd[:, HF_R:].sum(1) / band.sum(1)
        rows.append(pd.DataFrame({"brightness": raw.mean((1, 2)), "kontras": raw.std((1, 2)),
                                  "orientasi": (ang_grad + 90) % 180,   # arah garis = tegak lurus gradien
                                  "koherensi": coh, "frekuensi": freq, "panjang_gel": 64 / freq,
                                  "rasio_hf": hf}))
        psds.append(psd)
    return pd.concat(rows, ignore_index=True), np.concatenate(psds)


def validate_estimators(seed: int, n: int = 3000) -> dict:
    """Uji estimator pada garis sintetis dengan sudut SERAGAM dan lambda diketahui (5 / 11 px).
    Dipakai untuk membedakan sifat data dari bias estimator (histogram orientasi tidak rata)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, np.pi, n)
    lam = rng.choice([5.0, 11.0], n)
    ph = rng.uniform(0, 2 * np.pi, n)
    y, x = np.mgrid[0:64, 0:64]
    img = np.empty((n, 64, 64), np.uint8)
    for i in range(n):
        u = x * np.cos(ang[i]) + y * np.sin(ang[i])
        img[i] = np.clip(128 + 40 * np.sin(2 * np.pi * u / lam[i] + ph[i]) + rng.normal(0, 8, (64, 64)), 0, 255)
    st, _ = image_stats(img)
    err = (st.orientasi - (np.degrees(ang) + 90) % 180 + 90) % 180 - 90   # arah garis = vektor gelombang + 90
    h, _ = np.histogram(st.orientasi, bins=12, range=(0, 180))
    return {"galat_orientasi_median_deg": round(float(np.median(np.abs(err))), 2),
            "galat_orientasi_p95_deg": round(float(np.quantile(np.abs(err), .95)), 2),
            "histogram_12_bin_sudut_seragam": h.tolist(), "rasio_bin_maks_min_sudut_seragam": round(float(h.max() / h.min()), 2),
            "lambda_terukur_median": {str(l): round(float(st.panjang_gel[lam == l].median()), 2) for l in (5.0, 11.0)}}


def mic_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Seberapa mudah mikroskop ditebak dari fitur X (AUC CV 5-fold). 0.5 = tidak ada jejak mikroskop."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def _hist_by_mic(ax, v, mic, bins, xlabel, title):
    for mk in ("M1", "M2"):
        ax.hist(v[mic == mk], bins=bins, alpha=.6, color=MIC_COLOR[mk], density=True, label=mk)
    ax.set(xlabel=xlabel, ylabel="kepadatan", title=title)


def run(args):
    args = resolve_paths(args, "task3")
    set_seed(args.seed)
    out = args.task_out / "eda"
    out.mkdir(parents=True, exist_ok=True)
    img, spec, meta = load(args)
    mic = meta.mikroskop.to_numpy()
    rng = np.random.default_rng(args.seed)
    print(f"[task3-eda] {len(meta)} sampel; mikroskop {meta.mikroskop.value_counts().to_dict()}")
    summary = {"n": len(meta), "mikroskop": meta.mikroskop.value_counts().to_dict(),
               "gambar_shape": list(img.shape[1:]), "spektrum_len": spec.shape[1],
               "nan_spektrum": int(np.isnan(spec).sum())}

    st, psd = image_stats(img)
    st["mikroskop"] = mic

    # ---------- 01 contoh gambar per mikroskop ----------
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.4))
    for row, mk in enumerate(["M1", "M1", "M2", "M2"]):
        pick = rng.choice(np.where(mic == mk)[0], 8, replace=False)
        for ax, i in zip(axes[row], pick):
            ax.imshow(img[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{mk} #{meta.indeks[i]}", fontsize=7)
            ax.grid(False)
    fig.suptitle("Contoh citra mikrostruktur (skala abu-abu tetap 0-255): tekstur garis/lamela, sel/butir, "
                 "pulau gelap, halus; arah & ukuran bervariasi", fontsize=9)
    savefig(fig, out / "01_contoh_gambar.png")

    # ---------- 02 brightness & kontras ----------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    _hist_by_mic(axes[0], st.brightness, mic, 60, "rata-rata piksel", "Brightness")
    _hist_by_mic(axes[1], st.kontras, mic, 60, "std piksel", "Kontras")
    axes[0].legend()
    # beda brightness di dalam mikroskop yang sama (sebaran) vs antar-mikroskop (pergeseran)
    axes[2].scatter(st.brightness[::7], st.kontras[::7], s=2, alpha=.3,
                    c=[MIC_COLOR[m] for m in mic[::7]])
    axes[2].set(xlabel="brightness", ylabel="kontras", title="Brightness vs kontras (warna = mikroskop)")
    savefig(fig, out / "02_brightness_kontras.png")
    g = st.groupby("mikroskop")
    summary["brightness"] = g.brightness.describe()[["mean", "std", "min", "max"]].round(2).to_dict("index")
    summary["kontras"] = g.kontras.describe()[["mean", "std", "min", "max"]].round(2).to_dict("index")

    # ---------- 03 rotasi / orientasi ----------
    COH = 0.5
    ori = st[st.koherensi > COH]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    axes[0].hist(st.koherensi, bins=60, color=SERIES[0])
    axes[0].axvline(COH, color=SERIES[7], ls="--", lw=1)
    axes[0].set(xlabel="koherensi orientasi", ylabel="jumlah gambar",
                title="Seberapa 'berarah' teksturnya")
    _hist_by_mic(axes[1], ori.orientasi, ori.mikroskop.to_numpy(), np.arange(0, 181, 7.5),
                 "arah garis (derajat)", f"Orientasi (koherensi > {COH})")
    axes[1].legend()
    ex = ori.sample(5, random_state=args.seed).sort_values("orientasi")
    axes[2].axis("off")
    for j, (i, r_) in enumerate(ex.iterrows()):
        a = axes[2].inset_axes([j / 5 + .01, .15, .18, .7])
        a.imshow(img[i], cmap="gray"), a.set_xticks([]), a.set_yticks([])
        a.set_title(f"{r_.orientasi:.0f}°", fontsize=8)
    axes[2].set_title("Tekstur sama, arah berbeda")
    savefig(fig, out / "03_orientasi.png")
    hist_o, _ = np.histogram(ori.orientasi, bins=12, range=(0, 180))
    summary["validasi_estimator_sintetis"] = validate_estimators(args.seed)
    summary["orientasi"] = {"frac_koherensi_gt_0.5": float((st.koherensi > COH).mean()),
                            "histogram_12_bin": hist_o.tolist(),
                            "rasio_bin_maks_min": float(hist_o.max() / max(hist_o.min(), 1))}

    # ---------- 04 skala ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    _hist_by_mic(axes[0], st.panjang_gel, mic, np.geomspace(st.panjang_gel.min(), st.panjang_gel.max(), 60),
                 "panjang gelombang dominan (px, log)", "Skala struktur")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[1].scatter(ori.panjang_gel[::3], ori.koherensi[::3], s=2, alpha=.3, color=SERIES[0])
    axes[1].set_xscale("log")
    axes[1].set(xlabel="panjang gelombang (px, log)", ylabel="koherensi",
                title="Garis: 2 kelompok skala")
    # contoh tekstur garis dari skala kecil ke besar
    lam = ori[ori.koherensi > .8].sort_values("panjang_gel")
    pick = lam.iloc[np.linspace(0, len(lam) - 1, 5).astype(int)] if len(lam) >= 5 else lam
    axes[2].axis("off")
    for j, (i, r_) in enumerate(pick.iterrows()):
        a = axes[2].inset_axes([j / 5 + .01, .15, .18, .7])
        a.imshow(img[i], cmap="gray"), a.set_xticks([]), a.set_yticks([])
        a.set_title(f"λ={r_.panjang_gel:.1f}", fontsize=8)
    axes[2].set_title("Garis/lamela: skala kecil → besar")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    for ax in axes[:2]:                                   # label log yang terbaca: 4, 6, 10, 20, 30
        ax.xaxis.set_major_locator(FixedLocator([4, 6, 10, 20, 30]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    savefig(fig, out / "04_skala.png")
    summary["panjang_gelombang_px"] = st.panjang_gel.describe(percentiles=[.05, .5, .95]).round(2).to_dict()
    grp = {"garis halus (λ<8)": (st.koherensi > COH) & (st.panjang_gel < 8),
           "garis kasar (λ>=8)": (st.koherensi > COH) & (st.panjang_gel >= 8),
           "isotropik (koh<0.2)": st.koherensi < .2}
    summary["rasio_median_lambda_M2_M1"] = {
        k: round(float(st.panjang_gel[m & (mic == "M2")].median() / st.panjang_gel[m & (mic == "M1")].median()), 3)
        for k, m in grp.items()}

    # ---------- 05 beda mikroskop ----------
    y = (mic == "M2").astype(int)
    auc = {
        "gambar: brightness+kontras": mic_auc(st[["brightness", "kontras"]].to_numpy(), y, args.seed),
        "gambar: PSD (terstandardisasi)": mic_auc(np.log(psd[:, 1:] + 1e-6), y, args.seed),
        "gambar: rasio frek. tinggi": mic_auc(st[["rasio_hf"]].to_numpy(), y, args.seed),
        "spektrum mentah": mic_auc(spec, y, args.seed),
        "spektrum diproses": mic_auc(preprocess_spectrum(spec), y, args.seed),
    }
    summary["auc_tebak_mikroskop"] = {k: round(v, 3) for k, v in auc.items()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.75})
    rr = np.arange(1, R_MAX + 1)
    for mk in ("M1", "M2"):
        m_ = mic == mk
        axes[0].plot(rr, psd[m_, 1:].mean(0), color=MIC_COLOR[mk], label=mk)
    axes[0].set(xscale="log", yscale="log", xlabel="frekuensi (siklus / 64 px)", ylabel="daya",
                title="PSD radial rata-rata (terstandardisasi)")
    axes[0].legend()
    _hist_by_mic(axes[1], st.rasio_hf, mic, 60, f"energi r ≥ {HF_R} / total", "Ketajaman / noise")
    names = list(auc)
    axes[2].barh(names, [auc[n] for n in names], color=[SERIES[0]] * 3 + [SERIES[2]] * 2)
    axes[2].axvline(.5, color=MUTED, ls="--", lw=1)
    for j, n in enumerate(names):
        axes[2].text(auc[n] + .01, j, f"{auc[n]:.2f}", va="center", fontsize=7, color=TEXT2)
    axes[2].set(xlim=(.4, 1.05), xlabel="AUC (0.5 = tidak terbedakan)", title="Seberapa mudah mikroskop ditebak")
    axes[2].invert_yaxis()
    savefig(fig, out / "05_beda_mikroskop.png")

    # ---------- 06 contoh spektrum ----------
    ds, smask = despike(spec, return_mask=True)
    bl = baseline(ds)
    clean = preprocess_spectrum(spec)
    n_sp = smask.sum(1)
    pick = list(rng.choice(np.where(n_sp > 0)[0], 3, replace=False)) + list(rng.choice(len(spec), 3, replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), sharex=True)
    for ax, i in zip(axes.ravel(), pick):
        ax.plot(spec[i], color=MUTED, lw=1, label="mentah")
        ax.plot(bl[i], color=SERIES[3], lw=1.5, ls="--", label="baseline")
        ax.plot(ds[i] - bl[i], color=SERIES[0], lw=1.2, label="despike − baseline")
        sp = np.where(smask[i])[0]
        ax.scatter(sp, spec[i][sp], color=SERIES[7], s=18, zorder=3, label="spike")
        ax.set_title(f"sampel #{meta.indeks[i]}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("kanal")
    fig.suptitle("Spektrum: puncak komposisi + baseline lambat + spike 1 kanal", fontsize=10)
    savefig(fig, out / "06_contoh_spektrum.png")

    # ---------- 07 spike ----------
    from scipy.ndimage import median_filter
    resid_max = (spec - median_filter(spec, size=(1, SPIKE_WIN), mode="mirror")).max(1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    axes[0].hist(resid_max, bins=np.arange(0, resid_max.max() + .05, .05), color=SERIES[0])
    axes[0].axvline(SPIKE_THR, color=SERIES[7], ls="--", lw=1.2)
    axes[0].text(SPIKE_THR + .05, axes[0].get_ylim()[1] * .6, f"ambang {SPIKE_THR}", color=SERIES[7], fontsize=8)
    axes[0].set(yscale="log", xlabel=f"residual maks thd median {SPIKE_WIN} kanal",
                ylabel="jumlah spektrum (log)", title="Bimodal: noise ≤ 0.4, spike 0.7–4")
    axes[1].bar(np.arange(n_sp.max() + 1), np.bincount(n_sp), color=SERIES[0])
    axes[1].set(xlabel="jumlah spike per spektrum", ylabel="jumlah spektrum", yscale="log",
                title="Spike per spektrum")
    savefig(fig, out / "07_spike.png")
    summary["spike"] = {"pct_titik": float(100 * smask.mean()), "pct_spektrum_dgn_spike": float(100 * (n_sp > 0).mean()),
                        "jumlah_per_spektrum": np.bincount(n_sp).tolist(),
                        "residual_maks_celah": [float(resid_max[resid_max < SPIKE_THR].max()),
                                                float(resid_max[resid_max > SPIKE_THR].min())],
                        "residual_min": float((spec - median_filter(spec, size=(1, SPIKE_WIN), mode="mirror")).min())}

    # ---------- 08 baseline & noise ----------
    coef = baseline_coef(ds)                         # [a2, a1, a0] di x in [-1, 1]
    noise = (ds - median_filter(ds, size=(1, 5), mode="mirror")).std(1)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, j, lab in zip(axes[:3], [2, 1, 0], ["offset (a0)", "kemiringan (a1)", "lengkung (a2)"]):
        ax.hist(coef[:, j], bins=80, color=SERIES[0])
        ax.set(xlabel=lab, title=f"Baseline: {lab.split()[0]}")
    axes[0].set_ylabel("jumlah spektrum")
    axes[3].hist(noise, bins=np.linspace(noise.min(), np.quantile(noise, .999), 80), color=SERIES[2])
    axes[3].set(xlabel="std residual halus", title="Level noise per spektrum")
    fig.suptitle("Baseline polinom-2 (modpoly) per spektrum: offset, kemiringan, dan lengkung sangat bervariasi",
                 fontsize=9, y=1.06)
    savefig(fig, out / "08_baseline_noise.png")
    summary["baseline"] = {k: pd.Series(coef[:, j]).describe(percentiles=[.05, .5, .95]).round(3).to_dict()
                           for k, j in [("offset", 2), ("kemiringan", 1), ("lengkung", 0)]}
    summary["noise_std"] = pd.Series(noise).describe(percentiles=[.05, .5, .95]).round(4).to_dict()

    # ---------- 09 posisi puncak, shift, intensitas ----------
    from scipy.signal import find_peaks
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    corr = ds - bl
    peaks, n_peaks, trip = [], [], []
    for s_ in corr:
        p, _ = find_peaks(s_, prominence=0.25 * s_.max(), distance=4)
        peaks.extend(p)
        n_peaks.append(len(p))
        if len(p) == 3:
            trip.append(p)
    height = corr.max(1)
    X3 = np.array(trip, float)
    # Pola 3 puncak dikelompokkan (KMeans, k dipilih silhouette). Di dalam kelompok: jika shift adalah
    # translasi GLOBAL per spektrum, ketiga puncak bergeser bersama -> deviasinya berkorelasi tinggi dan
    # jarak antar-puncak lebih stabil daripada posisi absolut.
    sub = rng.choice(len(X3), min(len(X3), 6000), replace=False)
    ks = list(range(2, 11))
    sils, fits = [], {}
    for k in ks:
        fits[k] = KMeans(k, n_init=4, random_state=args.seed).fit(X3)
        sils.append(silhouette_score(X3[sub], fits[k].labels_[sub]))
    kb = ks[int(np.argmax(sils))]
    dev = X3 - fits[kb].cluster_centers_[fits[kb].labels_]
    shift = {"k_pola_puncak": kb, "silhouette": dict(zip(ks, np.round(sils, 3).tolist())),
             "std_posisi_absolut": dev.std(0).round(2).tolist(),
             "std_jarak_p2_p1": float((dev[:, 1] - dev[:, 0]).std()),
             "std_jarak_p3_p1": float((dev[:, 2] - dev[:, 0]).std()),
             "korelasi_deviasi_p1_p2": float(np.corrcoef(dev[:, 0], dev[:, 1])[0, 1]),
             "korelasi_deviasi_p1_p3": float(np.corrcoef(dev[:, 0], dev[:, 2])[0, 1]),
             "shift_global_p2.5_p97.5": np.quantile(dev.mean(1), [.025, .975]).round(2).tolist()}
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), gridspec_kw={"width_ratios": [1.6, .8, 1, 1], "wspace": .35})
    axes[0].hist(peaks, bins=np.arange(0, 129), color=SERIES[0])
    axes[0].set(xlabel="posisi puncak (kanal)", ylabel="jumlah", title="Posisi semua puncak: pusat-pusat yang melebar")
    axes[1].plot(ks, sils, "o-", color=SERIES[0])
    axes[1].axvline(kb, color=SERIES[7], ls="--", lw=1)
    axes[1].set(xlabel="k", ylabel="silhouette", title=f"Pola 3 puncak: k = {kb}")
    j = rng.choice(len(dev), min(len(dev), 3000), replace=False)
    axes[2].scatter(dev[j, 0], dev[j, 1] + rng.uniform(-.3, .3, len(j)), s=2, alpha=.3, color=SERIES[0])
    axes[2].plot([-6, 6], [-6, 6], color=MUTED, ls="--", lw=1)
    axes[2].set(xlabel="deviasi puncak 1 (kanal)", ylabel="deviasi puncak 2",
                title=f"Bergeser bersama: r = {shift['korelasi_deviasi_p1_p2']:.2f}")
    axes[3].hist(height, bins=80, color=SERIES[2])
    axes[3].set(xlabel="tinggi puncak tertinggi", title="Intensitas (→ normalisasi L2)")
    savefig(fig, out / "09_puncak_shift_intensitas.png")
    summary["puncak"] = {"jumlah_puncak_per_spektrum": dict(zip(*[a.tolist() for a in np.unique(n_peaks, return_counts=True)])),
                         "tinggi_puncak": pd.Series(height).describe(percentiles=[.05, .5, .95]).round(3).to_dict()}
    summary["shift"] = shift

    # ---------- 10 spektrum per mikroskop ----------
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for mk in ("M1", "M2"):
        m_ = mic == mk
        ax.plot(clean[m_].mean(0), color=MIC_COLOR[mk], label=f"{mk} rata-rata")
        ax.fill_between(np.arange(clean.shape[1]), *np.percentile(clean[m_], [25, 75], 0),
                        color=MIC_COLOR[mk], alpha=.15)
    ax.set(xlabel="kanal", ylabel="intensitas (L2-norm)",
           title="Spektrum setelah preprocessing: M1 ≈ M2 (mikroskop hanya memengaruhi gambar)")
    ax.legend()
    savefig(fig, out / "10_spektrum_per_mikroskop.png")

    st.assign(indeks=meta.indeks, jumlah_spike=n_sp, noise_spektrum=noise).round(4).to_csv(
        out / "ukuran_per_sampel.csv", index=False)
    save_json(summary, out / "ringkasan_eda.json")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if not args.debug:
        dst = args.results_dir / "task3" / "eda"
        dst.mkdir(parents=True, exist_ok=True)
        for f in out.iterdir():
            if f.suffix in (".png", ".json"):           # ukuran_per_sampel.csv (36 rb baris) tidak disalin
                shutil.copy2(f, dst / f.name)
        print(f"[task3-eda] disalin ke {dst}")
    return summary


if __name__ == "__main__":
    run(add_common_args(argparse.ArgumentParser(description="Task 3 EDA")).parse_args())
