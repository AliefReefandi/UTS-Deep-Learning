"""Task 2f - prediction interval (MC dropout + kalibrasi) dan kebijakan kapan mesin diservis.

Sumber ketidakpastian
- MC dropout: dropout tetap aktif saat inferensi, `mc` forward pass per sampel -> sebaran prediksi
  satu model (ketidakpastian epistemik, pendekatan Bayesian Gal & Ghahramani 2016).
- Test: 5 model fold x `mc` pass digabung (deep ensemble + MC dropout), sehingga perbedaan antar
  model fold ikut masuk ke std.

Kalibrasi (kenapa perlu)
MC dropout biasanya terlalu percaya diri (interval terlalu sempit), karena dropout 0.2 tidak
memodelkan noise data (aleatorik). Maka interval dibentuk sebagai
    [pred + q_lo * std, pred + q_hi * std]
dengan q_lo, q_hi = kuantil residual ternormalisasi z = (aktual - pred) / std pada data validasi
out-of-fold (split conformal ternormalisasi). Cakupan dilaporkan secara CROSS-FIT: q untuk fold k
diestimasi dari fold lain saja, supaya angka cakupan tidak dikalibrasi di data yang sama.

Kebijakan servis
Mesin perlu diservis sebelum rusak, dan servis butuh waktu persiapan H siklus (default 20).
Aturan: servis jika BATAS BAWAH interval <= H. Batas bawah 90% (kuantil 5%) berarti
"kemungkinan mesin rusak sebelum H siklus lagi sekitar <= 5% jika tidak diservis sekarang".
Dievaluasi pada validasi: berapa kerusakan yang terlewat vs berapa servis yang terlalu dini.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kode.common.utils import save_json
from kode.task2.evaluate import RUL_BINS, RUL_LABELS

LEVELS = [0.5, 0.8, 0.9, 0.95]
STD_FLOOR = 1.0     # hindari std ~0 (mis. semua pass ter-clip di 150)
CAP = 150.0


def _q(z, level):
    a = (1 - level) / 2
    return float(np.quantile(z, a)), float(np.quantile(z, 1 - a))


def load_oof(mdir, name):
    """Gabungkan validasi out-of-fold + sampel MC dropout tiap fold."""
    vals, te_mc, te_det = [], [], []
    for f in sorted((mdir / name).glob("fold*_val.csv")):
        k = int(f.stem.split("_")[0][4:])
        with np.load(mdir / name / f"fold{k}_mc.npz") as z:     # disimpan float16 -> hitung di float32
            mc = {key: z[key].astype(np.float32) for key in ("va", "te")}
        v = pd.read_csv(f)
        v["mc_mean"] = mc["va"].mean(0)
        v["mc_std"] = np.maximum(mc["va"].std(0), STD_FLOOR)
        for L in LEVELS:
            v[f"raw_lo{int(L * 100)}"] = np.quantile(mc["va"], (1 - L) / 2, axis=0)
            v[f"raw_hi{int(L * 100)}"] = np.quantile(mc["va"], (1 + L) / 2, axis=0)
        vals.append(v)
        te_mc.append(mc["te"])
        te_det.append(pd.read_csv(mdir / name / f"fold{k}_test.csv").set_index("id_unit").pred)
    oof = pd.concat(vals, ignore_index=True)
    oof["z"] = (oof.rul - oof.pred) / oof.mc_std
    return oof, np.concatenate(te_mc, 0), pd.concat(te_det, axis=1)


def calibrate(oof):
    """Cakupan interval MC mentah vs terkalibrasi (cross-fit per fold)."""
    rows = []
    for L in LEVELS:
        tag = int(L * 100)
        raw = ((oof.rul >= oof[f"raw_lo{tag}"]) & (oof.rul <= oof[f"raw_hi{tag}"])).mean()
        lo = np.empty(len(oof))
        hi = np.empty(len(oof))
        for k in oof.fold.unique():
            m = (oof.fold == k).to_numpy()
            ql, qh = _q(oof.z[~m], L)
            lo[m] = np.clip(oof.pred[m] + ql * oof.mc_std[m], 0, CAP)
            hi[m] = np.clip(oof.pred[m] + qh * oof.mc_std[m], 0, CAP)
        oof[f"lo{tag}"], oof[f"hi{tag}"] = lo, hi
        cal = ((oof.rul >= lo) & (oof.rul <= hi)).mean()
        rows.append({"level": L, "cakupan_mc_mentah": raw, "cakupan_terkalibrasi": cal,
                     "lebar_mc_mentah": float((oof[f"raw_hi{tag}"] - oof[f"raw_lo{tag}"]).mean()),
                     "lebar_terkalibrasi": float((hi - lo).mean()),
                     "q_lo": _q(oof.z, L)[0], "q_hi": _q(oof.z, L)[1]})
    return pd.DataFrame(rows)


def per_range(oof, tag=90):
    b = pd.cut(oof.rul, RUL_BINS, labels=RUL_LABELS, right=False)
    g = oof.assign(cov=(oof.rul >= oof[f"lo{tag}"]) & (oof.rul <= oof[f"hi{tag}"]),
                   width=oof[f"hi{tag}"] - oof[f"lo{tag}"]).groupby(b, observed=True)
    return pd.DataFrame({"n": g.size(), f"cakupan_{tag}": g["cov"].mean(),
                         f"lebar_{tag}": g["width"].mean(), "std_mc": g["mc_std"].mean()}).reset_index(
        names="rentang_rul")


def policy_table(oof, H):
    """Bandingkan aturan servis: titik prediksi vs batas bawah interval."""
    need = oof.rul <= H
    rows = []
    rules = {"prediksi titik <= H": oof.pred}
    rules.update({f"batas bawah {int(L * 100)}% <= H": oof[f"lo{int(L * 100)}"] for L in LEVELS})
    for name, bound in rules.items():
        flag = bound <= H
        rows.append({"aturan": name, "H": H,
                     "kerusakan_tertangkap_pct": 100 * (flag & need).sum() / max(need.sum(), 1),
                     "kerusakan_terlewat": int((~flag & need).sum()),
                     "servis_terlalu_dini": int((flag & ~need).sum()),
                     "rata2_sisa_umur_saat_servis": float(oof.rul[flag].mean()) if flag.any() else np.nan})
    return pd.DataFrame(rows)


def plot_policy(pol, H, path):
    """Grafik 15: trade-off kerusakan terlewat vs umur terbuang, dari tabel_kebijakan_servis."""
    from kode.common.plot import SERIES, TEXT2, plt, savefig
    p = pol[pol.H == H]
    miss = (100 - p.kerusakan_tertangkap_pct).to_numpy()
    life = p.rata2_sisa_umur_saat_servis.to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(life, miss, "o-", color=SERIES[0], zorder=3)
    for a, x, y in zip(p.aturan, life, miss):
        lab = "prediksi titik" if a.startswith("prediksi") else "lo " + a.split()[2]
        ax.annotate(f"{lab}: {y:.1f}%", (x, y), fontsize=7, color=TEXT2, xytext=(6, 5),
                    textcoords="offset points")
    ax.set_xlim(life.min() - 1, life.max() + 3.5)
    ax.set(xlabel="rata-rata sisa umur saat diservis (siklus terbuang)",
           ylabel=f"% kerusakan terlewat (RUL aktual <= {H})",
           title=f"Kebijakan servis (H = {H}); lo X% = batas bawah interval X%")
    savefig(fig, path)


def report(args, mdir, best, log=print):
    from kode.common.plot import MUTED, SERIES, TEXT, plt, savefig
    out = mdir / "uncertainty"
    out.mkdir(exist_ok=True)
    H = args.service_h
    oof, te_mc, te_det = load_oof(mdir, best)
    cal = calibrate(oof)
    rng_tab = per_range(oof)
    pol = pd.concat([policy_table(oof, h) for h in sorted({10, H, 30})], ignore_index=True)
    cal.round(3).to_csv(out / "tabel_kalibrasi_interval.csv", index=False)
    rng_tab.round(3).to_csv(out / "tabel_interval_per_rentang.csv", index=False)
    pol.round(2).to_csv(out / "tabel_kebijakan_servis.csv", index=False)
    log("\n[task2-f] interval " + best + "\n" + cal.round(3).to_string(index=False))
    log(rng_tab.round(2).to_string(index=False))
    log(pol[pol.H == H].round(2).to_string(index=False))

    # ---- interval test: pusat = rata-rata prediksi deterministik 5 fold; std = MC x fold ----
    q_lo, q_hi = _q(oof.z, 0.9)
    center = te_det.mean(axis=1).clip(0, CAP)
    std = np.maximum(te_mc.std(0), STD_FLOOR)
    te = pd.DataFrame({"id_unit": te_det.index, "rul": center.values.round(2),
                       "rul_lo90": np.clip(center.values + q_lo * std, 0, CAP).round(2),
                       "rul_hi90": np.clip(center.values + q_hi * std, 0, CAP).round(2),
                       "std": std.round(2)})
    te["status"] = np.select([te.rul_lo90 <= H, te.rul_lo90 <= 2.5 * H],
                             ["SERVIS SEGERA", "JADWALKAN"], "AMAN")
    te.to_csv(out / "prediksi_rul_interval_test.csv", index=False)
    counts = te.status.value_counts().to_dict()
    log(f"[task2-f] status test (H={H}): {counts}")

    # ---- grafik 13: kalibrasi ----
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([0.4, 1], [0.4, 1], color=MUTED, ls="--", lw=1)
    ax.plot(cal.level, cal.cakupan_mc_mentah, "o-", color=SERIES[1], label="MC dropout mentah")
    ax.plot(cal.level, cal.cakupan_terkalibrasi, "o-", color=SERIES[0], label="terkalibrasi (cross-fit)")
    ax.set(xlabel="cakupan nominal", ylabel="cakupan empiris (val)", title=f"Kalibrasi interval - {best}")
    ax.legend()
    savefig(fig, out / "13_kalibrasi_interval.png")

    # ---- grafik 14: contoh interval pada validasi ----
    s = oof.sample(min(len(oof), 200), random_state=0).sort_values("rul")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(s))
    cov = (s.rul >= s.lo90) & (s.rul <= s.hi90)
    ax.vlines(x, s.lo90, s.hi90, color=SERIES[0], alpha=.35, lw=2, label="interval 90%")
    ax.scatter(x, s.pred, s=6, color=SERIES[0], label="prediksi")
    ax.scatter(x[~cov], s.rul[~cov], s=10, color=SERIES[7], zorder=3, label="aktual (di luar interval)")
    ax.scatter(x[cov], s.rul[cov], s=6, color=TEXT, zorder=3, label="aktual")
    ax.axhline(H, color=SERIES[3], ls=":", lw=1)
    ax.text(1, H + 2, f"H = {H}", fontsize=7, color=SERIES[3])
    ax.set(xlabel="sampel validasi (urut RUL aktual)", ylabel="RUL", title="Interval prediksi 90% (val, acak 200)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    savefig(fig, out / "14_contoh_interval_val.png")

    # ---- grafik 15: trade-off kebijakan servis ----
    plot_policy(pol, H, out / "15_kebijakan_servis.png")

    summary = {"model": best, "mc_pass": int(te_mc.shape[0] // max(te_det.shape[1], 1)),
               "n_model_fold": int(te_det.shape[1]), "q90": [q_lo, q_hi], "H": H,
               "status_test": counts, "kalibrasi": cal.round(4).to_dict("records")}
    save_json(summary, out / "ringkasan_uncertainty.json")
    return summary


