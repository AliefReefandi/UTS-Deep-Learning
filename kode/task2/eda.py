"""Task 2a - EDA & preprocessing. Semua grafik/tabel ke <out_dir>/task2/eda/.

  python -m kode.task2.eda [--debug] [--data_dir ...] [--out_dir ...]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from kode.common import add_common_args, resolve_paths, set_seed
from kode.common.plot import MUTED, SERIES, plt, savefig
from kode.common.utils import save_json
from kode.task2.data import SENSORS, SETTINGS, Preprocessor, add_rul_label, load_raw


def run(args):
    args = resolve_paths(args, "task2")
    set_seed(args.seed)
    out = args.task_out / "eda"
    out.mkdir(parents=True, exist_ok=True)
    tr, te = load_raw(args)
    tr = add_rul_label(tr)
    print(f"[task2-eda] train {tr.id_unit.nunique()} unit / {len(tr)} baris; "
          f"test {te.id_unit.nunique()} unit / {len(te)} baris")
    summary = {}

    # ---------- 1. Panjang histori ----------
    life = tr.groupby("id_unit").siklus.max()
    tlen = te.groupby("id_unit").siklus.max()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bins = np.linspace(0, max(life.max(), tlen.max()), 40)
    ax.hist(life, bins=bins, alpha=.85, label=f"train: umur total (n={len(life)})", color=SERIES[0])
    ax.hist(tlen, bins=bins, alpha=.85, label=f"test: panjang histori (n={len(tlen)})", color=SERIES[1])
    ax.set(xlabel="siklus", ylabel="jumlah mesin", title="Panjang histori per mesin")
    ax.legend()
    savefig(fig, out / "01_panjang_histori.png")
    summary["umur_train"] = life.describe().round(1).to_dict()
    summary["panjang_test"] = tlen.describe().round(1).to_dict()
    summary["frac_train_rul_raw_gt_cap"] = float((tr.rul_raw > 150).mean())

    # ---------- 2. Kondisi operasi ----------
    kond_tab = tr.groupby("kond")[SETTINGS].agg(["mean", "std"]).round(4)
    kond_tab["n_baris"] = tr.groupby("kond").size()
    kond_tab.to_csv(out / "tabel_kondisi_operasi.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (k, g) in enumerate(tr.groupby("kond")):
        s = g.sample(min(len(g), 1500), random_state=0)
        ax.scatter(s.setel_1, s.setel_2, s=10, color=SERIES[i], label=f"kond {k} (setel_3={g.setel_3.iloc[0]:.0f})")
    ax.set(xlabel="setel_1", ylabel="setel_2", title="6 rezim operasi diskret")
    ax.legend(loc="lower right")
    savefig(fig, out / "02_kondisi_operasi.png")

    # ---------- 3. Pengaruh kondisi pada sensor ----------
    g = tr.groupby("kond")[SENSORS]
    eta2 = 1 - (tr[SENSORS] - g.transform("mean")).var() / tr[SENSORS].var()
    unit = tr.id_unit.iloc[0]
    u = tr[tr.id_unit == unit]
    pp_all = Preprocessor(select=False).fit(tr)            # semua sensor, hanya untuk ilustrasi
    zu = pp_all.transform(u)
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.4), sharex=True)
    for j, s in enumerate(["s04", "s11"]):
        for k in sorted(u.kond.unique()):
            m = u.kond == k
            axes[0, j].scatter(u.siklus[m], u[s][m], s=4, color=SERIES[sorted(u.kond.unique()).index(k)])
        axes[0, j].set_title(f"{s} mentah (warna = kondisi)")
        axes[1, j].plot(zu.siklus, zu[s], color=SERIES[0], lw=1)
        axes[1, j].plot(zu.siklus, zu[s].rolling(15, min_periods=1).mean(), color=SERIES[1], lw=2,
                        label="rata-rata bergerak 15")
        axes[1, j].set_title(f"{s} setelah z-score per kondisi")
        axes[1, j].set_xlabel("siklus")
    axes[1, 0].legend()
    fig.suptitle(f"Mesin {unit}: tren degradasi baru terlihat setelah efek kondisi dihapus", fontsize=10)
    savefig(fig, out / "03_efek_kondisi_mentah_vs_norm.png")

    # ---------- 4. Sensor informatif vs tidak ----------
    pp = Preprocessor().fit(tr)
    z = pp_all._normalize(tr)
    corr = z.corrwith(tr.rul, method="spearman")                       # global (semua baris)
    zu_ = z.assign(_r=tr.rul.values, _u=tr.id_unit.values)
    corr_within = zu_.groupby("_u").apply(                             # rata-rata |korelasi| per mesin
        lambda g: g[SENSORS].corrwith(g._r, method="spearman").abs()).mean()
    tab = pd.DataFrame({
        "eta2_kondisi": eta2, "noise_ratio_diff": pp.noise_ratio_,
        "spearman_global": corr, "abs_spearman_per_mesin": corr_within,
        "missing_pct": tr[SENSORS].isna().mean() * 100,
        "dipakai": [s in pp.sensors_ for s in SENSORS]}).round(4)
    tab.index.name = "sensor"
    tab.to_csv(out / "tabel_sensor.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    col = [SERIES[0] if s in pp.sensors_ else MUTED for s in SENSORS]
    axes[0].bar(SENSORS, pp.noise_ratio_, color=col)
    axes[0].axhline(np.sqrt(2), color=MUTED, lw=1, ls=":")
    axes[0].axhline(pp.noise_thr, color=SERIES[7], lw=1, ls="--")
    axes[0].text(20.4, np.sqrt(2) + .03, "√2 = white noise", ha="right", fontsize=8, color=MUTED)
    axes[0].set(ylabel="std(Δz) / std(z)", title="Uji white noise (tanpa label)")
    axes[1].bar(SENSORS, corr_within, color=col)
    axes[1].set(ylabel="|Spearman| per mesin, rata-rata", title="Korelasi dengan RUL di dalam tiap mesin")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
    fig.suptitle("Sensor abu-abu = dibuang (noise murni)", fontsize=10)
    savefig(fig, out / "04_seleksi_sensor.png")
    summary["sensor_dipakai"] = pp.sensors_
    summary["sensor_dibuang"] = [s for s in SENSORS if s not in pp.sensors_]

    # ---------- 5. Dua mode kerusakan & profil degradasi ----------
    zt = pp.transform(tr)
    # mode ditentukan dari arah s11 pada 10 siklus terakhir (hanya untuk analisis, BUKAN fitur)
    near = zt[tr.rul_raw < 10].groupby("id_unit")["s11"].mean()
    mode = (near > 0).map({True: "B (s11 naik)", False: "A (s11 turun)"})
    summary["mode_kerusakan_n"] = mode.value_counts().to_dict()
    mode.rename("mode").to_csv(out / "mode_kerusakan_per_unit.csv")
    bins_r = (tr.rul_raw.clip(upper=300) // 10) * 10
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    near_all = zt[tr.rul_raw < 10].groupby(tr.id_unit.map(mode))[pp.sensors_].mean()
    for ax, (mname, idx) in zip(axes, mode.groupby(mode).groups.items()):
        m = tr.id_unit.isin(idx)
        prof = zt.loc[m, pp.sensors_].groupby(bins_r[m]).mean()
        for s in pp.sensors_:
            v = near_all.loc[mname, s]
            c = SERIES[0] if v > 1 else SERIES[1] if v < -1 else MUTED
            ax.plot(prof.index, prof[s], color=c, lw=1.3)
            if abs(v) > 1:
                ax.text(-3, prof[s].iloc[0], s, fontsize=7, color=c, va="center")
        ax.axvline(150, color=MUTED, ls="--", lw=1)
        ax.invert_xaxis()
        ax.set(title=f"Mode {mname}, n={len(idx)}", xlabel="RUL sebenarnya (dipotong 300)")
    axes[0].set_ylabel("z rata-rata")
    fig.suptitle("Dua mode kerusakan: sensor yang terdegradasi berbeda (biru naik, oranye turun, "
                 "abu-abu datar). Sensor datar sampai RUL ≈ 150 → alasan cap 150", fontsize=9)
    savefig(fig, out / "05_profil_degradasi_per_mode.png")

    # ---------- 6. Missing values ----------
    miss = tr[SENSORS].isna()
    rul_bin = pd.cut(tr.rul, [-1, 25, 50, 75, 100, 125, 149, 150])
    miss_by_rul = miss.any(axis=1).groupby(rul_bin, observed=True).mean() * 100
    summary["missing_pct_sel_train"] = float(miss.values.mean() * 100)
    summary["missing_pct_sel_test"] = float(te[SENSORS].isna().values.mean() * 100)
    summary["baris_dgn_missing_pct"] = float(miss.any(axis=1).mean() * 100)
    summary["missing_per_baris_max"] = int(miss.sum(axis=1).max())
    # apakah missing berurutan? panjang run NaN maksimal
    runs = miss.groupby(tr.id_unit).apply(lambda m: (m & m.shift(1, fill_value=False)).values.sum())
    summary["pasangan_nan_berurutan"] = int(runs.sum())
    summary["pasangan_nan_berurutan_jika_acak"] = float(
        (miss.mean() ** 2 * (len(tr) - tr.id_unit.nunique())).sum())
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    axes[0].bar(SENSORS, miss.mean() * 100, color=SERIES[0])
    axes[0].set(ylabel="% NaN", title="Missing per sensor (train)")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar([str(b) for b in miss_by_rul.index], miss_by_rul.values, color=SERIES[0])
    axes[1].set(ylabel="% baris dgn ≥1 NaN", xlabel="rentang RUL", title="Missing tidak bergantung RUL")
    axes[1].tick_params(axis="x", rotation=45)
    savefig(fig, out / "06_missing.png")

    # ---------- 7. Spike ----------
    _, _, spike = pp.transform(tr, return_masks=True)
    summary["spike_pct"] = float(spike.values.mean() * 100)
    summary["spike_pct_per_sensor"] = (spike.mean() * 100).round(3).to_dict()
    u_sp = spike.groupby(tr.id_unit).sum().sum(axis=1).idxmax()
    s_sp = spike[tr.id_unit == u_sp].sum().idxmax()
    m = tr.id_unit == u_sp
    zraw = z.loc[m, s_sp]
    zcl = zt.loc[m, s_sp]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(tr.siklus[m], zraw, color=MUTED, lw=1, label="z sebelum")
    ax.plot(tr.siklus[m], zcl, color=SERIES[0], lw=1.2, label="setelah hapus spike + ffill + clip")
    sp = spike.loc[m, s_sp]
    ax.scatter(tr.siklus[m][sp], zraw[sp], color=SERIES[7], s=20, zorder=3, label="spike terdeteksi")
    ax.set(xlabel="siklus", ylabel="z", title=f"Deteksi spike kausal - mesin {u_sp}, {s_sp}")
    ax.legend(ncol=3, loc="lower left")
    savefig(fig, out / "07_spike.png")

    # ---------- 8. Distribusi label ----------
    last_rul_hist = tr.rul
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.hist(last_rul_hist, bins=30, color=SERIES[0])
    ax.set(xlabel="RUL (cap 150)", ylabel="jumlah baris", title="Distribusi label train (banyak di cap 150)")
    savefig(fig, out / "08_distribusi_label.png")

    save_json(summary, out / "ringkasan_eda.json")
    print(tab.to_string())
    for k, v in summary.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    run(add_common_args(argparse.ArgumentParser(description="Task 2 EDA")).parse_args())
