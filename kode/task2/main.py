"""Task 2 - Prediksi Remaining Useful Life (RUL), cap 150.

  python -m kode.task2.main [--debug] [--models baseline lstm tcn transformer] [--run_folds 2] ...

Alur per fold (split per mesin, 5-fold):
  fit Preprocessor di mesin train -> window (W=50, padding+mask) -> latih tiap model
  -> evaluasi pada val terpotong (meniru test) -> prediksi test.
Prediksi test akhir = rata-rata model fold (ensemble 5 model) dari arsitektur DL terbaik
(RMSE val gabungan; baseline non-DL hanya pembanding). Lalu 2f: interval MC dropout terkalibrasi
+ kebijakan servis (uncertainty.py). Hasil final disalin ke --results_dir/task2 dan --models_dir.
Resume: tiap (model, fold) yang sudah selesai disimpan ke <task_out>/models/ dan dilewati;
training yang terputus dilanjutkan dari checkpoint per epoch.
Output: <pred_dir>/prediksi_rul.csv (id_unit,rul ; rul di 0..150) + tabel/grafik di <task_out>/models/.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time

import numpy as np
import pandas as pd

from kode.common import add_common_args, get_device, resolve_paths, set_seed
from kode.common.utils import save_json
from kode.task2.data import Preprocessor, add_rul_label, load_raw
from kode.task2.evaluate import by_group, by_range, summarize
from kode.task2.windows import (UnitArrays, WindowBank, all_end_indices, last_indices,
                                truncated_indices, unit_folds)

PRED_FILE = "prediksi_rul.csv"
RUL_CAP = 150


def asymmetric_score(y_pred, y_true):   # dipertahankan untuk kompatibilitas impor lama
    from kode.task2.evaluate import asymmetric_score as f
    return f(y_pred, y_true)


def failure_modes(tr: pd.DataFrame) -> pd.Series:
    """Mode kerusakan per mesin train (A: s11 turun, B: s11 naik) - HANYA untuk analisis error."""
    pp = Preprocessor(select=False).fit(tr)
    z = pp.transform(tr)
    near = z[tr.rul_raw < 10].groupby("id_unit")["s11"].mean()
    return (near > 0).map({True: "B", False: "A"}).rename("mode")


def preprocessor_state(pp, W) -> dict:
    """Statistik yang di-fit di mesin train fold ini; dibutuhkan untuk inferensi ulang model."""
    return {"W": W, "sensors": pp.sensors_, "fitur": pp.sensors_ + ["umur(siklus/300)", "mask"],
            "mean_per_kond": pp.mean_[pp.sensors_].to_dict("index"),
            "std_per_kond": pp.std_[pp.sensors_].to_dict("index"),
            "sigma_spike": pp.sigma_.to_dict(), "spike_k": pp.spike_k, "spike_win": pp.spike_win,
            "clip": pp.clip}


def config(args) -> dict:
    cfg = dict(epochs=args.epochs, samples_per_epoch=args.samples_per_epoch, bs=args.bs, lr=args.lr,
               patience=args.patience, resume=args.resume)
    if args.debug:
        cfg.update(epochs=2, samples_per_epoch=2048, bs=256, patience=5)
    return cfg


def run_fold(k, tr_units, va_units, tr, te, args, cfg, device, mdir, log):
    tr_f = tr[tr.id_unit.isin(tr_units)]
    va_f = tr[tr.id_unit.isin(va_units)]
    pp = Preprocessor().fit(tr_f)
    ua_tr = UnitArrays(pp.transform(tr_f), pp.sensors_)
    ua_va = UnitArrays(pp.transform(va_f), pp.sensors_)
    ua_te = UnitArrays(pp.transform(te), pp.sensors_)
    banks = {n: WindowBank(u, args.W, device) for n, u in [("tr", ua_tr), ("va", ua_va), ("te", ua_te)]}
    idx_tr = all_end_indices(ua_tr)
    idx_va = truncated_indices(ua_va, n_cuts=5, min_len=50, seed=args.seed + k)
    idx_te = last_indices(ua_te)
    y_va = banks["va"].labels(idx_va)
    meta_va = pd.DataFrame({"id_unit": [ua_va.units[i] for i, _ in idx_va],
                            "siklus": [int(ua_va.cycles[i][t]) for i, t in idx_va], "rul": y_va})

    for name in args.models:
        done_file = mdir / name / f"fold{k}.json"
        if done_file.exists() and args.resume:
            log(f"  [{name} fold {k}] sudah selesai, dilewati")
            continue
        (mdir / name).mkdir(parents=True, exist_ok=True)
        log(f"  [{name} fold {k}] mulai")
        t0 = time.time()
        if name == "baseline":
            from kode.task2.baseline import fit_baseline, predict_baseline
            m = fit_baseline(banks["tr"], idx_tr, seed=args.seed + k, debug=args.debug)
            p_va, p_te = predict_baseline(m, banks["va"], idx_va), predict_baseline(m, banks["te"], idx_te)
            extra = {"n_params": None, "epochs": None}
        else:
            from kode.task2.models import n_params
            from kode.task2.train import predict, train_model
            ckpt = args.ckpt_dir / f"{name}_W{args.W}_fold{k}.pt"
            m, st = train_model(name, banks["tr"], idx_tr, banks["va"], idx_va, cfg, ckpt, device,
                                seed=args.seed + k, log=log)
            p_va, p_te = predict(m, banks["va"], idx_va), predict(m, banks["te"], idx_te)
            # 2f: MC dropout - `mc` forward pass dengan dropout aktif (seed tetap -> reproducible)
            import torch
            torch.manual_seed(args.seed + 1000 + k)
            np.savez_compressed(mdir / name / f"fold{k}_mc.npz",
                                va=predict(m, banks["va"], idx_va, mc=args.mc).astype(np.float16),
                                te=predict(m, banks["te"], idx_te, mc=args.mc).astype(np.float16))
            torch.save(m.state_dict(), mdir / name / f"fold{k}_best.pt")
            save_json(preprocessor_state(pp, args.W), mdir / name / f"fold{k}_preprocessor.json")
            extra = {"n_params": n_params(m), "epochs": st["epoch"], "history": st["history"]}
        sec = time.time() - t0
        meta_va.assign(pred=p_va, fold=k).to_csv(mdir / name / f"fold{k}_val.csv", index=False)
        pd.DataFrame({"id_unit": ua_te.units, "pred": p_te}).to_csv(mdir / name / f"fold{k}_test.csv", index=False)
        res = {"model": name, "fold": k, "sec": sec, **summarize(p_va, y_va), **extra}
        save_json(res, done_file)
        log(f"  [{name} fold {k}] RMSE {res['rmse']:.2f} | skor/300 {res['skor_per_300']:.0f} | {sec:.0f}s")


def fold_results(d):
    """Ringkasan hasil per fold: fold<k>.json saja (bukan fold<k>_preprocessor.json)."""
    return sorted(f for f in d.glob("fold*.json") if f.stem[4:].isdigit())


def summarize_all(args, mdir, modes, log):
    from kode.common.plot import SERIES, plt, savefig
    rows, oof, ranges, modes_tab = [], {}, [], []
    for name in args.models:
        files = fold_results(mdir / name)
        if not files:
            continue
        res = [json.loads(f.read_text()) for f in files]
        val = pd.concat([pd.read_csv(mdir / name / f"fold{r['fold']}_val.csv") for r in res])
        val["mode"] = val.id_unit.map(modes)
        oof[name] = val
        s = summarize(val.pred, val.rul)
        rows.append({"model": name, "n_fold": len(res),
                     "rmse_mean_fold": np.mean([r["rmse"] for r in res]),
                     "rmse_std_fold": np.std([r["rmse"] for r in res]),
                     "rmse_gabungan": s["rmse"], "mae": s["mae"], "bias": s["bias"],
                     "skor_per_300": s["skor_per_300"],
                     "n_params": res[0].get("n_params"), "detik_per_fold": np.mean([r["sec"] for r in res])})
        ranges.append(by_range(val).assign(model=name))
        modes_tab.append(by_group(val, "mode").assign(model=name))
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(mdir / "tabel_perbandingan_model.csv", index=False)
    rng_tab = pd.concat(ranges).round(3)
    rng_tab.to_csv(mdir / "tabel_per_rentang_rul.csv", index=False)
    pd.concat(modes_tab).round(3).to_csv(mdir / "tabel_per_mode.csv", index=False)
    log("\n" + tab.to_string(index=False))

    # Grafik 10: RMSE per rentang RUL per model
    fig, ax = plt.subplots(figsize=(7, 3))
    labs = rng_tab.rentang_rul.unique()
    wbar = 0.8 / len(oof)
    for j, name in enumerate(oof):
        g = rng_tab[rng_tab.model == name].set_index("rentang_rul").reindex(labs)
        ax.bar(np.arange(len(labs)) + j * wbar, g.rmse, wbar * 0.92, label=name, color=SERIES[j])
    ax.set_xticks(np.arange(len(labs)) + wbar * (len(oof) - 1) / 2, labs)
    ax.set(xlabel="RUL sebenarnya", ylabel="RMSE", title="RMSE per rentang RUL (val terpotong, semua fold)")
    ax.legend()
    savefig(fig, mdir / "10_rmse_per_rentang.png")

    # Grafik 11: prediksi vs aktual
    fig, axes = plt.subplots(1, len(oof), figsize=(3.4 * len(oof), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, v), c in zip(axes, oof.items(), SERIES):
        ax.scatter(v.rul, v.pred, s=4, alpha=.35, color=c)
        ax.plot([0, 150], [0, 150], color="#8a8984", lw=1, ls="--")
        ax.set(title=name, xlabel="RUL aktual")
    axes[0].set_ylabel("RUL prediksi")
    savefig(fig, mdir / "11_prediksi_vs_aktual.png")

    # Grafik 12: kurva training model DL
    dl = [n for n in oof if n != "baseline"]
    if dl:
        fig, axes = plt.subplots(1, len(dl), figsize=(4 * len(dl), 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, dl):
            for f in fold_results(mdir / name):
                h = pd.DataFrame(json.loads(f.read_text())["history"])
                ax.plot(h.epoch, h.val_rmse, lw=1.2, label=f"val {f.stem}")
                ax.plot(h.epoch, h.train_rmse, lw=.8, ls=":", color="#8a8984")
            ax.set(title=f"{name} (titik-titik = train)", xlabel="epoch")
        axes[0].set_ylabel("RMSE")
        axes[0].legend(fontsize=7)
        savefig(fig, mdir / "12_kurva_training.png")
    return tab


def pick_best(tab) -> str:
    """Model utama WAJIB deep learning: pilih RMSE gabungan terendah di antara model DL saja.
    Baseline hanya pembanding."""
    dl = tab[tab.model != "baseline"]
    return (dl if len(dl) else tab).sort_values("rmse_gabungan").iloc[0]["model"]


def write_prediction(args, mdir, tab, log):
    best = pick_best(tab)
    preds = [pd.read_csv(f).set_index("id_unit").pred for f in sorted((mdir / best).glob("fold*_test.csv"))]
    ens = pd.concat(preds, axis=1)
    out = pd.DataFrame({"id_unit": ens.index, "rul": ens.mean(axis=1).clip(0, RUL_CAP).round(2).values})
    out.to_csv(mdir / PRED_FILE, index=False)                 # salinan per eksperimen
    path = args.pred_dir / PRED_FILE
    if not args.write_pred:
        log(f"[task2] --no-write_pred: prediksi hanya disimpan di {mdir / PRED_FILE}")
        return mdir / PRED_FILE
    out.to_csv(path, index=False)
    (args.pred_dir / "prediksi").mkdir(exist_ok=True)
    out.to_csv(args.pred_dir / "prediksi" / PRED_FILE, index=False)    # salinan di prediksi/
    ens.assign(mean=ens.mean(axis=1), std=ens.std(axis=1)).to_csv(mdir / f"test_prediksi_per_fold_{best}.csv")
    log(f"[task2] model terbaik: {best} (ensemble {len(preds)} fold) -> {path}")
    save_json({"model_terbaik": best, "n_fold_ensemble": len(preds)}, mdir / "pilihan_model.json")
    return path


def run(args):
    args = resolve_paths(args, "task2")
    set_seed(args.seed)
    device = get_device(args.device)
    # tiap panjang window punya folder hasil sendiri, supaya eksperimen W tidak saling menimpa
    mdir = args.task_out / ("models" if args.W == 50 else f"models_W{args.W}")
    mdir.mkdir(parents=True, exist_ok=True)
    logf = open(mdir / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = config(args)
    log(f"[task2] device={device} models={args.models} cfg={cfg} mc={args.mc}")
    if args.eda:                                   # 2a-c: grafik EDA & window (bagian dari run_all)
        from kode.task2 import eda, windows
        eda.run(copy.copy(args))
        windows.report(copy.copy(args), args.W)
    tr, te = load_raw(args)
    tr = add_rul_label(tr)
    modes = failure_modes(tr)
    folds = unit_folds(tr.id_unit.unique(), args.n_folds, seed=args.seed)
    n_run = 2 if args.debug and args.run_folds is None else (args.run_folds or args.n_folds)
    for k in range(n_run):
        log(f"[task2] fold {k + 1}/{n_run} (val {len(folds[k][1])} mesin)")
        run_fold(k, *folds[k], tr, te, args, cfg, device, mdir, log)
    tab = summarize_all(args, mdir, modes, log)
    path = write_prediction(args, mdir, tab, log)
    best = pick_best(tab)
    if best != "baseline":
        from kode.task2.uncertainty import report as unc_report
        unc_report(args, mdir, best, log)
    if args.write_pred:
        export(args, mdir, best, log)
    logf.close()
    return path


def export(args, mdir, best, log):
    """Salin hasil final ke results/task2/ (grafik & tabel laporan) dan models/ (bobot model).
    Checkpoint per epoch, log mentah, dan CSV per fold tetap di <out_dir> (tidak di-commit)."""
    dst = args.results_dir / "task2"
    for sub in ("eda", "windows"):
        if (args.task_out / sub).exists():
            shutil.copytree(args.task_out / sub, dst / sub, dirs_exist_ok=True)
    (dst / "models").mkdir(parents=True, exist_ok=True)
    for f in mdir.iterdir():
        if f.is_file() and f.suffix in (".png", ".csv", ".json", ".txt"):
            shutil.copy2(f, dst / "models" / f.name)
    for name in args.models:                       # ringkasan per fold (kecil) untuk tiap model
        for f in fold_results(mdir / name):
            (dst / "models" / name).mkdir(exist_ok=True)
            shutil.copy2(f, dst / "models" / name / f.name)
    if (mdir / "uncertainty").exists():
        shutil.copytree(mdir / "uncertainty", dst / "uncertainty", dirs_exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    weights = sorted((mdir / best).glob("fold*_best.pt"))
    for f in weights + sorted((mdir / best).glob("fold*_preprocessor.json")):
        shutil.copy2(f, args.models_dir / f"task2_{best}_{f.name}")
    log(f"[task2] hasil final -> {dst} ; bobot {best} ({len(weights)} fold) -> {args.models_dir}")


def build_parser():
    p = add_common_args(argparse.ArgumentParser(description="Task 2 - RUL"))
    p.add_argument("--models", nargs="+", default=["baseline", "lstm", "tcn", "transformer"])
    p.add_argument("--mc", type=int, default=50, help="jumlah forward pass MC dropout (2f)")
    p.add_argument("--service_h", type=int, default=20,
                   help="waktu persiapan servis (siklus): servis jika batas bawah interval <= H")
    p.add_argument("--eda", action=argparse.BooleanOptionalAction, default=True,
                   help="buat ulang grafik EDA & window (2a-c) sebelum training")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--run_folds", type=int, default=None, help="jalankan k fold pertama saja")
    p.add_argument("--W", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples_per_epoch", type=int, default=65536)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--write_pred", action=argparse.BooleanOptionalAction, default=True,
                   help="tulis prediksi_rul.csv final ke --pred_dir (matikan untuk eksperimen)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
