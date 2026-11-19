"""Task 3 - Clustering multimodal (gambar mikrostruktur + spektrum), 36.000 sampel tanpa label.

  a. eda.py        - distorsi gambar (rotasi, skala, brightness, M1 vs M2) & spektrum (shift, baseline, spike)
  b. ssl.py        - SimCLR per modalitas (augmentasi dari EDA) -> embedding
  c. fusion.py     - blok per modalitas, koreksi mikroskop, FusionAE (deep) vs concat; ablasi satu modalitas
  d. cluster.py    - KMeans/GMM (klasik) vs IDEC (deep clustering, model utama)
  e. select_k.py   - k dari silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap (+ BIC)
  f. viz.py        - UMAP/t-SNE, profil cluster, pasangan cluster yang sulit dipisahkan
Output: <pred_dir>/cluster.csv dan <pred_dir>/prediksi/cluster.csv  (id_sampel,cluster ; integer)

Embedding 3b sudah ada (mis. dari Kaggle, diekstrak ke <out_dir>/task3/ssl)? Jalankan c-f saja:
  python -m kode.task3.main --no-eda --no-ssl
"""
import argparse
import copy
import shutil

import numpy as np
import pandas as pd
import torch

from kode.common import add_common_args, get_device, resolve_paths, set_seed
from kode.common.utils import save_json

PRED_FILE = "cluster.csv"


def load_embeddings(sdir, n):
    emb = {}
    for m in ("gambar", "spektrum"):
        f = sdir / f"emb_{m}.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} tidak ada: jalankan 3b (tanpa --no-ssl) atau ekstrak "
                                    f"embedding_task3.zip ke {sdir}")
        emb[m] = np.load(f).astype(np.float32)
        assert len(emb[m]) == n, f"{f}: {len(emb[m])} baris, data {n} (embedding debug vs penuh tertukar?)"
    return emb


def run_clustering(args, log=print):
    from kode.task3 import cluster as C, fusion as Fu, select_k as K, viz as V
    from kode.task3.data import load, preprocess_spectrum
    device = get_device(args.device)
    cdir = args.task_out / "clustering"
    cdir.mkdir(parents=True, exist_ok=True)
    img, spec, meta = load(args)
    mic = meta.mikroskop.to_numpy()
    emb = load_embeddings(args.task_out / "ssl", len(meta))
    dbg = args.debug
    summary = {"n": len(meta), "device": str(device)}

    # ---- c. fusion
    log("[task3-c] fusion")
    blocks, binfo = Fu.build_blocks(emb, mic, args.seed, log)
    X = np.concatenate([blocks["gambar"], blocks["spektrum"]], 1).astype(np.float32)
    splits = (blocks["gambar"].shape[1], blocks["spektrum"].shape[1])
    ae, ainfo = Fu.train_fusion_ae(X, splits, device, args.seed, epochs=10 if dbg else args.ae_epochs, log=log)
    Z = Fu.encode(ae, X, device)
    summary["fusion"] = {"blok": binfo, "fusion_ae": {k: v for k, v in ainfo.items() if k != "loss"}}

    # ---- e. pilih k (dibutuhkan sebelum d)
    log("[task3-e] pemilihan k")
    ks = range(2, (8 if dbg else args.k_max) + 1)
    kdf = K.scan(Z, ks, args.seed, n_sil=2000 if dbg else 10000, log=log)
    k_auto, kdf = K.choose(kdf)
    k = args.k or k_auto
    kdf.to_csv(cdir / "tabel_pemilihan_k.csv", index=False)
    K.plot(kdf, k, cdir / "14_pemilihan_k.png")
    summary["k"] = {"otomatis": k_auto, "dipakai": k, "override": args.k is not None}
    log(f"  k otomatis = {k_auto}; dipakai k = {k}")

    # ---- d. clustering
    log("[task3-d] clustering")
    reps = {"concat": X, "FusionAE": Z}
    mdf, labels, agree, idec_out = C.run_methods(reps, X, ae, splits, k, mic, device, args.seed,
                                                 seeds=2 if dbg else args.cluster_seeds,
                                                 idec_epochs=5 if dbg else args.idec_epochs, log=log)
    final_name = C.pick_final(mdf)
    final = labels[final_name]
    mdf["final"] = mdf.metode == final_name
    mdf.to_csv(cdir / "tabel_metode.csv", index=False)
    agree.to_csv(cdir / "tabel_kesepakatan_metode.csv")
    C.plot_methods(mdf, agree, idec_out["history"], cdir / "15_perbandingan_metode.png")
    summary["metode"] = mdf.to_dict("records")
    summary["metode_final"] = final_name
    summary["ukuran_cluster"] = np.bincount(final).tolist()
    log(f"  metode final: {final_name}; ukuran cluster {summary['ukuran_cluster']}")

    # ---- c. ablasi satu modalitas (butuh label final)
    log("[task3-c] ablasi modalitas")
    abl, single = Fu.ablation({"gambar saja": blocks["gambar"], "spektrum saja": blocks["spektrum"],
                               "concat": X, "FusionAE": Z}, final, mic, k, args.seed, log)
    abl.to_csv(cdir / "tabel_ablasi_modalitas.csv", index=False)
    summary["ablasi"] = abl.to_dict("records")
    single2 = {"gambar saja": single["gambar saja"], "spektrum saja": single["spektrum saja"]}
    V.plot_ablation(abl, final, single2, cdir / "16_ablasi_modalitas.png")

    # ---- f. visualisasi & cluster sulit
    log("[task3-f] visualisasi")
    space = idec_out["z"] if final_name.startswith("IDEC") else Z
    proj = V.embed_2d(space, args.seed, n_tsne=1500 if dbg else 8000, log=log)
    V.plot_2d(proj, final, mic, single2, cdir / "17_umap_tsne.png")
    spec_pp = preprocess_spectrum(spec)
    V.plot_profiles(img, spec_pp, final, mic, cdir / "18_profil_cluster.png", seed=args.seed)
    M = V.knn_overlap(space, final, seed=args.seed)
    pairs = V.hard_pairs(M, idec_out["q"] if final_name.startswith("IDEC") else None, final)
    pairs.to_csv(cdir / "tabel_pasangan_sulit.csv", index=False)
    M.to_csv(cdir / "tabel_overlap_knn.csv")
    V.plot_hard(M, pairs, img, spec_pp, final, cdir / "19_cluster_sulit.png", seed=args.seed)
    summary["pasangan_sulit"] = pairs.to_dict("records")
    log(pairs.round(2).to_string(index=False))

    # ---- simpan
    out = pd.DataFrame({"id_sampel": meta.id_sampel, "cluster": final.astype(int)})
    for d in (args.pred_dir, args.pred_dir / "prediksi"):
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / PRED_FILE, index=False)
    np.save(cdir / "latent_final.npy", space.astype(np.float16))
    torch.save({"fusion_ae": ae.state_dict(), "idec": idec_out["state"], "splits": splits, "k": k},
               cdir / "fusion_idec.pt")
    save_json(summary, cdir / "ringkasan_clustering.json")
    if not dbg:
        dst = args.results_dir / "task3" / "clustering"
        dst.mkdir(parents=True, exist_ok=True)
        for f in cdir.iterdir():
            if f.suffix in (".png", ".csv", ".json"):
                shutil.copy2(f, dst / f.name)
        args.models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cdir / "fusion_idec.pt", args.models_dir / "task3_fusion_idec.pt")
    log(f"[task3] cluster.csv -> {args.pred_dir / PRED_FILE} (k={k}, {final_name})")
    return out


def run(args):
    args = resolve_paths(args, "task3")
    set_seed(args.seed)
    if args.eda:
        from kode.task3 import eda
        eda.run(copy.copy(args))
    if args.ssl:
        from kode.task3 import ssl
        try:
            ssl.run(copy.copy(args))
        except ssl.TimeUp:
            print("[task3] SSL berhenti karena --max_minutes; jalankan ulang perintah yang sama untuk melanjutkan")
            return None
    return run_clustering(args)


def build_parser():
    p = add_common_args(argparse.ArgumentParser(description="Task 3 - Clustering"))
    p.add_argument("--eda", action=argparse.BooleanOptionalAction, default=True, help="buat ulang grafik EDA (3a)")
    p.add_argument("--ssl", action=argparse.BooleanOptionalAction, default=True,
                   help="jalankan/lanjutkan SimCLR (3b); --no-ssl memakai embedding yang sudah ada")
    p.add_argument("--k", type=int, default=None, help="paksa jumlah cluster (default: otomatis, 3e)")
    p.add_argument("--k_max", type=int, default=15)
    p.add_argument("--ae_epochs", type=int, default=60)
    p.add_argument("--idec_epochs", type=int, default=100)
    p.add_argument("--cluster_seeds", type=int, default=3)
    from kode.task3.ssl import add_ssl_args
    return add_ssl_args(p)


if __name__ == "__main__":
    run(build_parser().parse_args())
