"""Task 3b - representation learning tanpa label: SimCLR terpisah untuk gambar dan spektrum.

  python -m kode.task3.ssl [--debug] [--data_dir ...] [--out_dir ...] [--max_minutes 80]

Alur:
1. Gambar uint8 dan spektrum (despike -> baseline -> L2, lihat data.py) dimuat SEKALI ke GPU.
2. Per modalitas: SimCLR = dua view teraugmentasi per sampel (augment.py) -> encoder -> head -> NT-Xent.
   Augmentasi di GPU per batch, sehingga CPU Colab/Kaggle (2 core) tidak jadi bottleneck.
3. Embedding = output encoder (bukan head). Gambar: rata-rata 8 transformasi dihedral (rotasi 90°
   x flip) sebagai test-time augmentation, supaya embedding akhir lebih tahan rotasi.
4. Diagnostik tanpa label, dibandingkan dengan baseline klasik (gambar: log PSD radial 32 dim, fitur
   tekstur invarian rotasi & translasi; spektrum: PCA 32 dim dari spektrum terproses). PCA piksel
   TIDAK dipakai sebagai pembanding: bergantung fase/posisi tekstur sehingga hampir tidak informatif
   (retrieval ~1%), pembanding yang terlalu lemah.
   - AUC menebak mikroskop dari embedding (makin dekat 0,5 makin baik: EDA menunjukkan 0,88).
   - retrieval view-ke-view: dua view teraugmentasi independen dari 2.000 sampel; tetangga terdekat
     view A di antara view B = sampel yang sama? (invarian terhadap augmentasi DAN tetap membedakan
     sampel). View-ke-view agar kedua metode membandingkan input berukuran sama (crop 36x36).
   - kesepakatan lintas-modalitas: AMI antara KMeans(embedding gambar) dan KMeans(embedding spektrum).
     Kedua modalitas mengukur material yang sama, jadi representasi yang baik saling sepakat.
   - rank efektif: deteksi collapse (semua embedding sama).

Resume: checkpoint tiap epoch (model, optimizer, scheduler, scaler, RNG). Urutan batch dan
augmentasi diturunkan dari (seed, epoch), jadi hasil resume sama dengan tanpa putus.
--max_minutes: berhenti rapi setelah epoch yang melewati batas waktu (sesi Colab/Kaggle terbatas);
jalankan ulang perintah yang sama untuk melanjutkan.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time

import numpy as np
import torch

from kode.common import add_common_args, get_device, resolve_paths, set_seed
from kode.common.utils import load_checkpoint, save_checkpoint, save_json
from kode.task3.augment import augment_images, augment_spectra, img_to_input, spec_to_input
from kode.task3.data import load, preprocess_spectrum
from kode.task3.encoders import build, n_params, nt_xent

MODALITIES = ("gambar", "spektrum")


def config(args, modality: str) -> dict:
    cfg = dict(epochs=args.epochs_img if modality == "gambar" else args.epochs_spec, bs=args.bs,
               lr=args.lr, wd=1e-4, tau=args.tau, warmup=0.05)
    if args.debug:
        cfg.update(epochs=2, bs=128)
    return cfg


class TimeUp(Exception):
    pass


def _augment(modality, xb, gen):
    return augment_images(xb, gen) if modality == "gambar" else augment_spectra(xb, gen)


def train_ssl(modality, data, cfg, ckpt_path, device, seed, log=print, deadline=None):
    """data: tensor di device (gambar uint8 (N,H,W) / spektrum float (N,L))."""
    torch.manual_seed(seed)
    model = build(modality).to(device)
    is_img = modality == "gambar"
    if is_img and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    N = len(data)
    steps = N // cfg["bs"]
    total = cfg["epochs"] * steps
    warm = max(1, int(cfg["warmup"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = {"epoch": 0, "history": [], "train_sec": 0.0, "done": False}
    ck = load_checkpoint(ckpt_path, map_location=device) if cfg.get("resume", True) else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        torch.set_rng_state(ck["rng_cpu"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        log(f"    resume {modality} dari epoch {state['epoch']}")

    while not state["done"]:
        ep = state["epoch"]
        t0 = time.time()
        model.train()
        g_perm = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N, generator=g_perm)[: steps * cfg["bs"]].view(steps, cfg["bs"]).to(device)
        gen = torch.Generator(device=device).manual_seed(seed * 1000 + ep)
        tot = 0.0
        for b in range(steps):
            xb = data[perm[b]]
            with torch.no_grad():
                v = torch.cat([_augment(modality, xb, gen), _augment(modality, xb, gen)])
            if is_img and device.type == "cuda":
                v = v.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                z1, z2 = model(v).chunk(2)
            loss = nt_xent(z1, z2, cfg["tau"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        sec = time.time() - t0
        state["train_sec"] += sec
        state["history"].append({"epoch": ep + 1, "loss": tot / steps, "detik": sec})
        state["epoch"] = ep + 1
        state["done"] = state["epoch"] >= cfg["epochs"]
        save_checkpoint(ckpt_path, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), state=state, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        eta = (cfg["epochs"] - state["epoch"]) * state["train_sec"] / max(1, len(state["history"]))
        log(f"    {modality} ep {state['epoch']:3d}/{cfg['epochs']} loss {tot / steps:.4f} "
            f"({sec:.1f}s, sisa ~{eta / 60:.1f} mnt)")
        if deadline is not None and time.time() > deadline and not state["done"]:
            raise TimeUp(f"batas waktu tercapai di {modality} epoch {state['epoch']}; jalankan ulang untuk lanjut")
    model.eval()
    return model, state


@torch.no_grad()
def encode(model, x: torch.Tensor, modality: str, bs: int = 1024, tta: bool = True) -> np.ndarray:
    """x = input yang sudah siap (gambar (N,1,H,W) terstandardisasi / spektrum (N,1,L)).
    Gambar + tta: rata-rata 8 transformasi dihedral (4 rotasi 90° x flip)."""
    model.eval()
    enc = model.encoder
    out = []
    for k in range(0, len(x), bs):
        xb = x[k:k + bs]
        if modality == "gambar" and tta:
            hs = [enc(torch.rot90(xb, r, (2, 3))) for r in range(4)]
            hs += [enc(torch.rot90(xb.flip(3), r, (2, 3))) for r in range(4)]
            h = torch.stack(hs).mean(0)
        else:
            h = enc(xb)
        out.append(h.float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed(model, data, modality, bs: int = 1024, tta: bool = True) -> np.ndarray:
    """Embedding encoder (bukan head) untuk data asli tanpa augmentasi (gambar penuh 64x64)."""
    out = []
    for k in range(0, len(data), bs):
        xb = data[k:k + bs]
        x = img_to_input(xb) if modality == "gambar" else spec_to_input(xb)
        out.append(encode(model, x, modality, bs, tta))
    return np.concatenate(out)


# ------------------------------------------------------------------ diagnostik
def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def effective_rank(x: np.ndarray) -> float:
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def retrieval_top1(orig: np.ndarray, view: np.ndarray) -> float:
    o, v = _l2(orig), _l2(view)
    return float(((v @ o.T).argmax(1) == np.arange(len(o))).mean())


def cross_modal_ami(a: np.ndarray, b: np.ndarray, ks, seed) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score
    red = lambda x: PCA(min(32, x.shape[1]), random_state=seed).fit_transform(_l2(x))  # noqa: E731
    a, b = red(a), red(b)
    res = {}
    for k in ks:
        la = KMeans(k, n_init=3, random_state=seed).fit_predict(a)
        lb = KMeans(k, n_init=3, random_state=seed).fit_predict(b)
        res[k] = float(adjusted_mutual_info_score(la, lb))
    return res


def diagnostics(img, spec_pp, mic, emb, models, device, seed, n_ret=2000, log=print) -> dict:
    """Bandingkan embedding SSL dengan baseline klasik pada metrik tanpa label."""
    from sklearn.decomposition import PCA
    from kode.task3.eda import image_stats, mic_auc
    rng = np.random.default_rng(seed)
    y_mic = (mic == "M2").astype(int)
    fit_idx = rng.choice(len(img), min(len(img), 10000), replace=False)
    pca_spec = PCA(32, random_state=seed).fit(spec_pp[fit_idx])
    psd_feat = lambda x: np.log(image_stats(x)[1][:, 1:] + 1e-6)   # noqa: E731  (N, 32)
    base = {"gambar": psd_feat(img), "spektrum": pca_spec.transform(spec_pp)}
    klasik = {"gambar": "PSD radial", "spektrum": "PCA"}

    ret_idx = np.sort(rng.choice(len(img), min(len(img), n_ret), replace=False))
    res = {"auc_mikroskop": {}, "retrieval_top1": {}, "rank_efektif": {}, "dim": {}}
    for m in MODALITIES:
        src = torch.from_numpy((img if m == "gambar" else spec_pp)[ret_idx]).to(device)
        aug = augment_images if m == "gambar" else augment_spectra
        # dua view independen, dibuat SEKALI dan dipakai bersama oleh metode klasik dan SimCLR
        va, vb = (aug(src, torch.Generator(device=device).manual_seed(seed + s_)) for s_ in (11, 12))
        if m == "gambar":
            fa, fb = (psd_feat(v[:, 0].cpu().numpy()) for v in (va, vb))
        else:
            fa, fb = (pca_spec.transform((v[:, 0] / math.sqrt(v.shape[-1])).cpu().numpy()) for v in (va, vb))
        ea, eb = encode(models[m], va, m), encode(models[m], vb, m)
        for name, rep, (qa, qb) in [(f"{m}: {klasik[m]}", base[m], (fa, fb)), (f"{m}: SimCLR", emb[m], (ea, eb))]:
            res["auc_mikroskop"][name] = mic_auc(rep, y_mic, seed)
            res["retrieval_top1"][name] = retrieval_top1(qb, qa)
            res["rank_efektif"][name] = effective_rank(rep[ret_idx])
            res["dim"][name] = int(rep.shape[1])
    ks = (3, 6, 10)
    res["ami_lintas_modalitas"] = {"klasik": cross_modal_ami(base["gambar"], base["spektrum"], ks, seed),
                                   "SimCLR": cross_modal_ami(emb["gambar"], emb["spektrum"], ks, seed)}
    for k, v in res.items():
        log(f"  {k}: " + json.dumps(v, default=lambda x: round(x, 3)))
    return res


# ------------------------------------------------------------------ grafik
def plot_augmentations(img, spec_pp, device, seed, path):
    from kode.common.plot import SERIES, TEXT, plt, savefig
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(img), 3, replace=False)
    gen = torch.Generator(device=device).manual_seed(seed)
    xi = torch.from_numpy(img[idx]).to(device)
    xs = torch.from_numpy(spec_pp[idx]).to(device)
    views = [augment_images(xi, gen).cpu().numpy() for _ in range(4)]
    sviews = [augment_spectra(xs, gen).cpu().numpy() for _ in range(3)]
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(3, 7, width_ratios=[1] * 5 + [.12, 4.2], wspace=.08, hspace=.25)
    for r in range(3):
        panels = [img_to_input(xi[r:r + 1])[0, 0].cpu().numpy()] + [v[r, 0] for v in views]
        for j, im in enumerate(panels):
            a = fig.add_subplot(gs[r, j])
            a.imshow(im, cmap="gray")
            a.set_xticks([]), a.set_yticks([]), a.grid(False)
            if r == 0:
                a.set_title("asli" if j == 0 else f"view {j}", fontsize=8)
        ax = fig.add_subplot(gs[r, 6])
        ax.plot(spec_to_input(xs[r:r + 1])[0, 0].cpu(), color=TEXT, lw=1.6, label="asli")
        for j, v in enumerate(sviews):
            ax.plot(v[r, 0], lw=.9, color=SERIES[j], alpha=.85, label=f"view {j + 1}")
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("kanal")
        if r == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper right")
            ax.set_title("spektrum: shift ±7, stretch ±3%, sisa baseline, noise, spike", fontsize=8)
    fig.suptitle("Augmentasi SimCLR - gambar: crop 36x36 diputar 0-360° + flip, skala ±15%, gamma, blur, noise", fontsize=9)
    savefig(fig, path)


def plot_history(hist: dict, path):
    from kode.common.plot import SERIES, plt, savefig
    fig, axes = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 2.8))
    axes = np.atleast_1d(axes)
    for ax, (m, h), c in zip(axes, hist.items(), SERIES):
        ax.plot([r["epoch"] for r in h], [r["loss"] for r in h], color=c)
        ax.set(xlabel="epoch", ylabel="NT-Xent", title=f"SimCLR {m}")
    savefig(fig, path)


def plot_diagnostics(res: dict, path):
    from kode.common.plot import MUTED, SERIES, TEXT2, plt, savefig
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.0), gridspec_kw={"wspace": .45})
    names = list(res["auc_mikroskop"])
    col = [SERIES[1] if "SimCLR" in n else SERIES[0] for n in names]
    for ax, key, title, ref in [(axes[0], "auc_mikroskop", "AUC tebak mikroskop (↓, ideal 0,5)", .5),
                                (axes[1], "retrieval_top1", "Retrieval view-ke-view (↑)", None)]:
        vals = [res[key][n] for n in names]
        ax.barh(names, vals, color=col)
        for j, v in enumerate(vals):
            ax.text(v + .01, j, f"{v:.2f}", va="center", fontsize=7, color=TEXT2)
        if ref is not None:
            ax.axvline(ref, color=MUTED, ls="--", lw=1)
        ax.set(xlim=(0, 1.1), title=title)
        ax.invert_yaxis()
    axes[1].set_yticklabels([])
    ks = list(res["ami_lintas_modalitas"]["klasik"])
    w = .38
    for j, (rep, c) in enumerate([("klasik", SERIES[0]), ("SimCLR", SERIES[1])]):
        v = [res["ami_lintas_modalitas"][rep][k] for k in ks]
        axes[2].bar(np.arange(len(ks)) + (j - .5) * w, v, w * .95, color=c, label=rep)
    axes[2].set_xticks(np.arange(len(ks)), [f"k={k}" for k in ks])
    axes[2].set(ylabel="AMI", title="Kesepakatan cluster gambar vs spektrum (↑)")
    axes[2].legend()
    savefig(fig, path)


# ------------------------------------------------------------------ main
def run(args):
    args = resolve_paths(args, "task3")
    set_seed(args.seed)
    device = get_device(args.device)
    sdir = args.task_out / "ssl"
    sdir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 60 * args.max_minutes if args.max_minutes else None
    logf = open(sdir / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    img, spec, meta = load(args)
    spec_pp = preprocess_spectrum(spec)
    log(f"[task3-ssl] device={device} n={len(img)} max_minutes={args.max_minutes}")
    data = {"gambar": torch.from_numpy(img).to(device), "spektrum": torch.from_numpy(spec_pp).to(device)}
    models, emb, hist = {}, {}, {}
    try:
        for m in MODALITIES:
            cfg = config(args, m)
            cfg["resume"] = args.resume
            log(f"[task3-ssl] {m}: {cfg}")
            model, st = train_ssl(m, data[m], cfg, args.ckpt_dir / f"ssl_{m}.pt", device, args.seed, log, deadline)
            models[m], hist[m] = model, st["history"]
            emb_path = sdir / f"emb_{m}.npy"
            emb[m] = embed(model, data[m], m)
            np.save(emb_path, emb[m].astype(np.float16))
            torch.save(model.state_dict(), sdir / f"simclr_{m}.pt")
            save_json({"modalitas": m, "n_params_encoder": n_params(model.encoder),
                       "n_params_total": n_params(model), "dim_embedding": int(emb[m].shape[1]),
                       "epoch": st["epoch"], "detik_training": st["train_sec"], "cfg": cfg},
                      sdir / f"info_{m}.json")
            log(f"[task3-ssl] {m} selesai: {st['epoch']} epoch, {st['train_sec'] / 60:.1f} menit, "
                f"embedding {emb[m].shape}")
    except TimeUp as e:
        log(f"[task3-ssl] {e}")
        logf.close()
        raise

    plot_history(hist, sdir / "11_kurva_loss.png")
    plot_augmentations(img, spec_pp, device, args.seed, sdir / "12_contoh_augmentasi.png")
    res = diagnostics(img, spec_pp, meta.mikroskop.to_numpy(), emb, models, device, args.seed,
                      n_ret=500 if args.debug else 2000, log=log)
    save_json(res, sdir / "diagnostik_representasi.json")
    plot_diagnostics(res, sdir / "13_diagnostik_representasi.png")
    logf.close()
    if not args.debug:
        dst = args.results_dir / "task3" / "ssl"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sdir.iterdir():
            if f.suffix in (".png", ".json", ".txt"):
                shutil.copy2(f, dst / f.name)
        args.models_dir.mkdir(parents=True, exist_ok=True)
        for m in MODALITIES:
            shutil.copy2(sdir / f"simclr_{m}.pt", args.models_dir / f"task3_simclr_{m}.pt")
        print(f"[task3-ssl] hasil -> {dst}; bobot -> {args.models_dir}")
    return emb


def add_ssl_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--epochs_img", type=int, default=150)
    p.add_argument("--epochs_spec", type=int, default=100)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.2, help="temperatur NT-Xent")
    p.add_argument("--max_minutes", type=float, default=None,
                   help="berhenti rapi setelah N menit (lanjutkan dengan menjalankan ulang)")
    return p


if __name__ == "__main__":
    try:
        run(add_ssl_args(add_common_args(argparse.ArgumentParser(description="Task 3b - SimCLR"))).parse_args())
    except TimeUp:
        sys.exit(3)          # kode keluar khusus: belum selesai, jalankan ulang untuk melanjutkan
