"""Task 3d - clustering: KMeans & GMM (klasik) vs deep clustering IDEC.

Semua metode memakai k yang sama (dari select_k.py) dan representasi fusion yang sama.
  KMeans (concat)   klasik penuh: early-fusion embedding + KMeans (baseline)
  KMeans (FusionAE) KMeans pada latent FusionAE
  GMM (FusionAE)    cluster elips dengan kovarians penuh (bentuk cluster lebih fleksibel dari KMeans)
  IDEC (FusionAE)   deep clustering (model utama)

IDEC = DEC (Xie et al. 2016) + rekonstruksi (Guo et al. 2017):
  1. Encoder = encoder FusionAE yang sudah dilatih (pretraining autoencoder, tahap 1 DEC).
  2. Pusat cluster μ diinisialisasi KMeans pada latent.
  3. Soft assignment Student-t: q_ij ∝ (1 + ||z_i - μ_j||²)^-1.
  4. Target p_ij ∝ q_ij² / Σ_i q_ij (mempertajam assignment yang sudah yakin; membagi dengan frekuensi
     cluster mencegah satu cluster besar menelan yang lain). p dihitung ulang tiap epoch.
  5. Loss = KL(P || Q) + γ · rekonstruksi (γ = 0,1). Encoder DAN pusat cluster ikut dilatih, sehingga
     representasi disesuaikan untuk clustering. Rekonstruksi (tambahan IDEC) menjaga struktur lokal agar
     latent tidak rusak/collapse.
  6. Berhenti jika < 0,1% label berubah antar-epoch (tol) atau mencapai max_epoch.

Setelah IDEC: refinement hierarkis (refine_by_block) memeriksa apakah ada cluster yang di dalamnya
masih berisi dua kelompok yang hanya terlihat di blok gambar. Hasilnya analisis ALTERNATIF
(cluster_alternatif_k*.csv); cluster final tetap k dari aturan 3e kecuali --refine.

Pembanding tanpa label (tidak ada ARI asli):
  silhouette di latent, stabilitas antar-seed (ARI 3 seed), kesepakatan antar-metode (ARI),
  Cramér's V terhadap mikroskop (rendah = tidak mengelompok menurut alat), ukuran cluster terkecil.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from kode.task3.fusion import block_mse, cramers_v


# ------------------------------------------------------------------ IDEC
def soft_assign(z, mu, alpha: float = 1.0):
    d2 = torch.cdist(z, mu) ** 2
    q = (1 + d2 / alpha) ** (-(alpha + 1) / 2)
    return q / q.sum(1, keepdim=True)


def target_dist(q):
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdim=True)


def idec(ae, X: np.ndarray, splits, k: int, device, seed: int, gamma=0.1, lr=1e-4, bs=256,
         max_epoch=100, tol=1e-3, log=print):
    from sklearn.cluster import KMeans
    torch.manual_seed(seed)
    ae = copy.deepcopy(ae).to(device)
    Xt = torch.from_numpy(X).to(device)
    with torch.no_grad():
        z0 = ae.enc(Xt).cpu().numpy()
    km = KMeans(k, n_init=20, random_state=seed).fit(z0)
    mu = nn.Parameter(torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device))
    opt = torch.optim.Adam(list(ae.parameters()) + [mu], lr=lr)
    prev = km.labels_
    hist = []
    steps = math.ceil(len(X) / bs)
    for ep in range(max_epoch):
        ae.eval()
        with torch.no_grad():
            q_all = soft_assign(ae.enc(Xt), mu)
            p_all = target_dist(q_all)
        lab = q_all.argmax(1).cpu().numpy()
        changed = float((lab != prev).mean())
        prev = lab
        if ep > 0 and changed < tol:
            hist.append({"epoch": ep, "label_berubah": changed, "kl": float("nan"), "rek": float("nan")})
            break
        ae.train()
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        kl_tot = rec_tot = 0.0
        for b in range(steps):
            j = perm[b * bs:(b + 1) * bs]
            z, xhat = ae(Xt[j])
            q = soft_assign(z, mu)
            kl = (p_all[j] * (torch.log(p_all[j] + 1e-10) - torch.log(q + 1e-10))).sum(1).mean()
            rec = block_mse(xhat, Xt[j], splits)
            loss = kl + gamma * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            kl_tot += kl.item() * len(j)
            rec_tot += rec.item() * len(j)
        hist.append({"epoch": ep + 1, "label_berubah": changed, "kl": kl_tot / len(X), "rek": rec_tot / len(X)})
    ae.eval()
    with torch.no_grad():
        z = ae.enc(Xt)
        q = soft_assign(z, mu).cpu().numpy()
    log(f"    IDEC seed {seed}: {len(hist)} epoch, label berubah terakhir {hist[-1]['label_berubah']:.4f}")
    state = {"ae": {k_: v.cpu() for k_, v in ae.state_dict().items()}, "mu": mu.detach().cpu()}
    return q.argmax(1), q, z.cpu().numpy(), hist, state


# ------------------------------------------------------------------ metode
def relabel_by_size(lab):
    """Nomor cluster diurutkan dari yang terbesar (0 = terbesar), supaya keluaran mudah dibaca."""
    order = np.argsort(-np.bincount(lab))
    m = np.empty_like(order)
    m[order] = np.arange(len(order))
    return m[lab]


def run_methods(reps: dict, X_ae, ae, splits, k, mic, device, seed, seeds=3, idec_epochs=100, log=print):
    """reps: {'concat': X, 'FusionAE': Z}. Return tabel perbandingan, label per metode, info IDEC."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    Z = reps["FusionAE"]
    runs = {
        "KMeans (concat)": lambda s: KMeans(k, n_init=10, random_state=s).fit_predict(reps["concat"]),
        "KMeans (FusionAE)": lambda s: KMeans(k, n_init=10, random_state=s).fit_predict(Z),
        "GMM (FusionAE)": lambda s: GaussianMixture(k, covariance_type="full", n_init=2, reg_covar=1e-4,
                                                    random_state=s).fit(Z).predict(Z),
    }
    labels, all_runs, idec_out = {}, {}, {}
    for name, f in runs.items():
        all_runs[name] = [f(seed + i) for i in range(seeds)]
    res = [idec(ae, X_ae, splits, k, device, seed + i, max_epoch=idec_epochs, log=log) for i in range(seeds)]
    all_runs["IDEC (FusionAE)"] = [r[0] for r in res]
    idec_out = {"q": res[0][1], "z": res[0][2], "history": res[0][3], "state": res[0][4]}

    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(len(Z), min(10000, len(Z)), replace=False)
    rows = []
    for name, labs in all_runs.items():
        lab = relabel_by_size(labs[0])
        labels[name] = lab
        space = idec_out["z"] if name.startswith("IDEC") else (reps["concat"] if "concat" in name else Z)
        stab = np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(seeds) for j in range(i + 1, seeds)])
        rows.append({"metode": name, "jenis": "deep" if name.startswith("IDEC") else "klasik",
                     "silhouette_ruang_sendiri": silhouette_score(space[sil_idx], lab[sil_idx]),
                     "silhouette_latent_FusionAE": silhouette_score(Z[sil_idx], lab[sil_idx]),
                     "stabilitas_antar_seed_ARI": stab,
                     "cramers_v_mikroskop": cramers_v(lab, mic),
                     "cluster_terkecil_pct": 100 * np.bincount(lab).min() / len(lab),
                     "cluster_terbesar_pct": 100 * np.bincount(lab).max() / len(lab)})
    df = pd.DataFrame(rows)
    names = list(labels)
    agree = pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names],
                         index=names, columns=names)
    log(df.round(3).to_string(index=False))
    return df, labels, agree, idec_out


def pick_final(df: pd.DataFrame, min_pct: float = 0.5) -> str:
    """Model utama harus deep learning -> IDEC, kecuali hasilnya degeneratif (cluster < min_pct%
    data, tanda collapse); maka pakai metode klasik terbaik pada representasi deep (FusionAE)."""
    r = df.set_index("metode")
    if r.loc["IDEC (FusionAE)", "cluster_terkecil_pct"] >= min_pct:
        return "IDEC (FusionAE)"
    alt = r.drop(index=["IDEC (FusionAE)", "KMeans (concat)"])
    return str(alt.silhouette_latent_FusionAE.idxmax())


# ------------------------------------------------------------------ penyempurnaan hierarkis
PURITY_MIN = 0.99   # cluster baru harus >= 99% berasal dari SATU cluster lama (pemecahan murni)


def refine_by_block(final: np.ndarray, B: np.ndarray, mic: np.ndarray, seed: int, max_steps: int = 3,
                    log=print):
    """Pecah cluster final yang di dalamnya masih ada dua kelompok yang hanya terlihat di satu modalitas.

    Alasan: k dipilih di ruang gabungan. Dua tekstur dengan spektrum yang SAMA hanya berbeda di blok
    gambar, sehingga di ruang gabungan perbedaannya tertutup variasi spektrum di dalam cluster, dan
    kriteria global (silhouette, DB, CH) cenderung memilih partisi yang lebih kasar.
    Aturan (diulang sampai gagal):
      1. KMeans(k+1) pada blok B (blok gambar setelah koreksi mikroskop).
      2. Diterima hanya jika hasilnya PEMECAHAN MURNI: tepat satu cluster lama terbagi dua dan
         semua cluster baru >= PURITY_MIN berasal dari satu cluster lama (cluster lain tidak terganggu).
      3. Dan pemecahan itu tidak mengikuti mikroskop: Cramér's V(sub-cluster, mikroskop) < MIC_V_MAX.
    """
    from sklearn.cluster import KMeans
    from kode.task3.fusion import MIC_V_MAX
    lab = final.copy()
    steps = []
    for _ in range(max_steps):
        k = int(lab.max()) + 1
        new = KMeans(k + 1, n_init=10, random_state=seed).fit_predict(B)
        ct = pd.crosstab(lab, new)
        parent = ct.idxmax(0)
        purity = float((ct.max(0) / ct.sum(0)).min())
        cnt = parent.value_counts()
        rec = {"k_dari": k, "k_ke": k + 1, "kemurnian_min": purity, "cluster_dipecah": None,
               "ukuran_sub": None, "cramers_v_mikroskop": None, "pct_M2_sub": None, "diterima": False}
        if purity >= PURITY_MIN and len(cnt) == k and (cnt == 2).sum() == 1:
            c = int(cnt.idxmax())
            kids = parent[parent == c].index.tolist()
            s = (lab == c) & np.isin(new, kids)
            v = cramers_v(new[s], mic[s])
            rec.update(cluster_dipecah=c, ukuran_sub=[int((new[s] == j).sum()) for j in kids],
                       cramers_v_mikroskop=v,
                       pct_M2_sub=[float(100 * np.mean(mic[s][new[s] == j] == "M2")) for j in kids],
                       diterima=bool(v < MIC_V_MAX))
        steps.append(rec)
        log(f"    refinement k {k}->{k + 1}: kemurnian {purity:.3f}, cluster dipecah {rec['cluster_dipecah']}, "
            f"V mikroskop {rec['cramers_v_mikroskop']}, diterima {rec['diterima']}")
        if not rec["diterima"]:
            break
        lab = np.where((lab == c) & (new == kids[1]), k, lab)
    return relabel_by_size(lab), steps


def plot_methods(df, agree, idec_hist, path):
    from kode.common.plot import SERIES, TEXT2, plt, savefig
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), gridspec_kw={"wspace": .55, "width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    y = np.arange(len(df))
    w = .27
    for j, (col, lab) in enumerate([("silhouette_latent_FusionAE", "silhouette (latent FusionAE)"),
                                    ("stabilitas_antar_seed_ARI", "stabilitas antar-seed"),
                                    ("cramers_v_mikroskop", "Cramér's V mikroskop (↓)")]):
        ax.barh(y + (j - 1) * w, df[col], w * .95, color=SERIES[j], label=lab)
    ax.set_yticks(y, df.metode)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2)
    ax.set(xlim=(0, 1.05), title="Perbandingan metode (tanpa label)")
    ax = axes[1]
    ax.imshow(agree.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    short = [n.replace(" (FusionAE)", "\n(AE)").replace(" (concat)", "\n(concat)") for n in agree.index]
    ax.set_xticks(range(len(short)), short, fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    for i in range(len(short)):
        for j in range(len(short)):
            v = agree.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > .6 else TEXT2)
    ax.grid(False)
    ax.set_title("Kesepakatan antar-metode (ARI)")
    ax = axes[2]
    h = pd.DataFrame(idec_hist)          # baris terakhir (cek konvergensi) tidak punya KL -> celah
    ax.plot(h.epoch, h.kl, "o-", ms=3, color=SERIES[0], label="KL(P||Q)")
    ax2 = ax.twinx()
    ax2.plot(h.epoch, 100 * h.label_berubah, "s--", ms=3, color=SERIES[1], label="% label berubah")
    ax.set(xlabel="epoch", ylabel="KL", title="Training IDEC")
    ax2.set_ylabel("% label berubah")
    ax2.grid(False)
    fig.legend(*[sum(x, []) for x in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())],
               loc="upper right", fontsize=7, bbox_to_anchor=(0.99, 0.98))
    savefig(fig, path)
