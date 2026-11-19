"""Task 3c - fusion embedding gambar + spektrum, dan ablasi satu modalitas.

Input: embedding SimCLR (3b) per modalitas, emb_gambar.npy (N,256) dan emb_spektrum.npy (N,128).

1. prep_block (per modalitas, klasik dan transparan):
   L2-normalisasi baris (NT-Xent memakai cosinus, jadi panjang vektor tidak bermakna) -> PCA ke
   dimensi yang menjelaskan 95% varians (8-64) -> skala blok agar TOTAL varians = 1.
   Dengan begitu kedua modalitas berbobot sama, walaupun dimensinya berbeda (256 vs 128).

2. Koreksi mikroskop (khusus gambar, hanya jika lolos uji):
   EDA 3a: spektrum tidak dipengaruhi mikroskop (AUC 0,50), sedangkan gambar sangat dipengaruhi
   (AUC 0,88). Jika komposisi material TIDAK bergantung pada mikroskop, beda rata-rata embedding gambar
   M1 vs M2 murni artefak alat dan aman dihapus (centering per mikroskop). Uji: cluster spektrum
   (k = 3..8) vs mikroskop, Cramér's V < 0,10 (Cohen: dapat diabaikan) -> sebaran material sama di kedua mikroskop -> koreksi
   dipakai. Jika gagal, koreksi TIDAK dipakai (bisa menghapus sinyal material).

3. Dua representasi fusion yang dibandingkan:
   - concat (klasik, early fusion): [blok gambar, blok spektrum].
   - FusionAE (deep): autoencoder MLP pada concat dengan MODALITY DROPOUT. Saat training, satu blok
     kadang dinolkan (p = 0,25 masing-masing) tetapi KEDUANYA harus direkonstruksi. Akibatnya latent
     dipaksa memakai informasi yang dibagi kedua modalitas (memprediksi spektrum dari gambar dan
     sebaliknya) dan tetap menyimpan informasi khusus tiap modalitas. Latent 16 dim = representasi
     fusion, sekaligus pretraining untuk DEC/IDEC (cluster.py).
   Kenapa bukan CLIP/contrastive lintas-modal? Contrastive hanya menyimpan informasi yang SAMA di
   kedua modalitas. Jika dua material berkomposisi sama tetapi teksturnya beda, spektrum tidak bisa
   membedakannya, dan informasi tekstur itu akan dibuang. Autoencoder merekonstruksi keduanya.

4. Ablasi (setelah clustering final): cluster tiap modalitas sendiri dengan metode & k yang sama,
   lalu ukur apa yang hilang:
   - AMI terhadap cluster final;
   - pasangan cluster final yang TIDAK bisa dibedakan oleh satu modalitas (akurasi kNN a-vs-b < 0,75):
     inilah yang hilang jika hanya memakai modalitas itu;
   - akurasi kNN memprediksi cluster final dari embedding satu modalitas (berapa info yang hanya
     ada di modalitas lain);
   - Cramér's V cluster vs mikroskop (gambar saja cenderung mengelompok menurut mikroskop).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

MODALITIES = ("gambar", "spektrum")
MIC_V_MAX = 0.10    # Cramér's V < 0,1 = asosiasi dapat diabaikan (Cohen 1988)


# ------------------------------------------------------------------ utilitas statistik
def cramers_v(a, b) -> float:
    """Asosiasi dua variabel kategorik (0 = independen, 1 = identik), dengan koreksi bias Bergsma."""
    from scipy.stats import chi2_contingency
    t = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    if min(t.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.sum()
    r, c = t.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    r2, c2 = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    return float(math.sqrt(phi2 / max(1e-12, min(r2 - 1, c2 - 1))))


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


# ------------------------------------------------------------------ 1-2. blok per modalitas
def prep_block(x: np.ndarray, seed: int, var: float = 0.95, dmin: int = 8, dmax: int = 64):
    from sklearn.decomposition import PCA
    x = l2n(x.astype(np.float64))
    full = PCA(min(dmax, x.shape[1]), random_state=seed).fit(x)
    d = int(np.clip(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var) + 1, dmin, dmax))
    z = full.transform(x)[:, :d]
    info = {"dim": d, "varians_dijelaskan": float(full.explained_variance_ratio_[:d].sum())}
    return z, info


def unit_variance(z):
    z = z - z.mean(0)
    return z / math.sqrt(z.var(0).sum())


def microscope_check(z_spec: np.ndarray, mic: np.ndarray, seed: int, ks=(3, 4, 6, 8)) -> dict:
    """Apakah sebaran komposisi (cluster spektrum) sama di M1 dan M2?"""
    from sklearn.cluster import KMeans
    v = {k: cramers_v(KMeans(k, n_init=3, random_state=seed).fit_predict(z_spec), mic) for k in ks}
    return {"cramers_v_cluster_spektrum_vs_mikroskop": v, "maks": max(v.values()),
            "koreksi_dipakai": bool(max(v.values()) < MIC_V_MAX)}


def center_per_group(z, g):
    z = z.copy()
    for v in np.unique(g):
        z[g == v] -= z[g == v].mean(0)
    return z


def build_blocks(emb: dict, mic: np.ndarray, seed: int, log=print):
    """Return blok terproses {'gambar','spektrum'} (masing-masing total varians 1) + info."""
    blocks, info = {}, {}
    for m in MODALITIES:
        blocks[m], info[m] = prep_block(emb[m], seed)
    chk = microscope_check(blocks["spektrum"], mic, seed)
    info["cek_mikroskop"] = chk
    if chk["koreksi_dipakai"]:
        blocks["gambar"] = center_per_group(blocks["gambar"], mic)
    for m in MODALITIES:
        blocks[m] = unit_variance(blocks[m]).astype(np.float32)
    log(f"  blok: gambar {info['gambar']['dim']} dim ({info['gambar']['varians_dijelaskan']:.0%} var), "
        f"spektrum {info['spektrum']['dim']} dim ({info['spektrum']['varians_dijelaskan']:.0%} var); "
        f"Cramér's V cluster spektrum vs mikroskop maks {chk['maks']:.3f} -> "
        f"koreksi mikroskop {'DIPAKAI' if chk['koreksi_dipakai'] else 'tidak dipakai'}")
    return blocks, info


# ------------------------------------------------------------------ 3. FusionAE
class FusionAE(nn.Module):
    def __init__(self, d_in: int, zdim: int = 16, h=(256, 128)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]), nn.ReLU(),
                                 nn.Linear(h[1], zdim))
        self.dec = nn.Sequential(nn.Linear(zdim, h[1]), nn.ReLU(), nn.Linear(h[1], h[0]), nn.ReLU(),
                                 nn.Linear(h[0], d_in))
        self.zdim = zdim

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def block_mse(xhat, x, splits):
    """MSE per blok dirata-rata antar blok: tiap modalitas berbobot sama walau dimensinya beda."""
    out, s = 0.0, 0
    for d in splits:
        out = out + ((xhat[:, s:s + d] - x[:, s:s + d]) ** 2).mean()
        s += d
    return out / len(splits)


def modality_dropout(x, splits, p, gen):
    """Nolkan blok gambar (prob p) ATAU blok spektrum (prob p), tidak pernah keduanya."""
    B = x.shape[0]
    u = torch.rand(B, generator=gen, device=x.device)
    x = x.clone()
    x[u < p, :splits[0]] = 0
    x[(u >= p) & (u < 2 * p), splits[0]:] = 0
    return x


def train_fusion_ae(X: np.ndarray, splits, device, seed, epochs=60, bs=256, lr=1e-3, p_drop=0.25,
                    noise=0.1, zdim=16, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    col_std = Xt.std(0, keepdim=True)
    model = FusionAE(X.shape[1], zdim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        tot = 0.0
        for b in range(steps):
            xb = Xt[perm[b * bs:(b + 1) * bs]]
            xin = modality_dropout(xb, splits, p_drop, gen)
            xin = xin + noise * col_std * torch.randn(xin.shape, generator=gen, device=device)
            _, xhat = model(xin)
            loss = block_mse(xhat, xb, splits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
        hist.append(tot / len(X))
    model.eval()
    # kualitas: rekonstruksi penuh vs saat satu modalitas hilang (prediksi lintas-modal)
    with torch.no_grad():
        full = block_mse(model(Xt)[1], Xt, splits).item()
        cross = {}
        for j, m in enumerate(MODALITIES):
            xin = Xt.clone()
            if j == 0:
                xin[:, :splits[0]] = 0
            else:
                xin[:, splits[0]:] = 0
            xhat = model(xin)[1]
            s = 0 if j == 0 else splits[0]
            d = splits[j]
            # R² blok yang HILANG direkonstruksi dari modalitas lain
            mse = ((xhat[:, s:s + d] - Xt[:, s:s + d]) ** 2).sum(1).mean().item()
            var = ((Xt[:, s:s + d] - Xt[:, s:s + d].mean(0)) ** 2).sum(1).mean().item()
            cross[f"R2_{m}_dari_modalitas_lain"] = 1 - mse / var
    log(f"  FusionAE: loss akhir {hist[-1]:.4f}; rekonstruksi penuh {full:.4f}; " +
        ", ".join(f"{k} {v:.2f}" for k, v in cross.items()))
    return model, {"loss": hist, "mse_penuh": full, **cross}


@torch.no_grad()
def encode(model, X, device) -> np.ndarray:
    return model.enc(torch.from_numpy(X).to(device)).cpu().numpy()


# ------------------------------------------------------------------ 4. ablasi
def knn_accuracy(X, y, seed, n=10000, k=15) -> float:
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    return float(cross_val_score(KNeighborsClassifier(k), X[idx], y[idx], cv=5).mean())


def pairwise_separability(X, final, seed, n_per=1000, k=15) -> pd.DataFrame:
    """Untuk tiap pasangan cluster final (a, b): akurasi kNN (CV 5-fold) membedakan a vs b HANYA dari X.
    ~0,5 = representasi ini tidak bisa membedakan kedua cluster (informasinya hilang)."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(seed)
    K = final.max() + 1
    idx = {c: rng.permutation(np.flatnonzero(final == c))[:n_per] for c in range(K)}
    rows = []
    for a in range(K):
        for b in range(a + 1, K):
            i = np.concatenate([idx[a], idx[b]])
            y = (final[i] == b).astype(int)
            if min(np.bincount(y, minlength=2)) < 10:
                continue
            acc = cross_val_score(KNeighborsClassifier(min(k, len(i) // 10)), X[i], y, cv=5).mean()
            rows.append({"cluster_a": a, "cluster_b": b, "akurasi": float(acc)})
    return pd.DataFrame(rows)


LOST_ACC = 0.75     # pasangan dengan akurasi < 0,75 dianggap TIDAK terpisahkan oleh modalitas itu


def ablation(reps: dict, final: np.ndarray, mic: np.ndarray, k: int, seed: int, log=print):
    """reps: {'gambar saja','spektrum saja','concat','FusionAE'} -> representasi (N,d)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    rows, labels = [], {}
    for name, X in reps.items():
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        labels[name] = lab
        rows.append({"representasi": name, "dim": X.shape[1],
                     "AMI_vs_final": adjusted_mutual_info_score(final, lab),
                     "ARI_vs_final": adjusted_rand_score(final, lab),
                     "akurasi_kNN_cluster_final": knn_accuracy(X, final, seed),
                     "cramers_v_mikroskop": cramers_v(lab, mic),
                     "pasangan_final_tak_terpisah": "",
                     "akurasi_pasangan_min": np.nan})
        sep = pairwise_separability(X, final, seed)
        lost = sep[sep.akurasi < LOST_ACC]
        rows[-1]["pasangan_final_tak_terpisah"] = ", ".join(f"{a}-{b}" for a, b in zip(lost.cluster_a, lost.cluster_b))
        rows[-1]["akurasi_pasangan_min"] = float(sep.akurasi.min()) if len(sep) else np.nan
    df = pd.DataFrame(rows)
    log(df.round(3).to_string(index=False))
    return df, labels
