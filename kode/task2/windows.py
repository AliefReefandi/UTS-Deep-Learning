"""Task 2c - sliding window tanpa kebocoran masa depan + split per mesin.

Desain:
- Sampel = (mesin, siklus t). Input = baris [t-W+1 .. t] mesin itu SAJA; label = RUL pada t.
  Tidak ada baris sesudah t yang masuk input (diuji di test_leakage.py).
- Histori lebih pendek dari W (awal umur mesin) di-pad di KIRI dengan 0 (= rata-rata kondisi
  setelah normalisasi) + channel mask (1 = data asli, 0 = padding). Model jadi tahu berapa
  panjang histori yang benar-benar ada, dan semua panjang histori (bukan hanya >= W) ikut dilatih.
- W default 50 = panjang histori test terpendek, sehingga di test tidak ada padding.
- Fitur tambahan: umur mesin (siklus/300). Ini informasi masa lalu yang sah (diketahui saat prediksi).
- Split validasi PER MESIN (GroupKFold): semua baris satu mesin berada di fold yang sama.
- Validasi "seperti test": tiap mesin val dipotong di titik acak (seeded), dan hanya window
  terakhir dari histori terpotong itu yang dievaluasi, persis seperti prediksi di test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_SCALE = 300.0


class UnitArrays:
    """Simpan tiap mesin sebagai array (T, F) agar window dibuat cepat dengan indexing."""

    def __init__(self, df: pd.DataFrame, feat_cols: list[str], add_age: bool = True):
        self.feat_cols = list(feat_cols)
        self.add_age = add_age
        self.units, self.X, self.y, self.cycles = [], [], [], []
        for u, g in df.groupby("id_unit", sort=True):
            x = g[self.feat_cols].to_numpy(np.float32)
            if add_age:
                x = np.concatenate([x, (g["siklus"].to_numpy(np.float32) / AGE_SCALE)[:, None]], 1)
            self.units.append(u)
            self.X.append(x)
            self.cycles.append(g["siklus"].to_numpy())
            self.y.append(g["rul"].to_numpy(np.float32) if "rul" in g else None)

    @property
    def n_features(self) -> int:          # termasuk channel mask
        return self.X[0].shape[1] + 1

    def window(self, i: int, t: int, W: int) -> np.ndarray:
        """Window berakhir di indeks baris t (inklusif) mesin ke-i -> (W, F+1)."""
        x = self.X[i]
        lo = max(0, t - W + 1)
        seg = x[lo:t + 1]
        out = np.zeros((W, x.shape[1] + 1), np.float32)
        out[W - len(seg):, :-1] = seg
        out[W - len(seg):, -1] = 1.0         # mask
        return out


def all_end_indices(ua: UnitArrays, min_len: int = 1, stride: int = 1):
    """Semua (unit_idx, t) untuk training: tiap siklus jadi titik akhir window."""
    return [(i, t) for i, x in enumerate(ua.X) for t in range(min_len - 1, len(x), stride)]


def last_indices(ua: UnitArrays):
    """Satu sampel per mesin: siklus terakhir (dipakai untuk test)."""
    return [(i, len(x) - 1) for i, x in enumerate(ua.X)]


def truncated_indices(ua: UnitArrays, n_cuts: int = 5, min_len: int = 50, seed: int = 0):
    """Validasi meniru test: potong histori di titik acak (panjang >= min_len) beberapa kali
    per mesin; tiap potongan hanya dievaluasi pada siklus terakhirnya."""
    rng = np.random.default_rng(seed)
    idx = []
    for i, x in enumerate(ua.X):
        T = len(x)
        lo = min(min_len, T - 1) - 1
        ts = rng.integers(lo, T - 1, size=n_cuts)   # t <= T-2: berhenti SEBELUM rusak (RUL >= 1)
        idx += [(i, int(t)) for t in ts]
    return idx


def make_windows(ua: UnitArrays, idx, W: int):
    X = np.stack([ua.window(i, t, W) for i, t in idx]) if idx else np.zeros((0, W, ua.n_features), np.float32)
    y = np.array([ua.y[i][t] for i, t in idx], np.float32) if ua.y[0] is not None else None
    return X, y


def unit_folds(units, n_splits: int = 5, seed: int = 42):
    """GroupKFold per mesin, diacak dengan seed. Mengembalikan list (train_units, val_units)."""
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    parts = np.array_split(u, n_splits)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != k]), parts[k]) for k in range(n_splits)]


def to_torch_dataset(ua: UnitArrays, idx, W: int):
    """Dataset torch yang membuat window on-the-fly (hemat memori untuk ~250 rb sampel)."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, k):
            i, t = idx[k]
            x = torch.from_numpy(ua.window(i, t, W))
            y = torch.tensor(ua.y[i][t] if ua.y[i] is not None else np.nan)
            return x, y

    return _DS()


class WindowBank:
    """Semua mesin digabung jadi satu array (N_total, F+1) dengan W-1 baris nol (mask 0) di depan
    tiap mesin. Window yang berakhir di baris t mesin i = bank[pos-W+1 : pos+1] dengan
    pos = offset_i + W-1 + t. Satu batch dipotong lewat indexing (bisa di GPU), tanpa DataLoader.
    Hasilnya identik dengan UnitArrays.window (diuji di test_leakage.py)."""

    def __init__(self, ua: UnitArrays, W: int, device="cpu"):
        import torch
        self.W = W
        F = ua.X[0].shape[1]
        parts, self.offsets = [], []
        off = 0
        for x in ua.X:
            blk = np.zeros((W - 1 + len(x), F + 1), np.float32)
            blk[W - 1:, :F] = x
            blk[W - 1:, F] = 1.0
            parts.append(blk)
            self.offsets.append(off)
            off += len(blk)
        self.offsets = np.array(self.offsets)
        self.bank = torch.from_numpy(np.concatenate(parts)).to(device)
        self.ar = torch.arange(-W + 1, 1, device=device)
        self.y = [y for y in ua.y]
        self.device = device

    def positions(self, idx) -> np.ndarray:
        idx = np.asarray(idx).reshape(-1, 2)
        return self.offsets[idx[:, 0]] + self.W - 1 + idx[:, 1]

    def labels(self, idx) -> np.ndarray:
        return np.array([self.y[i][t] for i, t in idx], np.float32)

    def get(self, pos):
        """pos: tensor posisi (B,) -> tensor window (B, W, F+1)."""
        return self.bank[pos[:, None] + self.ar]


def report(args, W: int = 50):
    """Grafik label piecewise-linear + statistik window untuk laporan (2b-c)."""
    from kode.common import resolve_paths
    from kode.common.plot import MUTED, SERIES, plt, savefig
    from kode.common.utils import save_json
    from kode.task2.data import Preprocessor, add_rul_label, load_raw

    args = resolve_paths(args, "task2")
    out = args.task_out / "windows"
    out.mkdir(parents=True, exist_ok=True)
    tr, te = load_raw(args)
    tr = add_rul_label(tr)
    pp = Preprocessor().fit(tr)
    ua_tr = UnitArrays(pp.transform(tr), pp.sensors_)
    ua_te = UnitArrays(pp.transform(te), pp.sensors_)

    # Grafik 09: label RUL mentah vs cap untuk satu mesin, dengan contoh window
    i = 0
    g = tr[tr.id_unit == ua_tr.units[i]]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(g.siklus, g.rul_raw, color=MUTED, ls="--", label="RUL asli = siklus_maks − siklus")
    ax.plot(g.siklus, g.rul, color=SERIES[0], lw=2, label="label (cap 150)")
    t = len(g) - 80
    ax.axvspan(g.siklus.iloc[t - W + 1], g.siklus.iloc[t], color=SERIES[1], alpha=.15)
    ax.scatter([g.siklus.iloc[t]], [g.rul.iloc[t]], color=SERIES[1], zorder=3, s=25)
    ax.annotate(f"window W={W} → label RUL di ujung kanan\n(tidak ada baris sesudahnya yang dipakai)",
                (g.siklus.iloc[t], g.rul.iloc[t]), xytext=(-215, -45), textcoords="offset points",
                fontsize=8, color="#52514e")
    ax.set(xlabel="siklus", ylabel="RUL", title=f"Label RUL piecewise-linear, mesin {ua_tr.units[i]}")
    ax.legend(loc="upper right")
    savefig(fig, out / "09_label_dan_window.png")

    idx = all_end_indices(ua_tr)
    padded = sum(t < W - 1 for _, t in idx)
    stats = {"W": W, "n_fitur_per_langkah": ua_tr.n_features,
             "fitur": pp.sensors_ + ["umur(siklus/300)", "mask"],
             "n_window_train_total": len(idx), "pct_window_dgn_padding": 100 * padded / len(idx),
             "n_window_test": len(ua_te.units),
             "test_dgn_padding": int(sum(len(x) < W for x in ua_te.X))}
    save_json(stats, out / "statistik_window.json")
    print(stats)


if __name__ == "__main__":
    import argparse
    from kode.common import add_common_args
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--W", type=int, default=50)
    a = p.parse_args()
    report(a, a.W)
