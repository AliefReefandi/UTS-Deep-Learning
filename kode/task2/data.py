"""Task 2 - pemuatan data, label RUL, dan preprocessing.

Keputusan desain (hasil EDA, lihat eda.py dan results/task2/eda/):
1. Kondisi operasi: setel_1 hanya punya 6 nilai (0,10,20,25,35,42) dan setel_2/setel_3
   ditentukan sepenuhnya oleh setel_1 -> ada 6 rezim operasi. Rezim menjelaskan 37-100% varians
   sensor, jadi tren degradasi tertutup oleh loncatan antar-rezim.
   => normalisasi z-score PER KONDISI (mean/std dihitung per rezim, hanya dari unit train).
2. Sensor tidak informatif: setelah normalisasi per kondisi, 7 sensor (s01,s05,s06,s10,s16,
   s18,s19) adalah white noise murni: std(selisih antar-siklus)/std ~ sqrt(2)=1.41, sedangkan
   sensor bertren < 1. Kriteria ini TANPA label, jadi tidak ada kebocoran.
   Korelasi linear dengan RUL sengaja tidak dipakai: ada 2 mode kerusakan dengan arah tren
   berlawanan (mis. s11 naik di satu mode, turun di mode lain), sehingga korelasi global ~0
   padahal sensornya informatif.
3. Missing (~2% per sensor, hampir acak, tidak bergantung RUL, juga muncul di baris terakhir
   test): forward-fill per unit (hanya memakai masa lalu), awal histori diisi 0 = rata-rata kondisi.
4. Spike (~0.2% titik, |z| sampai 10): dideteksi KAUSAL terhadap median beberapa siklus
   sebelumnya; jika menyimpang > k * sigma robust, diganti median tersebut. Lalu clip ke [-5, 5].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kode.common.config import data_path

SENSORS = [f"s{i:02d}" for i in range(1, 22)]
SETTINGS = ["setel_1", "setel_2", "setel_3"]
RUL_CAP = 150


def load_raw(args, debug_units: tuple[int, int] = (80, 40)):
    """Baca train & test. Mode --debug: ambil subset unit (bukan subset baris) agar
    histori tiap mesin tetap utuh."""
    tr = pd.read_csv(data_path(args, "task2_train"))
    te = pd.read_csv(data_path(args, "task2_test"))
    if getattr(args, "debug", False):
        rng = np.random.default_rng(args.seed)
        tr = tr[tr.id_unit.isin(rng.choice(tr.id_unit.unique(), debug_units[0], replace=False))]
        te = te[te.id_unit.isin(rng.choice(te.id_unit.unique(), debug_units[1], replace=False))]
    tr = tr.sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    te = te.sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    return add_condition(tr), add_condition(te)


def add_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Rezim operasi = setel_1 dibulatkan (6 nilai diskret, noise kecil ~0.005)."""
    df = df.copy()
    df["kond"] = df["setel_1"].round(0).astype(int)
    return df


def add_rul_label(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Train run-to-failure: siklus terakhir = rusak, jadi RUL = max_siklus - siklus,
    lalu dipotong di `cap` (asumsi piecewise-linear: mesin sehat di awal umur tidak
    menunjukkan degradasi, sehingga RUL > 150 tidak bisa dibedakan dari 150)."""
    df = df.copy()
    rul = df.groupby("id_unit")["siklus"].transform("max") - df["siklus"]
    df["rul_raw"] = rul
    df["rul"] = rul.clip(upper=cap).astype(np.float32)
    return df


class Preprocessor:
    """Di-fit HANYA pada unit train (per fold), lalu dipakai untuk val/test."""

    def __init__(self, noise_thr: float = 1.2, spike_k: float = 5.0, spike_win: int = 10,
                 clip: float = 5.0, select: bool = True):
        self.noise_thr, self.spike_k, self.spike_win = noise_thr, spike_k, spike_win
        self.clip, self.select = clip, select

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        g = train_df.groupby("kond")[SENSORS]
        self.mean_ = g.mean()            # pandas mengabaikan NaN
        self.std_ = g.std().replace(0, 1.0)
        z = self._normalize(train_df)
        # rasio ~1.41 = white noise (tidak ada autokorelasi/tren); < 1 = sinyal halus bertren
        self.noise_ratio_ = z.groupby(train_df["id_unit"]).diff().std() / z.std()
        self.sensors_ = [s for s in SENSORS
                         if not self.select or self.noise_ratio_[s] < self.noise_thr]
        # sigma robust residual terhadap median kausal, untuk ambang spike
        resid = z[self.sensors_] - self._causal_median(z[self.sensors_], train_df["id_unit"])
        self.sigma_ = 1.4826 * resid.abs().median()
        return self

    def _normalize(self, df):
        m = self.mean_.loc[df["kond"]].to_numpy()
        s = self.std_.loc[df["kond"]].to_numpy()
        return pd.DataFrame((df[SENSORS].to_numpy() - m) / s, columns=SENSORS, index=df.index)

    def _causal_median(self, z, units):
        # median dari spike_win siklus SEBELUMNYA (shift 1) -> tidak melihat masa depan
        return z.groupby(units).transform(
            lambda s: s.shift(1).rolling(self.spike_win, min_periods=3).median())

    def transform(self, df: pd.DataFrame, return_masks: bool = False):
        z = self._normalize(df)[self.sensors_]
        units = df["id_unit"]
        nan_mask = z.isna()
        med = self._causal_median(z, units)
        spike = ((z - med).abs() > self.spike_k * self.sigma_) & med.notna()
        z = z.mask(spike, med)
        z = z.groupby(units).ffill().fillna(0.0).clip(-self.clip, self.clip)
        out = df[["id_unit", "siklus", "kond"]].copy()
        if "rul" in df:
            out["rul"] = df["rul"]
        out[self.sensors_] = z.astype(np.float32)
        return (out, nan_mask, spike) if return_masks else out
