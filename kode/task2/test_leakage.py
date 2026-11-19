"""Uji label RUL dan anti-kebocoran Task 2.

  python -m kode.task2.test_leakage                  # data sintetis + data asli (subset)
  python -m pytest kode/task2/test_leakage.py        # jika pytest terpasang

Yang diuji:
 1. Label: RUL = max_siklus - siklus, cap 150, RUL=0 di siklus terakhir train.
 2. Kausalitas end-to-end: mengubah SEMUA baris sesudah t (sensor, NaN, spike) tidak mengubah
    window yang berakhir di t (preprocessing + windowing).
 3. Isolasi antar-mesin: mengubah mesin lain tidak mengubah window mesin ini.
 4. Fit hanya di train: statistik Preprocessor tidak berubah jika data val diubah.
 5. Split per mesin: tidak ada mesin yang muncul di train dan val sekaligus; tiap mesin di
    tepat satu fold val.
 6. Window test berakhir tepat di siklus terakhir tiap mesin; padding+mask benar.
 7. WindowBank (versi cepat untuk training) identik dengan window referensi.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from kode.task2.data import SENSORS, Preprocessor, add_condition, add_rul_label
from kode.task2.windows import (UnitArrays, WindowBank, all_end_indices, last_indices, make_windows,
                                truncated_indices, unit_folds)

W = 30


def synthetic(n_units=12, seed=0):
    """Data mirip asli: 6 kondisi, tren degradasi, NaN, spike."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        T = int(rng.integers(60, 220))
        for c in range(1, T + 1):
            k = rng.choice([0, 10, 20, 25, 35, 42])
            r = {"id_unit": u, "siklus": c, "setel_1": k + rng.normal(0, .003),
                 "setel_2": k / 50, "setel_3": 60.0 if k == 25 else 100.0}
            for j, s in enumerate(SENSORS):
                trend = (c / T) ** 3 * (j % 3 - 1) * 5 if j % 4 else 0     # sebagian sensor noise
                r[s] = 100 + 10 * k * (j + 1) / 21 + trend + rng.normal()
            rows.append(r)
    df = pd.DataFrame(rows)
    m = rng.random((len(df), len(SENSORS))) < .02
    df[SENSORS] = df[SENSORS].mask(m)
    sp = rng.random((len(df), len(SENSORS))) < .003
    df[SENSORS] = df[SENSORS].where(~sp, df[SENSORS] + 40)
    return add_rul_label(add_condition(df))


def _pipeline(train_df, df):
    pp = Preprocessor().fit(train_df)
    ua = UnitArrays(pp.transform(df), pp.sensors_)
    return pp, ua


def test_label(df=None):
    df = synthetic() if df is None else df
    for u, g in df.groupby("id_unit"):
        T = g.siklus.max()
        assert g.rul.iloc[-1] == 0, "RUL di siklus terakhir harus 0"
        assert np.allclose(g.rul, np.minimum(T - g.siklus, 150))
        assert g.rul.max() <= 150 and g.rul.min() >= 0
        assert (np.diff(g.rul.to_numpy()) <= 0).all(), "RUL harus tidak naik"


def test_causal_end_to_end(df=None, n_checks=40, seed=1):
    df = synthetic() if df is None else df
    rng = np.random.default_rng(seed)
    pp = Preprocessor().fit(df)
    ua = UnitArrays(pp.transform(df), pp.sensors_)
    for _ in range(n_checks):
        i = int(rng.integers(len(ua.units)))
        u = ua.units[i]
        T = len(ua.X[i])
        t = int(rng.integers(0, T - 1))
        before = ua.window(i, t, W)
        # rusak total masa depan: ganti semua sensor sesudah t dengan nilai acak besar / NaN
        df2 = df.copy()
        fut = (df2.id_unit == u) & (df2.siklus > ua.cycles[i][t])
        noise = rng.normal(0, 50, (fut.sum(), len(SENSORS)))
        noise[rng.random(noise.shape) < .3] = np.nan
        df2.loc[fut, SENSORS] = df2.loc[fut, SENSORS].to_numpy() + noise
        ua2 = UnitArrays(pp.transform(df2), pp.sensors_)   # pp tetap (fit di train)
        after = ua2.window(i, t, W)
        assert np.array_equal(before, after), f"bocor masa depan: unit {u}, t={t}"


def test_unit_isolation(df=None):
    df = synthetic() if df is None else df
    pp = Preprocessor().fit(df)
    ua = UnitArrays(pp.transform(df), pp.sensors_)
    df2 = df.copy()
    other = df2.id_unit != ua.units[0]
    df2.loc[other, SENSORS] = df2.loc[other, SENSORS] * 3
    ua2 = UnitArrays(pp.transform(df2), pp.sensors_)
    assert np.array_equal(ua.X[0], ua2.X[0]), "fitur satu mesin dipengaruhi mesin lain"


def test_fit_train_only(df=None):
    df = synthetic() if df is None else df
    folds = unit_folds(df.id_unit.unique(), 4, seed=0)
    tr_u, va_u = folds[0]
    tr = df[df.id_unit.isin(tr_u)]
    pp1 = Preprocessor().fit(tr)
    df2 = df.copy()
    v = df2.id_unit.isin(va_u)
    df2.loc[v, SENSORS] = df2.loc[v, SENSORS] * 10 + 999
    pp2 = Preprocessor().fit(df2[df2.id_unit.isin(tr_u)])
    pd.testing.assert_frame_equal(pp1.mean_, pp2.mean_)
    pd.testing.assert_frame_equal(pp1.std_, pp2.std_)
    assert pp1.sensors_ == pp2.sensors_


def test_unit_split(units=None, k=5):
    units = np.arange(1, 101) if units is None else units
    folds = unit_folds(units, k, seed=42)
    seen = []
    for tr_u, va_u in folds:
        assert not set(tr_u) & set(va_u), "mesin bocor antara train dan val"
        assert set(tr_u) | set(va_u) == set(units)
        seen += list(va_u)
    assert sorted(seen) == sorted(set(units)), "tiap mesin harus di tepat satu fold val"


def test_windows_shape_and_padding(df=None):
    df = synthetic() if df is None else df
    pp, ua = _pipeline(df, df)
    # window di awal umur: harus ber-padding dengan mask 0
    w0 = ua.window(0, 4, W)
    assert w0.shape == (W, ua.n_features)
    assert (w0[:W - 5, -1] == 0).all() and (w0[W - 5:, -1] == 1).all()
    assert (w0[:W - 5] == 0).all()
    # window terakhir = baris terakhir mesin
    X, _ = make_windows(ua, last_indices(ua), W)
    for k, x in enumerate(ua.X):
        assert np.array_equal(X[k, -1, :-1], x[-1])
    # validasi terpotong: tidak pernah pada siklus rusak (RUL >= 1)
    idx = truncated_indices(ua, 5, min_len=20, seed=0)
    _, y = make_windows(ua, idx, W)
    assert (y >= 1).all()
    # semua indeks training valid
    idx = all_end_indices(ua)
    assert len(idx) == len(df)


def test_bank_equals_reference(df=None):
    """WindowBank (cepat, dipakai training) harus identik dengan UnitArrays.window (referensi)."""
    import torch
    df = synthetic() if df is None else df
    _, ua = _pipeline(df, df)
    bank = WindowBank(ua, W)
    idx = all_end_indices(ua)
    Xref, yref = make_windows(ua, idx, W)
    Xb = bank.get(torch.from_numpy(bank.positions(idx))).numpy()
    assert np.array_equal(Xref, Xb), "WindowBank berbeda dari referensi"
    assert np.array_equal(yref, bank.labels(idx))


def test_real_data(data_dir, n_units=60):
    """Ulangi uji kausal & label pada subset data asli."""
    tr = pd.read_csv(f"{data_dir}/rul_train.csv.gz")
    keep = np.random.default_rng(0).choice(tr.id_unit.unique(), n_units, replace=False)
    tr = add_rul_label(add_condition(tr[tr.id_unit.isin(keep)].sort_values(["id_unit", "siklus"])))
    test_label(tr)
    test_causal_end_to_end(tr, n_checks=15)
    test_unit_isolation(tr)
    test_fit_train_only(tr)
    test_unit_split(tr.id_unit.unique())
    test_windows_shape_and_padding(tr)
    test_bank_equals_reference(tr)


if __name__ == "__main__":
    from kode.common.config import DATA_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(DATA_DIR))
    a = p.parse_args()
    tests = [test_label, test_causal_end_to_end, test_unit_isolation, test_fit_train_only,
             test_unit_split, test_windows_shape_and_padding, test_bank_equals_reference]
    for f in tests:
        f()
        print(f"[OK] {f.__name__} (sintetis)")
    test_real_data(a.data_dir)
    print("[OK] semua uji diulang pada 60 mesin data asli")
