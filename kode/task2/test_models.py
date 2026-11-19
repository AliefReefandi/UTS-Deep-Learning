"""Uji model Task 2 (cepat, tanpa data asli).

  python -m kode.task2.test_models

1. Bentuk output (B,) untuk semua arsitektur.
2. Mode eval deterministik: dua kali prediksi identik.
3. MC dropout benar-benar acak di SEMUA model (std > 0). Penting untuk Transformer: layer bawaan
   PyTorch memakai fast path saat eval() yang melewati dropout; blok kita ditulis manual.
4. Transformer mengabaikan isi posisi padding (mask=0) lewat key_padding_mask.
"""
import numpy as np
import torch

from kode.task2.models import MODELS, build
from kode.task2.train import predict
from kode.task2.windows import UnitArrays, WindowBank

W, F = 50, 16


def _bank(n_units=6, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        T = int(rng.integers(20, 90))                    # sebagian < W -> ada padding
        for c in range(1, T + 1):
            rows.append({"id_unit": u, "siklus": c, "rul": float(T - c),
                         **{f"f{j}": rng.normal() for j in range(F - 2)}})
    df = pd.DataFrame(rows)
    return WindowBank(UnitArrays(df, [f"f{j}" for j in range(F - 2)]), W)


def test_all():
    torch.manual_seed(0)
    bank = _bank()
    idx = [(i, len(y) - 1) for i, y in enumerate(bank.y)]          # siklus terakhir tiap unit
    for name in MODELS:
        m = build(name, F)
        x = bank.get(torch.from_numpy(bank.positions(idx)))
        assert m(x).shape == (len(idx),), name
        p1, p2 = predict(m, bank, idx), predict(m, bank, idx)
        assert np.array_equal(p1, p2), f"{name}: eval tidak deterministik"
        mc = predict(m, bank, idx, mc=20)
        assert mc.shape == (20, len(idx))
        assert mc.std(0).mean() > 1e-4, f"{name}: MC dropout tidak menghasilkan sebaran"
        assert not m.training, f"{name}: model harus kembali ke eval setelah MC"
        print(f"[OK] {name}: MC std rata-rata {mc.std(0).mean():.3f}")

    m = build("transformer", F).eval()
    x = torch.zeros(2, W, F)
    x[:, -10:, :-1] = torch.randn(2, 10, F - 1)
    x[:, -10:, -1] = 1.0                                 # 10 langkah asli, 40 padding
    x2 = x.clone()
    x2[:, :-10, :-1] = 99.0                              # ubah isi padding saja
    with torch.no_grad():
        assert torch.allclose(m(x), m(x2), atol=1e-5), "transformer melihat isi padding"
    print("[OK] transformer mengabaikan padding")


if __name__ == "__main__":
    test_all()
