"""Uji Task 3b (cepat, CPU, data sintetis).

  python -m kode.task3.test_ssl

1. Augmentasi gambar: bentuk (crop 36x36), nilai hingga, terstandardisasi; deterministik; crop
   tidak pernah mengambil di luar gambar (gambar konstan, padding nol: view tidak boleh lebih gelap).
2. Augmentasi spektrum dengan semua parameter 0 = identitas (memvalidasi matematika grid shift).
3. Shift spektrum memindahkan puncak sejauh yang diminta.
4. NT-Xent: pasangan positif identik -> loss jauh lebih kecil daripada pasangan acak.
5. Resume: 2 epoch tanpa putus == 1 epoch + resume + 1 epoch (bobot identik).
6. Embedding gambar dengan TTA dihedral invarian terhadap rotasi 90° dan flip.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from kode.task3.augment import SPEC_CFG, augment_images, augment_spectra, spec_to_input
from kode.task3.encoders import build, nt_xent
from kode.task3.ssl import TimeUp, embed, train_ssl


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    img = np.stack([np.clip(128 + 50 * np.sin(2 * np.pi * (x * np.cos(a) + y * np.sin(a)) / l), 0, 255)
                    for a, l in zip(rng.uniform(0, np.pi, n), rng.choice([5, 11], n))]).astype(np.uint8)
    ch = np.arange(128)
    spec = np.stack([sum(np.exp(-(ch - c) ** 2 / 18) for c in rng.choice([20, 35, 60, 75, 100], 3, replace=False))
                     for _ in range(n)]).astype(np.float32)
    spec /= np.linalg.norm(spec, axis=1, keepdims=True)
    return torch.from_numpy(img), torch.from_numpy(spec)


def test_augment_images():
    img, _ = _data()
    a = augment_images(img, torch.Generator().manual_seed(1))
    b = augment_images(img, torch.Generator().manual_seed(1))
    assert a.shape == (64, 1, 36, 36) and torch.isfinite(a).all()
    assert torch.allclose(a.mean((2, 3)), torch.zeros(64, 1), atol=1e-4)
    assert torch.allclose(a.std((2, 3)), torch.ones(64, 1), atol=1e-3)
    assert torch.equal(a, b), "augmentasi tidak deterministik untuk seed sama"
    from kode.task3.augment import IMG_CFG
    frame = torch.full((2000, 64, 64), 200, dtype=torch.uint8)   # di luar gambar = 0 (padding zeros)
    geo = dict(IMG_CFG, gamma=(1.0, 1.0), p_blur=0.0, p_noise=0.0)
    from kode.task3 import augment as A
    raw = A.standardize                                 # nonaktifkan standardisasi untuk cek nilai mentah
    try:
        A.standardize = lambda x: x
        v = augment_images(frame, torch.Generator().manual_seed(2), geo)
    finally:
        A.standardize = raw
    assert v.min() > 0.99 * 200 / 255, f"crop mengambil di luar gambar: min {v.min():.4f}"
    print("[OK] augmentasi gambar (crop selalu di dalam gambar)")


def test_spectrum_identity_and_shift():
    _, spec = _data()
    zero = dict(SPEC_CFG, shift=0.0, stretch=0.0, base_std=0.0, noise_std=(0.0, 0.0), p_spike=0.0)
    out = augment_spectra(spec, torch.Generator().manual_seed(0), zero)
    assert torch.allclose(out, spec_to_input(spec), atol=1e-4), "augmentasi nol bukan identitas"
    # shift acak U(-6, 6): satu puncak tunggal di kanal 64 harus berpindah paling jauh 6 kanal,
    # dan rata-rata |perpindahan| ~3 (seragam). Satu puncak, agar argmax tidak melompat antar-puncak.
    one = dict(zero, shift=6.0)
    ch = torch.arange(128, dtype=torch.float32)
    x = torch.exp(-(ch - 64) ** 2 / 18).repeat(400, 1)
    out = augment_spectra(x, torch.Generator().manual_seed(0), one)[:, 0]
    d = (out.argmax(1) - 64).abs()
    assert d.max() <= 6 and 2.0 < d.float().mean() < 4.0, f"shift tidak sesuai: maks {d.max()}, rata {d.float().mean()}"
    print("[OK] augmentasi spektrum (identitas & shift)")


def test_nt_xent():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    same = nt_xent(z, z, 0.2).item()
    rand = nt_xent(z, torch.randn(32, 16), 0.2).item()
    assert same < rand - 1.0, (same, rand)
    print(f"[OK] NT-Xent: identik {same:.3f} < acak {rand:.3f}")


def test_resume_identical():
    img, spec = _data(48)
    dev = torch.device("cpu")
    cfg = dict(epochs=2, bs=16, lr=1e-3, wd=1e-4, tau=0.2, warmup=0.05, resume=True)
    for m, data in [("gambar", img), ("spektrum", spec)]:
        with tempfile.TemporaryDirectory() as d:
            a, _ = train_ssl(m, data, dict(cfg, resume=False), Path(d) / "a.pt", dev, seed=3, log=lambda *_: None)
            # tiru sesi putus: konfigurasi SAMA, deadline sudah lewat -> berhenti rapi setelah epoch 1
            try:
                train_ssl(m, data, cfg, Path(d) / "b.pt", dev, seed=3, log=lambda *_: None, deadline=0)
                raise AssertionError("TimeUp tidak terjadi")
            except TimeUp:
                pass
            b, st = train_ssl(m, data, cfg, Path(d) / "b.pt", dev, seed=3, log=lambda *_: None)
            assert st["epoch"] == 2
        for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
            assert torch.equal(va, vb), f"{m}: resume berbeda di {k}"
    print("[OK] resume identik (gambar & spektrum)")


def test_tta_rotation_invariance():
    img, _ = _data(8)
    m = build("gambar").eval()
    e0 = embed(m, img, "gambar")
    e1 = embed(m, torch.rot90(img, 1, (1, 2)).contiguous(), "gambar")
    e2 = embed(m, img.flip(2).contiguous(), "gambar")
    assert np.allclose(e0, e1, atol=1e-4) and np.allclose(e0, e2, atol=1e-4)
    print("[OK] embedding TTA invarian rotasi 90° & flip")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)
    test_augment_images()
    test_spectrum_identity_and_shift()
    test_nt_xent()
    test_resume_identical()
    test_tta_rotation_invariance()
