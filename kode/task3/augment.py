"""Task 3b - augmentasi untuk SimCLR, dijalankan per batch di GPU (tanpa DataLoader CPU).

Setiap augmentasi menjawab satu temuan EDA (results/task3/eda/RINGKASAN.md). Prinsip SimCLR:
dua "view" dari sampel yang sama harus punya embedding mirip, jadi augmentasi = daftar perubahan
yang TIDAK mengubah jenis material. Yang mengubah jenis material (mis. skala besar) tidak dipakai.

Gambar (input uint8 (B,64,64) -> view float (B,1,36,36) terstandardisasi per gambar):
  crop 36x36 diputar       view diambil SELURUHNYA di dalam gambar asli untuk sudut & skala apa pun
                           (radius sudut crop 36*sqrt(2)/2/0.87 ~ 29 px < 32). Tanpa padding: versi
                           awal memakai padding reflection + translasi penuh dan menghasilkan pola
                           lipatan (garis lurus jadi chevron) = tekstur palsu. Translasi dibatasi agar
                           keempat sudut crop tetap di dalam gambar.
  rotasi 0-360 + flip      EDA #3: arah garis acak, bukan ciri material
  skala 0.87-1.15 (±15%)    EDA #4: variasi di dalam kelompok skala; tidak sampai menyatukan
                            garis halus (λ≈5) dan kasar (λ≈11), yang kemungkinan material berbeda
  posisi crop acak          tekstur stasioner: posisi tidak bermakna
  gamma 0.7-1.4             EDA #2: brightness/kontras LINEAR sudah dihapus standardisasi per
                            gambar, jadi yang ditambahkan adalah respons NONLINEAR
  blur (p=0.5) + noise (p=0.5)  EDA #5: M1 lebih halus, M2 lebih berderau (AUC 0.88 setelah
                            standardisasi) -> keduanya dijadikan view yang setara
  standardisasi per gambar  EDA #2: menghapus beda brightness M1/M2
Encoder memakai global average pooling, jadi bisa dilatih pada 36x36 dan dipakai (embedding) pada
gambar penuh 64x64.

Spektrum (input: hasil preprocess_spectrum, L2-norm; diskalakan ke RMS 1):
  shift ±7 kanal            EDA #11: shift = translasi global per spektrum, 95% dalam ±7
  stretch ±3%               EDA #11: std posisi naik dari puncak 1 ke 3 -> sedikit peregangan
  sisa baseline polinom-2 kecil   EDA #8: koreksi baseline tidak sempurna
  noise Gaussian            EDA #9
  spike sisa (p=0.2)        EDA #7: kebal terhadap spike yang lolos despike
  intensitas -> dinormalisasi ulang   EDA #12
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMG_CFG = dict(crop=36, scale=(0.87, 1.15), gamma=(0.7, 1.4), p_blur=0.5, blur_sigma=(0.5, 1.2),
               p_noise=0.5, noise_std=(0.05, 0.3))
SPEC_CFG = dict(shift=7.0, stretch=0.03, base_std=0.15, noise_std=(0.0, 0.08), p_spike=0.2,
                spike_amp=(1.0, 3.0))


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Per gambar: (x - mean) / std. x: (B,1,H,W)."""
    m = x.mean((2, 3), keepdim=True)
    s = x.std((2, 3), keepdim=True)
    return (x - m) / s.clamp_min(1e-3)


def img_to_input(img_u8: torch.Tensor) -> torch.Tensor:
    """Tanpa augmentasi: uint8 (B,H,W) -> float terstandardisasi (B,1,H,W)."""
    return standardize(img_u8.float().unsqueeze(1) / 255.0)


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss_kernel(sigma, x.device)
    r = (len(k) - 1) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def augment_images(img_u8: torch.Tensor, gen: torch.Generator, cfg: dict = IMG_CFG) -> torch.Tensor:
    """img_u8: (B,H,W) uint8 di device mana pun; gen: torch.Generator di device yang sama."""
    B, H, W = img_u8.shape
    assert H == W
    dev = img_u8.device
    u = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    x = img_u8.float().unsqueeze(1) / 255.0

    # --- geometri: crop SxS diputar + skala + flip, seluruhnya di dalam gambar (tanpa padding) ---
    S = cfg["crop"]
    ang = u(B) * 2 * math.pi
    lo, hi = cfg["scale"]
    sc = torch.exp(math.log(lo) + u(B) * (math.log(hi) - math.log(lo)))   # log-uniform; > 1 = perbesar
    flip = torch.where(u(B) < 0.5, -1.0, 1.0)
    a = (S / W) / sc                       # setengah sisi crop dalam koordinat ternormalisasi input
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = a * cos * flip, -a * sin
    theta[:, 1, 0], theta[:, 1, 1] = a * sin * flip, a * cos
    reach = a * (cos.abs() + sin.abs())    # jangkauan sudut crop terjauh per sumbu
    room = (1 - 1 / W - reach).clamp_min(0)   # sisa ruang (minus setengah piksel) untuk translasi
    theta[:, :, 2] = (u(B, 2) * 2 - 1) * room[:, None]
    grid = F.affine_grid(theta, (B, 1, S, S), align_corners=False)
    # padding "zeros": jika ada sampel di luar gambar, nilainya turun ke 0 dan langsung terlihat di uji
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # --- gamma (nonlinear) ---
    g = cfg["gamma"][0] + u(B) * (cfg["gamma"][1] - cfg["gamma"][0])
    x = x.clamp(1e-4, 1).pow(g.view(B, 1, 1, 1))

    # --- blur: 3 tingkat sigma dihitung sekali untuk seluruh batch, dipilih per sampel ---
    s_lo, s_hi = cfg["blur_sigma"]
    levels = torch.stack([_blur(x, s) for s in (s_lo, (s_lo + s_hi) / 2, s_hi)])   # (3,B,1,H,W)
    pick = (u(B) * 3).long().clamp_max(2)
    blurred = levels[pick, torch.arange(B, device=dev)]                    # (B,1,S,S)
    do_blur = (u(B) < cfg["p_blur"]).view(B, 1, 1, 1)
    x = torch.where(do_blur, blurred, x)

    # --- standardisasi, lalu noise dalam satuan std gambar ---
    x = standardize(x)
    n_lo, n_hi = cfg["noise_std"]
    nstd = (n_lo + u(B) * (n_hi - n_lo)) * (u(B) < cfg["p_noise"]).float()
    x = x + nstd.view(B, 1, 1, 1) * torch.randn(x.shape, generator=gen, device=dev)
    return standardize(x)


def spec_to_input(spec_pp: torch.Tensor) -> torch.Tensor:
    """Spektrum hasil preprocess (L2-norm) -> RMS 1, bentuk (B,1,L)."""
    return (spec_pp * math.sqrt(spec_pp.shape[-1])).unsqueeze(1)


def augment_spectra(spec_pp: torch.Tensor, gen: torch.Generator, cfg: dict = SPEC_CFG) -> torch.Tensor:
    """spec_pp: (B,L) spektrum setelah preprocess_spectrum. Keluaran (B,1,L), RMS 1."""
    B, L = spec_pp.shape
    dev = spec_pp.device
    u = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    x = spec_to_input(spec_pp)                                             # (B,1,L)

    # --- shift + stretch: resample linear di posisi baru (padding = nilai tepi) ---
    pos = torch.linspace(-1, 1, L, device=dev)
    shift = (u(B) * 2 - 1) * cfg["shift"] * 2 / (L - 1)                    # kanal -> koordinat [-1,1]
    stretch = 1 + (u(B) * 2 - 1) * cfg["stretch"]
    src = pos[None, :] / stretch[:, None] - shift[:, None]                 # sampel diambil dari sini
    grid = torch.stack([src, torch.zeros_like(src)], -1).unsqueeze(1)      # (B,1,L,2)
    x = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", padding_mode="border",
                      align_corners=True).squeeze(2)

    # --- sisa baseline polinom-2 ---
    coef = torch.randn(B, 3, generator=gen, device=dev) * cfg["base_std"]
    x = x + (coef[:, :1] + coef[:, 1:2] * pos + coef[:, 2:3] * pos ** 2).unsqueeze(1)

    # --- noise ---
    n_lo, n_hi = cfg["noise_std"]
    x = x + (n_lo + u(B, 1, 1) * (n_hi - n_lo)) * torch.randn(x.shape, generator=gen, device=dev)

    # --- spike sisa (1 kanal) ---
    has = u(B) < cfg["p_spike"]
    ch = (u(B) * L).long().clamp_max(L - 1)
    amp = cfg["spike_amp"][0] + u(B) * (cfg["spike_amp"][1] - cfg["spike_amp"][0])
    x[has, 0, ch[has]] += amp[has] * x[has, 0].abs().amax(-1)

    # --- normalisasi ulang (intensitas tidak bermakna) ---
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
