"""Task 3b - encoder untuk SimCLR. Kecil sengaja: 36.000 sampel 64x64, target < 1,5 jam di T4.

ImageEncoder (~1,2 juta parameter): ResNet kecil. Stem 3x3 tanpa downsample (gambar hanya 64 px,
garis halus λ≈5 px akan hilang jika langsung di-stride), lalu 4 tahap residual 32-64-128-256
(64->64->32->16->8 px), global average pooling -> 256 dim.
  Global pooling: tekstur stasioner, jadi "di mana" tidak penting, yang penting "pola apa".

SpecEncoder (~0,3 juta parameter): 1D-CNN 1->32->64->128 kanal dengan stride (128->64->32->16),
lalu AdaptiveAvgPool ke 8 posisi dan flatten -> linear 128 dim.
  BUKAN global pooling: posisi absolut puncak adalah ciri komposisi (EDA #10: pola 3 puncak di posisi
  berbeda). Pool ke 8 posisi (16 kanal per posisi) menyimpan posisi kasar, sedangkan toleransi shift
  ±7 kanal dipelajari dari augmentasi.

ProjectionHead: MLP 2 lapis -> 128 dim, hanya untuk loss. Embedding yang dipakai clustering adalah
output ENCODER (sebelum head): head cenderung membuang informasi yang berguna untuk tugas lain
(Chen et al. 2020, SimCLR).
"""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class ImageEncoder(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(BasicBlock(cin, w, 1 if i == 0 else 2))
            cin = w
        self.layers = nn.Sequential(*layers)
        self.dim = widths[-1]

    def forward(self, x):
        return self.layers(self.stem(x)).mean((2, 3))


class SpecEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()

        def blk(cin, cout, k, s):
            return [nn.Conv1d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*blk(1, 32, 7, 1), *blk(32, 32, 7, 2), *blk(32, 64, 5, 1), *blk(64, 64, 5, 2),
                                  *blk(64, 128, 3, 1), *blk(128, 128, 3, 2))       # L 128 -> 16
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True))
        self.dim = dim

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)))


class ProjectionHead(nn.Module):
    def __init__(self, din: int, dout: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, din), nn.BatchNorm1d(din), nn.ReLU(inplace=True),
                                 nn.Linear(din, dout))

    def forward(self, h):
        return self.net(h)


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = ProjectionHead(encoder.dim, proj_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


def build(modality: str) -> SimCLR:
    return SimCLR(ImageEncoder() if modality == "gambar" else SpecEncoder())


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """Loss SimCLR (NT-Xent). Untuk tiap view, pasangan positifnya adalah view lain dari sampel yang
    sama; 2B-2 view lain dalam batch adalah negatif. Similarity = cosinus / tau."""
    z = nn.functional.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.t() / tau
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))                     # tidak membandingkan view dengan dirinya
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return nn.functional.cross_entropy(sim, target)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
