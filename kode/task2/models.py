"""Task 2d - arsitektur sequence. Semua kecil (~30-60 rb parameter) agar 5-fold di T4 cepat.

Input: (B, W, F) dengan channel terakhir = mask (1 data asli, 0 padding kiri).
Output: RUL/150 (skala 0..1) -> dikali 150 dan di-clip [0, 150] saat prediksi.
Dropout ada di semua model supaya bisa dipakai untuk MC dropout (2f).
"""
from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, d: int, p: float):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LSTMReg(nn.Module):
    """LSTM 2 lapis. Membaca window dari kiri ke kanan; hidden state di langkah terakhir
    (= siklus saat prediksi) merangkum seluruh histori window. Padding ada di kiri, jadi
    langkah terakhir selalu data asli."""

    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 2, p: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(n_feat, hidden)
        self.rnn = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=p)
        self.head = Head(hidden, p)

    def forward(self, x):
        h, _ = self.rnn(torch.relu(self.inp(x)))
        return self.head(h[:, -1])


class CausalConvBlock(nn.Module):
    def __init__(self, c: int, k: int, dil: int, p: float):
        super().__init__()
        self.pad = (k - 1) * dil                    # pad kiri saja -> konvolusi kausal
        self.c1 = nn.Conv1d(c, c, k, dilation=dil)
        self.c2 = nn.Conv1d(c, c, k, dilation=dil)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(x + y)                    # residual


class TCNReg(nn.Module):
    """Temporal Convolutional Network: konvolusi 1D kausal ber-dilasi 1,2,4,8.
    Receptive field = 1 + 2*(k-1)*sum(dil) = 1 + 2*2*15 = 61 >= W=50, jadi output terakhir
    melihat seluruh window. Paralel di sepanjang waktu -> lebih cepat dari LSTM."""

    def __init__(self, n_feat: int, ch: int = 48, k: int = 3, dilations=(1, 2, 4, 8), p: float = 0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, ch, 1)
        self.blocks = nn.Sequential(*[CausalConvBlock(ch, k, d, p) for d in dilations])
        self.head = Head(ch, p)

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))   # (B, C, W)
        return self.head(h[:, :, -1])


MODELS = {"lstm": LSTMReg, "tcn": TCNReg}


def build(name: str, n_feat: int) -> nn.Module:
    return MODELS[name](n_feat)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
