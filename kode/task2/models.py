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


class AttnBlock(nn.Module):
    """Pre-norm Transformer block. Ditulis manual (bukan nn.TransformerEncoderLayer) karena layer
    bawaan memakai 'fast path' saat eval() yang melewati dropout -> MC dropout diam-diam mati.
    Di sini semua dropout adalah nn.Dropout biasa, sehingga bisa dinyalakan saat inferensi."""

    def __init__(self, d: int, heads: int, p: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, pad):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class TransformerReg(nn.Module):
    """Transformer encoder kecil: proyeksi linear + embedding posisi (dipelajari) -> 2 blok
    self-attention -> token terakhir (siklus saat prediksi) ke head.
    - Padding kiri (mask=0) diabaikan lewat key_padding_mask, jadi attention hanya ke data asli.
    - Tidak perlu causal mask: window hanya berisi masa lalu sampai t (kebocoran sudah dicegah
      di windowing), dan yang dibaca hanya output di t.
    Beda dari LSTM/TCN: tiap langkah bisa langsung 'melihat' langkah mana pun di window."""

    def __init__(self, n_feat: int, d: int = 64, heads: int = 4, layers: int = 2, p: float = 0.2,
                 max_len: int = 512):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([AttnBlock(d, heads, p) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = Head(d, p)

    def forward(self, x):
        W = x.shape[1]
        pad = x[:, :, -1] < 0.5                       # True = padding (diabaikan attention)
        h = self.inp(x) + self.pos[:, -W:]            # posisi dihitung dari kanan: t selalu sama
        for b in self.blocks:
            h = b(h, pad)
        return self.head(self.norm(h[:, -1]))


MODELS = {"lstm": LSTMReg, "tcn": TCNReg, "transformer": TransformerReg}


def build(name: str, n_feat: int) -> nn.Module:
    return MODELS[name](n_feat)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
