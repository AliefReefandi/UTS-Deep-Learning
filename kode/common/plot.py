"""Gaya grafik bersama untuk laporan (PNG statis, latar terang untuk dicetak)."""
import matplotlib

matplotlib.use("Agg")   # tanpa GUI: aman di Colab/terminal
import matplotlib.pyplot as plt  # noqa: E402

# Urutan warna kategorikal tetap (tidak diputar ulang); sudah divalidasi aman buta warna
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8984"
TEXT, TEXT2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.labelsize": 9, "axes.titleweight": "bold",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.fontsize": 8, "legend.frameon": False,
    "font.size": 9,
})


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  grafik -> {path}")
