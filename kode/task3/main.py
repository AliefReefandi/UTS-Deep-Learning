"""Task 3 - Clustering multimodal (gambar mikrostruktur + spektrum), 36.000 sampel tanpa label.

Status: a-b selesai, c-f menyusul.
  a. eda.py        - distorsi gambar (rotasi, skala, brightness, M1 vs M2) & spektrum (shift, baseline, spike)
  b. ssl.py        - representation learning tanpa label (SimCLR/AE) + augmentasi
  c. fusion.py     - embedding gambar + spektrum, gabungan; ablasi satu modalitas
  d. cluster.py    - KMeans/GMM vs deep clustering (DEC)
  e. select_k.py   - jumlah cluster: silhouette, BIC, stabilitas
  f. viz.py        - UMAP/t-SNE
Output: <pred_dir>/cluster.csv  (id_sampel,cluster ; integer)
"""
import argparse
import copy

from kode.common import add_common_args, resolve_paths, set_seed

PRED_FILE = "cluster.csv"


def run(args):
    args = resolve_paths(args, "task3")
    set_seed(args.seed)
    if args.eda:
        from kode.task3 import eda
        eda.run(copy.copy(args))
    from kode.task3 import ssl
    try:
        emb = ssl.run(copy.copy(args))
    except ssl.TimeUp:
        print("[task3] SSL berhenti karena --max_minutes; jalankan ulang perintah yang sama untuk melanjutkan")
        return None
    print("[task3] c-f belum diimplementasi; cluster.csv belum dibuat")
    return emb


def build_parser():
    p = add_common_args(argparse.ArgumentParser(description="Task 3 - Clustering"))
    p.add_argument("--eda", action=argparse.BooleanOptionalAction, default=True,
                   help="buat ulang grafik EDA (3a)")
    from kode.task3.ssl import add_ssl_args
    return add_ssl_args(p)


if __name__ == "__main__":
    run(build_parser().parse_args())
