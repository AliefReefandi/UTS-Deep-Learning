"""PEMBUNGKUS run_all.py untuk Task 1. BUKAN implementasi Task 1.

Seluruh kode Task 1 (milik Alief) ada di kode/01_Task1_Classification.ipynb. File ini hanya
mengeksekusi notebook itu dengan nbconvert supaya `python run_all.py` mencakup Task 1.

- Source notebook TIDAK diubah. Yang dieksekusi adalah salinan di memori: satu sel pembuka
  disisipkan di depan, lalu semua sel Alief apa adanya. Hasil eksekusi disimpan ke
  <out_dir>/task1/01_Task1_Classification_executed.ipynb.
- Hanya jalan di Colab: notebook memakai google.colab (drive.mount, files.download) dan path
  /content/.... Di lokal atau dengan --debug, Task 1 dilewati dengan pesan.
- Sel pembuka membuat path zip bisa diatur: notebook membuka
  "/content/drive/MyDrive/Copy of data_uts_dl.zip" (hardcoded); path itu dialihkan ke
  --task1_zip / env UTS_ZIP_PATH (default /content/drive/MyDrive/data_uts_dl.zip).
  Juga: drive.mount -> cek saja (Drive harus sudah di-mount oleh notebook Colab pemanggil,
  karena kernel nbconvert tidak punya frontend untuk otorisasi), files.download -> dilewati.
- Setelah selesai: prediksi_klasifikasi.csv disalin ke --pred_dir (+ prediksi/), model ke <out_dir>/task1/.

  python -m kode.task1.main --task1_zip /content/drive/MyDrive/data_uts_dl.zip
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import shutil

from kode.common import add_common_args, resolve_paths
from kode.common.config import REPO_ROOT

PRED_FILE = "prediksi_klasifikasi.csv"   # header: id_unit,kelas
NOTEBOOK = REPO_ROOT / "kode" / "01_Task1_Classification.ipynb"
HARDCODED_ZIP = "/content/drive/MyDrive/Copy of data_uts_dl.zip"
DEFAULT_ZIP = os.environ.get("UTS_ZIP_PATH", "/content/drive/MyDrive/data_uts_dl.zip")
ARTEFAK = ["feature_fusion_final.pth", "decision_fusion_best.pth"]

PREAMBLE = '''# [Sel ini disisipkan oleh kode/task1/main.py - BUKAN bagian notebook Alief]
import os, zipfile
import google.colab.drive, google.colab.files
_ZIP_ASLI, _ZIP_BARU = {hard!r}, {zip!r}
class _ZipDialihkan(zipfile.ZipFile):
    def __init__(self, file, *a, **k):
        if file == _ZIP_ASLI:
            file = _ZIP_BARU
        super().__init__(file, *a, **k)
zipfile.ZipFile = _ZipDialihkan
def _mount(mountpoint, *a, **k):
    if not os.path.isdir(os.path.join(mountpoint, "MyDrive")):
        raise RuntimeError("Drive belum di-mount. Jalankan drive.mount di notebook Colab dulu.")
    print("(pembungkus) Drive sudah ter-mount di", mountpoint)
google.colab.drive.mount = _mount
google.colab.files.download = lambda p, *a, **k: print("(pembungkus) download dilewati:", p)
'''


def in_colab() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:          # paket 'google' sama sekali tidak ada
        return False


def build_notebook(zip_path: str):
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb = copy.deepcopy(nb)
    nb.cells.insert(0, nbformat.v4.new_code_cell(PREAMBLE.format(hard=HARDCODED_ZIP, zip=zip_path)))
    return nb


def run(args):
    if args.debug:
        print("[task1] --debug: notebook Task 1 (Alief) dilewati. Jalankan penuh di Colab.")
        return None
    if not in_colab():
        print("[task1] bukan Colab: notebook Task 1 memakai google.colab dan /content, dilewati. "
              f"Prediksi yang ada tetap: {PRED_FILE} di root repo.")
        return None
    args = resolve_paths(args, "task1")
    if not os.path.exists(args.task1_zip):
        raise FileNotFoundError(f"zip data tidak ada: {args.task1_zip} (atur --task1_zip / UTS_ZIP_PATH)")
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat
    work = args.task_out / "run"
    work.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(args.task1_zip)
    print(f"[task1] eksekusi {NOTEBOOK.name} ({len(nb.cells) - 1} sel) di {work} ...", flush=True)
    ep = ExecutePreprocessor(timeout=None, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(work)}})
    finally:
        nbformat.write(nb, args.task_out / "01_Task1_Classification_executed.ipynb")
    src = work / PRED_FILE
    if not src.exists():
        raise FileNotFoundError(f"notebook selesai tetapi {PRED_FILE} tidak dibuat")
    for d in (args.pred_dir, args.pred_dir / "prediksi"):
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / PRED_FILE)
    for f in ARTEFAK:
        if (work / f).exists():
            shutil.copy2(work / f, args.task_out / f)
    print(f"[task1] selesai -> {args.pred_dir / PRED_FILE}")
    return args.pred_dir / PRED_FILE


def build_parser():
    p = add_common_args(argparse.ArgumentParser(description="Task 1 - jalankan notebook Alief"))
    p.add_argument("--task1_zip", default=DEFAULT_ZIP,
                   help="zip data_uts_dl di Drive (default env UTS_ZIP_PATH atau MyDrive/data_uts_dl.zip)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
