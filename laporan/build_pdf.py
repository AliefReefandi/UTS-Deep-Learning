"""laporan/laporan.md -> laporan.pdf (root repo) lewat HTML + Chrome/Edge headless.

  python laporan/build_pdf.py              # butuh: pip install markdown pypdf
  python laporan/build_pdf.py --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"

Sintaks tambahan di Markdown: ![keterangan](path){width=60%}. Beberapa gambar dalam satu paragraf
menjadi satu figur bernomor dengan keterangan dari gambar pertama. Komentar HTML (<!-- -->) tidak tampil.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "laporan" / "laporan.md"
HTML = ROOT / "laporan" / "laporan.html"
PDF = ROOT / "laporan.pdf"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "msedge"]

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9.4pt; line-height: 1.38;
       color: #1d1d1f; background: #fff; }
h1 { font-size: 15pt; margin: 0 0 6pt; border-bottom: 1.5px solid #2b5797; padding-bottom: 3pt; color: #1a3a6b; }
h1.pb { page-break-before: always; }
h2 { font-size: 11.2pt; margin: 10pt 0 4pt; color: #1a3a6b; }
p { margin: 3pt 0 5pt; }
ul, ol { margin: 2pt 0 5pt; padding-left: 16pt; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 4pt 0 7pt; font-size: 8.4pt; width: 100%; page-break-inside: avoid; }
th, td { border: 0.6px solid #c8ccd2; padding: 2.2pt 4pt; vertical-align: top; }
th { background: #eef2f7; text-align: left; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: #f3f4f6; padding: 0 2px; }
figure { margin: 5pt 0 8pt; text-align: center; page-break-inside: avoid; }
figure img { display: inline-block; vertical-align: top; margin: 0 0.4%; }
figcaption { font-size: 8.2pt; color: #4b5563; margin-top: 2pt; }
.title-sub { font-size: 10.5pt; color: #374151; margin-bottom: 6pt; }
"""


def fix_lists(md: str) -> str:
    """python-markdown: daftar butuh baris kosong sebelumnya dan indentasi 4 spasi untuk sub-daftar."""
    out, prev = [], ""
    for line in md.splitlines():
        m = re.match(r"^( {1,3})([-*]|\d+\.) ", line)
        if m:
            line = "    " + line.lstrip(" ")
        is_item = re.match(r"^\s*([-*]|\d+\.) ", line) is not None
        prev_item = re.match(r"^\s*([-*]|\d+\.) ", prev) is not None
        cont = prev.startswith("  ") and not prev.strip().startswith("|")
        if is_item and prev.strip() and not prev_item and not cont:
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def preprocess(md: str) -> str:
    md = fix_lists(md)
    # ![alt](path dengan spasi){width=NN%} -> attr_list
    def img(m):
        alt, path, w = m.group(1), m.group(2).replace(" ", "%20"), m.group(3)
        attr = f'{{: style="width:{w}%" }}' if w else ""
        return f"![{alt}]({path}){attr}"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)(?:\{width=(\d+)%\})?", img, md)


def figures(html: str) -> str:
    n = [0]

    def fig(m):
        inner = m.group(1)
        alts = re.findall(r'alt="([^"]*)"', inner)
        cap = next((a for a in alts if a), "")
        n[0] += 1
        return f"<figure>{inner}<figcaption>Gambar {n[0]}. {cap}</figcaption></figure>"
    return re.sub(r"<p>((?:\s*<img [^>]*>\s*)+)</p>", fig, html)


def build_html() -> Path:
    import markdown
    md = preprocess(SRC.read_text(encoding="utf-8"))
    body = markdown.markdown(md, extensions=["tables", "attr_list", "sane_lists"])
    body = figures(body)
    body = body.replace("<h1>Task", '<h1 class="pb">Task')
    body = body.replace("<p><strong>Klasifikasi", '<p class="title-sub"><strong>Klasifikasi', 1)
    HTML.write_text(f'<!doctype html><html lang="id"><head><meta charset="utf-8">'
                    f"<title>Laporan UTS Deep Learning</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    return HTML


def find_browser(arg):
    for b in ([arg] if arg else []) + BROWSERS:
        if b and (Path(b).exists() or shutil.which(b)):
            return b
    sys.exit("Chrome/Edge tidak ditemukan; berikan --browser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()
    html = build_html()
    subprocess.run([find_browser(args.browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.as_uri()], check=True, capture_output=True)
    from pypdf import PdfReader
    print(f"{PDF} : {len(PdfReader(PDF).pages)} halaman")


if __name__ == "__main__":
    main()
