"""Render PRESENTATION_GUIDE.md to a print-ready PDF.

Uses python-markdown to render, embeds print-friendly CSS, and shells out to
Chrome (headless) to produce the PDF — no LaTeX or extra binaries needed.

Usage:
  python build_guide_pdf.py                  # writes PRESENTATION_GUIDE.pdf
  python build_guide_pdf.py --md OTHER.md    # render a different markdown file
  python build_guide_pdf.py --keep-html      # also keep the intermediate HTML
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

BASE = Path(__file__).resolve().parent

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("microsoft-edge") or "",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    sys.exit(
        "No Chrome/Chromium/Edge found. Install one, or convert the markdown "
        "with another tool (pandoc, weasyprint)."
    )


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-right { content: counter(page); font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #968d78; }
}
* { box-sizing: border-box; }
body {
  font-family: 'Charter', 'Iowan Old Style', 'Palatino Linotype', 'Georgia', serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1c1a16;
  max-width: 720px;
  margin: 0 auto;
  -webkit-font-smoothing: antialiased;
  hyphens: auto;
}
h1, h2, h3, h4 { font-family: 'Iowan Old Style', 'Palatino Linotype', 'Georgia', serif; }
h1 {
  font-size: 28pt;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 18pt;
  padding-bottom: 8pt;
  border-bottom: 1.5pt solid #b58b25;
  page-break-after: avoid;
}
h2 {
  font-size: 17pt;
  font-weight: 600;
  margin: 30pt 0 10pt;
  padding-bottom: 4pt;
  border-bottom: 0.5pt solid #d0c9b6;
  page-break-after: avoid;
}
h3 {
  font-size: 12.5pt;
  font-weight: 600;
  margin: 18pt 0 6pt;
  page-break-after: avoid;
}
h4 {
  font-size: 11pt;
  font-weight: 600;
  margin: 14pt 0 4pt;
  font-style: italic;
}
p { margin: 0 0 8pt; }
ul, ol { margin: 4pt 0 8pt; padding-left: 22pt; }
li { margin: 2pt 0; }
strong { font-weight: 600; color: #0e0d0b; }
em { font-style: italic; color: #5a5347; }
code {
  font-family: 'JetBrains Mono', 'Menlo', 'Monaco', monospace;
  font-size: 0.88em;
  background: #f4f1ea;
  padding: 1.5pt 4pt;
  border-radius: 2pt;
  color: #6e5713;
  border: 0.4pt solid #e3dccb;
}
pre {
  background: #15130f;
  color: #efe9d9;
  padding: 10pt 12pt;
  border-radius: 3pt;
  font-family: 'JetBrains Mono', 'Menlo', monospace;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  word-break: break-all;
  page-break-inside: avoid;
  margin: 8pt 0 12pt;
  border-left: 2.5pt solid #f4b942;
}
pre code {
  background: transparent;
  color: inherit;
  padding: 0;
  border: 0;
  font-size: inherit;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10pt 0 14pt;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 0.4pt solid #d0c9b6;
  padding: 5pt 8pt;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f4f1ea;
  font-weight: 600;
  font-family: 'Iowan Old Style', serif;
}
blockquote {
  border-left: 2pt solid #b58b25;
  margin: 10pt 0;
  padding: 4pt 14pt;
  color: #5a5347;
  font-style: italic;
}
hr {
  border: 0;
  border-top: 0.5pt solid #d0c9b6;
  margin: 22pt 0;
}
a { color: #6e5713; text-decoration: none; border-bottom: 0.4pt dotted #b58b25; }
input[type="checkbox"] { transform: scale(1.05); margin-right: 4pt; }

/* Section number flourish for top-level sections */
h2 { color: #1c1a16; }
h2::before {
  content: "";
  display: inline-block;
  width: 18pt;
  height: 1pt;
  background: #b58b25;
  vertical-align: middle;
  margin-right: 8pt;
  margin-bottom: 4pt;
}

.cover {
  page-break-after: always;
  height: 250mm;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 30mm 0 10mm;
}
.cover-eyebrow {
  font-family: 'JetBrains Mono', 'Menlo', monospace;
  font-size: 9pt;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #b58b25;
}
.cover-title {
  font-family: 'Iowan Old Style', 'Palatino Linotype', serif;
  font-size: 64pt;
  font-weight: 400;
  line-height: 0.95;
  letter-spacing: -0.02em;
  margin: 12pt 0 18pt;
}
.cover-title em {
  font-style: italic;
  color: #b58b25;
}
.cover-deck {
  font-family: 'Iowan Old Style', serif;
  font-size: 14pt;
  font-style: italic;
  color: #5a5347;
  max-width: 26em;
  line-height: 1.4;
  margin: 0;
}
.cover-meta {
  font-family: 'JetBrains Mono', 'Menlo', monospace;
  font-size: 9.5pt;
  letter-spacing: 0.04em;
  color: #5a5347;
  border-top: 0.5pt solid #d0c9b6;
  padding-top: 10pt;
  margin-top: auto;
  display: flex;
  justify-content: space-between;
}
"""


COVER_HTML = """
<div class="cover">
  <div>
    <div class="cover-eyebrow">TrueVoice / Forensic Audio Bureau</div>
    <h1 class="cover-title">Project &amp;<br><em>presentation guide.</em></h1>
    <p class="cover-deck">A complete handover for running, demoing, and defending the audio deepfake detection system &mdash; written for a Windows machine and a viva-style presentation.</p>
  </div>
  <div class="cover-meta">
    <span>Vol.01 / Issue №04</span>
    <span>Build 0.4.0</span>
    <span>For research &amp; educational use</span>
  </div>
</div>
"""


def render(md_path: Path, html_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md,
        extensions=["fenced_code", "tables", "sane_lists", "toc"],
        output_format="html5",
    )
    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>TrueVoice — Presentation Guide</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{COVER_HTML}\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )
    html_path.write_text(html, encoding="utf-8")


def chrome_to_pdf(chrome: str, html_path: Path, pdf_path: Path) -> None:
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        sys.exit(f"Chrome exited {res.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(BASE / "PRESENTATION_GUIDE.md"))
    ap.add_argument("--out", default=str(BASE / "PRESENTATION_GUIDE.pdf"))
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        sys.exit(f"Markdown not found: {md_path}")
    html_path = md_path.with_suffix(".html")
    pdf_path = Path(args.out)

    chrome = find_chrome()
    print(f"Rendering {md_path.name}…")
    render(md_path, html_path)
    print(f"Printing PDF via {Path(chrome).name}…")
    chrome_to_pdf(chrome, html_path, pdf_path)

    if not args.keep_html:
        try: html_path.unlink()
        except OSError: pass

    print(f"Wrote {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
