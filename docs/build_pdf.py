"""Render docs/blueprint.md + docs/simulation.md into one PDF.

    python3 docs/build_pdf.py            ->  docs/YFCE-blueprint.pdf

The markdown is canonical; this only typesets it. Nothing is installed:
markdown is parsed with `markdown_it` (already on the system Python), the
page is printed by the Windows Chrome that WSL can already reach, and page
numbers are stamped by the TeX Live that is already here (pdfpages). The
two mermaid diagrams in blueprint.md are replaced by hand-drawn SVG, since
no mermaid renderer is on this machine.

Requirements as found on this box: WSL2, Chrome at
`/mnt/c/Program Files/Google/Chrome/Application/chrome.exe`, `pdflatex`.
Set CHROME= to another browser (Edge works: same flags) if needed.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "YFCE-blueprint.pdf"
CHROME = os.environ.get(
    "CHROME", "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
)
DATE = "2026-08-17"

# A scratch directory that both WSL and Windows can see. Chrome runs on the
# Windows side, so its input and output paths must be Windows paths.
WIN_TMP = Path("/mnt/c/Users") / os.environ.get("WINUSER", "ROG") / "AppData/Local/Temp/yfce-pdf"


def winpath(p: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()


# --- diagrams -----------------------------------------------------------------
# Hand-drawn equivalents of the two mermaid blocks in blueprint.md.

_BOX = 'fill="#fbfaf7" stroke="#3a3f46" stroke-width="1.1" rx="3"'
_TXT = 'font-family="Segoe UI, Arial, sans-serif" text-anchor="middle" fill="#1c2024"'
_LBL = 'font-family="Segoe UI, Arial, sans-serif" font-size="9.5" fill="#7a4a22" text-anchor="middle"'
_LINE = 'stroke="#3a3f46" stroke-width="1.1" fill="none" marker-end="url(#arr)"'
_ARR = (
    '<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
    'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 z" '
    'fill="#3a3f46"/></marker><marker id="arr2" markerWidth="9" markerHeight="9" '
    'refX="1" refY="4.5" orient="auto" markerUnits="userSpaceOnUse">'
    '<path d="M9,0 L0,4.5 L9,9 z" fill="#3a3f46"/></marker></defs>'
)


def _box(x, y, w, h, lines, bold_first=True, double=False, dashed=False):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {_BOX}{" stroke-dasharray=\"4 3\"" if dashed else ""}/>'
    if double:
        s += f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" fill="none" stroke="#3a3f46" stroke-width="0.8" rx="2"/>'
    n = len(lines)
    lh = 13
    y0 = y + h / 2 - (n - 1) * lh / 2 + 4
    for i, ln in enumerate(lines):
        w_ = 'font-weight="600" font-size="11"' if (i == 0 and bold_first) else 'font-size="9.8"'
        s += f'<text x="{x + w/2}" y="{y0 + i*lh:.1f}" {_TXT} {w_}>{html.escape(ln)}</text>'
    return s


def _path(points, both=False, label=None, lx=None, ly=None):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    m = ' marker-start="url(#arr2)"' if both else ""
    s = f'<path d="{d}" {_LINE}{m}/>'
    if label:
        s += f'<text x="{lx}" y="{ly}" {_LBL}>{html.escape(label)}</text>'
    return s


def svg_datapath() -> str:
    W, H = 760, 235
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Data path">', _ARR]
    # row A
    s.append(_box(10, 22, 125, 50, ["NVMe flash", "coded blocks, 64 KiB,", "page-aligned"]))
    s.append(_box(180, 22, 130, 50, ["Residency manager", "CPU cores, lmz lineage"]))
    s.append(_box(350, 22, 90, 50, ["DMA engine"]))
    s.append(_box(470, 22, 130, 50, ["Memory controller", "+ PHY"]))
    s.append(_box(630, 22, 122, 50, ["DDR5 DIMMs, 8×64-bit", "coded weights", "+ KV cache"], double=True))
    s.append(_path([(135, 47), (180, 47)], label="demand page", lx=157, ly=40))
    s.append(_path([(310, 47), (350, 47)], label="descriptor ring", lx=330, ly=16))
    s.append(_path([(440, 47), (470, 47)]))
    s.append(_path([(600, 47), (630, 47)], both=True))
    # row B
    s.append(_box(300, 140, 200, 78, ["Decode block", "unpack + dequant: always", "entropy stage: Fork A", "lattice / VQ stage: Fork A′"]))
    s.append(_box(535, 150, 90, 58, ["Staging SRAM"]))
    s.append(_box(655, 150, 97, 58, ["NPU", "GEMV, dequant", "in the MAC path"]))
    # MC -> decode (coded stream), NPU -> MC (KV, activations)
    s.append(_path([(505, 72), (505, 105), (400, 105), (400, 140)], label="coded stream", lx=452, ly=99))
    s.append(_path([(500, 179), (535, 179)], label="quantised operands + scales", lx=520, ly=132))
    s.append(_path([(625, 179), (655, 179)]))
    s.append(_path([(703, 150), (703, 105), (565, 105), (565, 72)], label="KV, activations", lx=640, ly=99))
    s.append("</svg>")
    return "".join(s)


def _label(x, y, text, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="9.5" '
            f'fill="#7a4a22" text-anchor="{anchor}">{html.escape(text)}</text>')


def svg_aistack() -> str:
    W, H = 760, 350
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AI PC layers">', _ARR]
    s.append(_box(10, 8, 740, 44, ["The person", "voice in and out  ·  a glanceable screen  ·  a fixed set of keys and touch  ·  camera on demand"]))
    s.append(_box(10, 84, 740, 72, [
        "The operating system = the model set, resident in DRAM",
        "kernel model: policy, conversation, tools  ·  reflex model: UI, typing, draft tokens",
        "perception: vision, speech-to-text  ·  voice: TTS  ·  memory: embeddings + index",
        "apps = adapters + skills + grants, demand-paged as blocks"]))
    s.append(_box(10, 190, 740, 72, [
        "Runtime — mechanism only, no policy (firmware, not an OS)",
        "boot + PHY training  ·  residency manager  ·  descriptor ring, NPU scheduler, grammar-constrained sampler",
        "event bus, widgets, compositor  ·  capability guard + trusted surface",
        "drivers: NVMe, USB, display, audio, network"]))
    s.append(_box(10, 294, 740, 44, [
        "YFCE-1 hardware",
        "SoC (PHY, DMA, decode, staging, NPU, CPU cores)  ·  8 × DDR5 DIMMs  ·  NVMe  ·  display, USB, audio, network"]))
    s.append(_path([(70, 52), (70, 84)], both=True)); s.append(_label(84, 72, "speech, the glanceable screen, confirm keys"))
    s.append(_path([(70, 156), (70, 190)], both=True)); s.append(_label(84, 177, "tokens: events in; UI language, tool calls, speech out"))
    s.append(_path([(70, 262), (70, 294)], both=True)); s.append(_label(84, 282, "descriptors, DMA, interrupts, pixels, packets"))
    s.append("</svg>")
    return "".join(s)


def svg_phasemap() -> str:
    W, H = 760, 560
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Phase map">', _ARR]
    c1, c2, c3, c4, bw = 10, 200, 390, 580, 170
    mid = 295  # centre column, x of a 170-wide box centred on the page
    # rows
    s.append(_box(mid, 8, bw, 40, ["S0 environment, isolated"]))
    s.append(_box(c1, 80, bw, 74, ["S1 representation", "E1b 2-bit", "E1c cheap-decode incumbents", "E1d downstream"]))
    s.append(_box(c2, 80, bw, 50, ["S2 format v0", "+ golden model"]))
    s.append(_box(c3, 80, bw, 50, ["S5 memory-system", "+ pipeline simulation"]))
    s.append(_box(c4, 80, bw, 50, ["S9 AI-OS contracts", "+ prototype on this PC"]))
    cx, cy = c1 + bw / 2, 215
    s.append(f'<polygon points="{cx},{cy-30} {cx+85},{cy} {cx},{cy+30} {cx-85},{cy}" {_BOX}/>')
    s.append(f'<text x="{cx}" y="{cy-2}" {_TXT} font-weight="600" font-size="11">G1</text>')
    s.append(f'<text x="{cx}" y="{cy+12}" {_TXT} font-size="9.8">fork A / A′ / B</text>')
    s.append(_box(c2, 190, bw, 50, ["S3 decode block + DMA RTL,", "verification"]))
    s.append(_box(c3, 190, bw, 50, ["S7 E2-lite", "on this PC"]))
    s.append(_box(c4, 184, bw, 62, ["S10 bare-metal proof", "boots into the model, no OS", "(optional)"]))
    s.append(_box(c2, 275, bw, 50, ["S4 synthesis, timing,", "area, power"]))
    s.append(_box(c2, 357, bw, 50, ["S6 recalibrate", "model/*.py and docs/"]))
    s.append(_box(150, 435, 460, 40, ["simulation complete — the blueprint stops here"], double=True))
    s.append(_box(c1, 502, 340, 42, ["on-board: T1 Strix Halo · T2 DIMM server (bare metal)", "· T3 FPGA · T4 human I/O — BOMs in §8"], bold_first=False))
    s.append(_box(430, 502, 320, 42, ["on-chip: C1 MPW test chip", "BOM in §8"], bold_first=False))
    # edges from S0
    s.append(_path([(mid + 20, 48), (mid + 20, 60), (c1 + bw / 2, 60), (c1 + bw / 2, 80)]))        # S0 -> S1
    s.append(_path([(mid + 40, 48), (mid + 40, 68), (c2 + bw / 2, 68), (c2 + bw / 2, 80)]))        # S0 -> S2
    s.append(_path([(mid + bw - 40, 48), (mid + bw - 40, 68), (c3 + bw / 2, 68), (c3 + bw / 2, 80)]))  # S0 -> S5
    s.append(_path([(mid + bw - 20, 48), (mid + bw - 20, 60), (c4 + bw / 2, 60), (c4 + bw / 2, 80)]))  # S0 -> S9
    # main chains
    s.append(_path([(c1 + bw / 2, 154), (c1 + bw / 2, cy - 30)]))                                 # S1 -> fork
    s.append(_path([(cx + 85, cy), (c2, cy)]))                                                    # fork -> S3
    s.append(_path([(c2 + bw / 2, 130), (c2 + bw / 2, 190)]))                                     # S2 -> S3
    s.append(_path([(c2 + bw / 2, 240), (c2 + bw / 2, 275)]))                                     # S3 -> S4
    s.append(_path([(c2 + bw / 2, 325), (c2 + bw / 2, 357)]))                                     # S4 -> S6
    s.append(_path([(c2 + bw - 20, 130), (c2 + bw - 20, 168), (c3 + bw / 2, 168), (c3 + bw / 2, 190)]))  # S2 -> S7
    s.append(_path([(c4 + bw / 2, 130), (c4 + bw / 2, 184)]))                                     # S9 -> S10
    # into S6
    s.append(_path([(c1 + 12, 154), (c1 + 12, 382), (c2, 382)]))                                  # S1 -> S6
    s.append(_path([(c3 + bw / 2, 240), (c3 + bw / 2, 382), (c2 + bw, 382)]))                     # S7 -> S6
    s.append(_path([(c3 + bw + 10, 130), (c3 + bw + 10, 392), (c2 + bw, 392)]))                   # S5 -> S6
    s.append(_path([(c4 + bw + 6, 130), (c4 + bw + 6, 400), (c2 + bw, 400)]))                     # S9 -> S6
    # exit
    s.append(_path([(c2 + bw / 2, 407), (c2 + bw / 2, 435)]))                                     # S6 -> stop
    s.append(_path([(210, 475), (210, 488), (c1 + 170, 488), (c1 + 170, 502)]))                   # stop -> T
    s.append(_path([(550, 475), (550, 488), (590, 488), (590, 502)]))                              # stop -> C
    s.append("</svg>")
    return "".join(s)


SVGS = {"datapath": svg_datapath, "aistack": svg_aistack, "phasemap": svg_phasemap}

# --- markdown -> html ---------------------------------------------------------

md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")


def slug(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return t or "s"


def render_part(path: Path, part_id: str, eyebrow: str) -> tuple[str, list[tuple[str, str]]]:
    src = path.read_text(encoding="utf-8")
    # swap mermaid fences for markers, in order of appearance
    names = iter(["datapath", "aistack", "phasemap"])
    src = re.sub(r"```mermaid\n.*?```", lambda m: f"<!--SVG:{next(names)}-->", src, flags=re.S)
    tokens = md.parse(src)
    toc: list[tuple[str, str]] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            text = tokens[i + 1].content
            hid = f"{part_id}-{slug(text)}"
            tok.attrSet("id", hid)
            if tok.tag == "h2":
                toc.append((hid, text))
            if tok.tag == "h1":
                # part eyebrow rides on the H1
                tok.attrSet("class", "part-title")
    body = md.renderer.render(tokens, md.options, {})
    for name, fn in SVGS.items():
        body = body.replace(f"<!--SVG:{name}-->", f'<figure>{fn()}</figure>')
    body = body.replace('<h1 id=', f'<p class="eyebrow">{eyebrow}</p><h1 id=', 1)
    return body, toc


CSS = """
@page { size: A4; margin: 19mm 18mm 21mm 18mm; }
html { font-size: 10.4pt; }
body { font-family: Georgia, "Times New Roman", serif; color: #1c2024; line-height: 1.46;
       margin: 0; background: #fff; }
h1, h2, h3, h4 { font-family: "Segoe UI", "Segoe UI Variable Text", Arial, sans-serif; color: #1c2024;
                 text-wrap: balance; }
h1 { font-size: 24pt; font-weight: 600; margin: 0 0 8pt; letter-spacing: -0.012em; line-height: 1.15; }
h2 { font-size: 14.5pt; font-weight: 600; margin: 20pt 0 8pt; padding-top: 6pt;
     border-top: 1.4px solid #b06a3a; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11.6pt; font-weight: 600; margin: 15pt 0 5pt; break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
p.eyebrow { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.5pt; letter-spacing: 0.14em;
            text-transform: uppercase; color: #7a4a22; margin: 0 0 4pt; }
code, pre { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.9pt; }
code { background: #f3f0ea; padding: 0 3px; border-radius: 2px; }
pre { background: #f7f5f0; border-left: 3px solid #d9cfc2; padding: 7pt 10pt; overflow-x: auto;
      white-space: pre; line-height: 1.35; margin: 6pt 0 10pt; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt;
        font-family: "Segoe UI", Arial, sans-serif; font-size: 8.7pt; line-height: 1.32;
        font-variant-numeric: tabular-nums; }
thead { display: table-header-group; }
th { text-align: left; font-weight: 600; background: #f0ebe3; border-bottom: 1.4px solid #b06a3a;
     padding: 4pt 6pt; vertical-align: bottom; }
td { border-bottom: 0.6px solid #d8d2c8; padding: 4pt 6pt; vertical-align: top; }
tr { break-inside: avoid; page-break-inside: avoid; }
ul, ol { margin: 0 0 8pt 1.25em; padding: 0; }
li { margin-bottom: 3pt; }
a { color: #7a4a22; text-decoration: none; }
strong { font-weight: 700; }
hr { border: 0; border-top: 0.8px solid #d8d2c8; margin: 12pt 0; }
figure { margin: 8pt 0 14pt; break-inside: avoid; page-break-inside: avoid; }
figure svg { width: 100%; height: auto; display: block; }
.part { break-before: page; page-break-before: always; }
.title { height: 245mm; display: flex; flex-direction: column; justify-content: flex-start;
         page-break-after: always; break-after: page; }
.title h1 { font-size: 40pt; margin: 22mm 0 6mm; }
.title .lede { font-size: 13.5pt; line-height: 1.4; max-width: 34em; margin-bottom: 10mm; }
.title .meta { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; color: #4d5158; line-height: 1.55; }
.toc { margin-top: 12mm; display: grid; grid-template-columns: 1fr 1fr; gap: 0 12mm; font-family: "Segoe UI", Arial, sans-serif; font-size: 9.4pt; }
.toc h4 { margin: 0 0 4pt; font-size: 9pt; letter-spacing: 0.1em; text-transform: uppercase; color: #7a4a22; break-after: avoid; }
.toc ul { list-style: none; margin: 0 0 8pt; padding: 0; }
.toc li { margin: 0 0 2.5pt; break-inside: avoid; }
.toc a { color: #1c2024; }
.foot { position: fixed; bottom: 0; }
"""


def main() -> None:
    p1, toc1 = render_part(DOCS / "blueprint.md", "p1", "Part I")
    p2, toc2 = render_part(DOCS / "simulation.md", "p2", "Part II")

    def toc_html(title, items):
        lis = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in items)
        return f"<div><h4>{title}</h4><ul>{lis}</ul></div>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>YFCE Blueprint</title><style>{CSS}</style></head><body>
<section class="title">
  <p class="eyebrow">YFCE · design stage · {DATE}</p>
  <h1>YFCE Blueprint</h1>
  <p class="lede">A memory-first inference appliance on mature-node silicon: chip, board and
  filesystem designed as one system, targeting the cheapest hardware that runs a large model at
  usable speed.</p>
  <p class="meta">Part I — Blueprint: the machine in two variants — the headless appliance and
  the <b>AI PC</b>, where the model is the operating system and nothing runs Windows or Linux —
  its modules and interfaces, the decisions proposed, the gates and forks, the phase map, and
  bills of materials with next-plans for the board, FPGA and test-chip steps.<br>Part II —
  Simulation plan: the PC-side campaign S0–S10, sized to the machine the repository lives on.
  <br><br>The plan is detailed through the simulation campaign and stops there. Anything that
  needs hardware bought is listed, not designed.<br><br>Typeset from
  <code>docs/blueprint.md</code> and <code>docs/simulation.md</code>; the markdown is canonical.</p>
  <div class="toc">{toc_html("Part I — Blueprint", toc1)}{toc_html("Part II — Simulation plan", toc2)}</div>
</section>
<section class="part">{p1}</section>
<section class="part">{p2}</section>
</body></html>"""

    WIN_TMP.mkdir(parents=True, exist_ok=True)
    html_path = WIN_TMP / "yfce-blueprint.html"
    content_pdf = WIN_TMP / "content.pdf"
    html_path.write_text(page, encoding="utf-8")
    if content_pdf.exists():
        content_pdf.unlink()

    subprocess.run(
        [
            CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={winpath(WIN_TMP / 'profile')}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={winpath(content_pdf)}",
            "file:///" + winpath(html_path).replace("\\", "/"),
        ],
        check=False, timeout=180,
    )
    # chrome.exe launched through WSL interop returns before the renderer has
    # finished writing; wait for the file to appear and its size to settle.
    import time
    last, stable = -1, 0
    for _ in range(240):
        time.sleep(0.5)
        if content_pdf.exists():
            sz = content_pdf.stat().st_size
            stable = stable + 1 if sz == last and sz > 0 else 0
            last = sz
            if stable >= 4:
                break
    if not content_pdf.exists() or content_pdf.stat().st_size == 0:
        sys.exit("Chrome produced no PDF")

    # Page numbers: TeX Live's pdfpages overlays a footer on every page after the title.
    tex = r"""\documentclass{article}
\usepackage[a4paper,margin=0pt]{geometry}
\usepackage{pdfpages}
\usepackage{lastpage}
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\pagestyle{empty}
\setlength{\unitlength}{1mm}
\newcommand{\stamp}{\begin{picture}(0,0)\put(105,-286.5){\makebox(0,0){\footnotesize\sffamily\color[gray]{0.35}YFCE Blueprint\quad\textendash\quad\thepage\ / \pageref{LastPage}}}\end{picture}}
\begin{document}
\includepdf[pages=1,pagecommand={}]{content.pdf}
\setcounter{page}{1}
\includepdf[pages=2-,pagecommand={\stamp}]{content.pdf}
\end{document}
"""
    (WIN_TMP / "stamp.tex").write_text(tex)
    for _ in range(2):  # lastpage needs two passes
        subprocess.run(
            ["pdflatex", "-interaction=batchmode", "-halt-on-error", "stamp.tex"],
            cwd=WIN_TMP, check=True, stdout=subprocess.DEVNULL,
        )
    shutil.copyfile(WIN_TMP / "stamp.pdf", OUT)
    # leave no browser profile behind in the Windows temp dir
    shutil.rmtree(WIN_TMP / "profile", ignore_errors=True)
    for f in WIN_TMP.glob("stamp.*"):
        f.unlink(missing_ok=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
