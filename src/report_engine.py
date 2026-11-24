"""Minimal A4 typesetting on matplotlib: headings, wrapped paragraphs with inline math, display equations,
tables, figures, page numbers. Text is written as real text (pdf.fonttype 42), so the PDF is searchable."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                            "pdf.fonttype": 42, "ps.fonttype": 42})
W, H = 8.27, 11.69
ML, MR, MT, MB = 0.95, 0.95, 0.9, 0.85
TW = W - ML - MR


class Doc:
    def __init__(self, path, running_title, png_dir=None):
        self.png_dir = png_dir
        self.pdf = PdfPages(path)
        self.title = running_title
        self.page = 0
        self.fig = None
        self._new_page()

    # ---------- page handling
    def _new_page(self):
        if self.fig is not None:
            self._footer()
            self._save()
        self.fig = plt.figure(figsize=(W, H))
        self.canvas = FigureCanvasAgg(self.fig)
        self.r = self.canvas.get_renderer()
        self.page += 1
        self.y = H - MT

    def _footer(self):
        self.fig.text(ML / W, 0.45 / H, self.title, fontsize=7.5, color="0.45")
        self.fig.text((W - MR) / W, 0.45 / H, str(self.page), fontsize=7.5, color="0.45", ha="right")

    def need(self, h):
        if self.y - h < MB:
            self._new_page()

    def newpage(self):
        self._new_page()

    def _save(self):
        self.pdf.savefig(self.fig)
        if self.png_dir:
            self.fig.savefig(f"{self.png_dir}/page_{self.page:02d}.png", dpi=80)
        plt.close(self.fig)

    def close(self):
        self._footer()
        self._save()
        d = self.pdf.infodict()
        d["Title"] = self.title
        self.pdf.close()

    # ---------- measuring
    def _w(self, s, size, weight="normal", family=None):
        t = self.fig.text(0, 0, s, fontsize=size, weight=weight, family=family)
        bb = t.get_window_extent(self.r)
        t.remove()
        return bb.width / self.fig.dpi, bb.height / self.fig.dpi

    # ---------- blocks
    def heading(self, s, level=1):
        size = {0: 17, 1: 13, 2: 11}[level]
        gap = {0: 0.1, 1: 0.28, 2: 0.18}[level]
        self.need(gap + size / 72 * 1.6 + 0.5)
        self.y -= gap + size / 72 * 1.1
        self.fig.text(ML / W, self.y / H, s, fontsize=size, weight="bold")
        self.y -= size / 72 * 0.7

    def para(self, s, size=9.8, indent=0.0, color="k", style="normal", lh=1.5):
        tokens = re.findall(r"\$[^$]*\$|\S+", s)
        lines, cur = [], ""
        for tok in tokens:
            trial = tok if not cur else cur + " " + tok
            if self._w(trial, size)[0] > TW - indent and cur:
                lines.append(cur); cur = tok
            else:
                cur = trial
        if cur:
            lines.append(cur)
        step = size / 72 * lh
        for ln in lines:
            hasmath = "$" in ln
            self.need(step * (1.25 if hasmath else 1.0))
            self.y -= step * (1.1 if hasmath else 1.0)
            self.fig.text((ML + indent) / W, self.y / H, ln, fontsize=size, color=color, style=style)
        self.y -= step * 0.45

    def bullets(self, items, size=9.8):
        for it in items:
            y0 = self.y
            self.need(size / 72 * 1.6)
            self.fig.text((ML + 0.08) / W, (self.y - size / 72 * 1.5) / H, "•", fontsize=size)
            self.para(it, size=size, indent=0.28)
            self.y += size / 72 * 1.5 * 0.25

    def eq(self, latex, size=12, label=None):
        w, h = self._w(latex, size)
        self.need(h + 0.3)
        self.y -= h * 0.5 + 0.14
        self.fig.text(0.5, self.y / H, latex, fontsize=size, ha="center", va="center")
        if label:
            self.fig.text((W - MR) / W, self.y / H, f"({label})", fontsize=9, ha="right", va="center")
        self.y -= h * 0.5 + 0.14

    def table(self, rows, widths, size=8.4, header=True):
        widths = [w * TW / sum(widths) for w in widths]
        lh = size / 72 * 1.55
        # wrap each cell
        wrapped = []
        for r in rows:
            cells = []
            for c, w in zip(r, widths):
                toks, lines, cur = re.findall(r"\$[^$]*\$|\S+", str(c)), [], ""
                for tok in toks:
                    trial = tok if not cur else cur + " " + tok
                    if self._w(trial, size)[0] > w - 0.08 and cur:
                        lines.append(cur); cur = tok
                    else:
                        cur = trial
                lines.append(cur)
                cells.append(lines)
            wrapped.append(cells)
        self.need(lh * (1 + max(len(c) for c in wrapped[0])) + 0.1)
        top = self.y
        self.fig.add_artist(plt.Line2D([ML / W, (W - MR) / W], [self.y / H] * 2, color="k", lw=0.8))
        for k, cells in enumerate(wrapped):
            nl = max(len(c) for c in cells)
            if self.y - nl * lh - 0.06 < MB:
                self._new_page()
            x = ML
            for lines, w in zip(cells, widths):
                for j, ln in enumerate(lines):
                    self.fig.text((x + 0.04) / W, (self.y - (j + 0.78) * lh) / H, ln, fontsize=size,
                                  weight="bold" if (header and k == 0) else "normal")
                x += w
            self.y -= nl * lh + 0.05
            if header and k == 0 or k == len(wrapped) - 1:
                self.fig.add_artist(plt.Line2D([ML / W, (W - MR) / W], [self.y / H] * 2, color="k",
                                               lw=0.8 if k else 0.5))
        self.y -= 0.12

    def image(self, path, width=None, caption=None):
        img = plt.imread(path)
        width = width or TW
        h = width * img.shape[0] / img.shape[1]
        if h > H - MT - MB - 0.6:
            h = H - MT - MB - 0.6; width = h * img.shape[1] / img.shape[0]
        self.need(h + 0.45)
        ax = self.fig.add_axes([(W - width) / 2 / W, (self.y - h) / H, width / W, h / H])
        ax.imshow(img); ax.axis("off")
        self.y -= h + 0.05
        if caption:
            self.para(caption, size=8.4, color="0.25", style="italic")
        else:
            self.y -= 0.1

    def space(self, h=0.12):
        self.y -= h
