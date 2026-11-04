# -*- coding: utf-8 -*-
"""Self-contained HTML reports with inline SVG charts (no internet, no JS libraries).
Every mark carries an SVG <title>, so hovering shows the exact value."""
import html
import json
import math
from pathlib import Path

from amutil import t, log, emit_report, reports_dir, stamp

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CSS = """
.amr{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--accent:#2a78d6;--hl:#eb6834;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg);
line-height:1.45;font-size:14px;padding:16px;max-width:1180px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .amr{color-scheme:dark;--bg:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,.10);--accent:#3987e5;--hl:#d95926;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"] .amr,.dark .amr{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--accent:#3987e5;--hl:#d95926;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
.amr h1{font-size:22px;margin:4px 0 2px}.amr h2{font-size:17px;margin:26px 0 6px}
.amr h3{font-size:14px;margin:14px 0 4px;color:var(--ink2)}
.amr .sub{color:var(--ink2);margin:0 0 10px}.amr .note{color:var(--ink2);font-size:13px;margin:4px 0 10px}
.amr .card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px;margin:10px 0;overflow-x:auto}
.amr table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
.amr th,.amr td{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
.amr th:first-child,.amr td:first-child{text-align:left}.amr th{color:var(--ink2);font-weight:600}
.amr tr.best td{font-weight:700}
.amr .badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;margin-right:6px;border:1px solid var(--ring)}
.amr .b-good::before{content:"● ";color:var(--good)}.amr .b-warn::before{content:"▲ ";color:var(--warn)}
.amr .b-bad::before{content:"■ ";color:var(--bad)}.amr .b-info::before{content:"ℹ ";color:var(--accent)}
.amr svg{display:block;max-width:100%;height:auto}.amr svg text{fill:var(--ink2);font-size:11px;font-family:inherit}
.amr svg .ttl{fill:var(--ink);font-size:12px;font-weight:600}
.amr .legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.amr .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.amr pre{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:10px;overflow-x:auto;font-size:12px;white-space:pre-wrap}
.amr details summary{cursor:pointer;color:var(--ink2);margin:6px 0}
.amr rect.m:hover,.amr circle.m:hover{stroke:var(--ink);stroke-width:1.5}
"""


def esc(x):
    return html.escape(str(x))


def fmt(v, f="{:.3f}"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        s = f.format(v)
        return s[1:] if s.startswith("-") and not s.strip("-0.,%") else s   # "-0.00" -> "0.00"
    except Exception:
        return str(v)


def _nice_max(v):
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * e:
            return m * e
    return 10 * e


class Report:
    def __init__(self, kind, title, subtitle=""):
        self.kind = kind
        self.title = title
        self.parts = []
        self.text = [title, "=" * len(title)]
        if subtitle:
            self.parts.append(f'<p class="sub">{esc(subtitle)}</p>')
            self.text.append(subtitle)
        self.data = {"kind": kind, "title": title}

    # --- text
    def h2(self, s):
        self.parts.append(f"<h2>{esc(s)}</h2>")
        self.text += ["", s, "-" * len(s)]

    def h3(self, s):
        self.parts.append(f"<h3>{esc(s)}</h3>")
        self.text += ["", s]

    def note(self, s):
        self.parts.append(f'<p class="note">{esc(s)}</p>')
        self.text.append("  " + s)

    def badge(self, level, s):
        """level: good / warn / bad / info"""
        self.parts.append(f'<div><span class="badge b-{level}">{esc(s)}</span></div>')
        self.text.append({"good": "[OK] ", "warn": "[!] ", "bad": "[X] ", "info": "[i] "}[level] + s)

    def pre(self, s):
        self.parts.append(f"<pre>{esc(s)}</pre>")
        self.text.append(s)

    def details(self, summary, inner_html):
        self.parts.append(f"<details><summary>{esc(summary)}</summary>{inner_html}</details>")

    # --- table
    def table(self, headers, rows, best_row=None, fmts=None):
        fmts = fmts or {}
        h = "".join(f"<th>{esc(x)}</th>" for x in headers)
        body = []
        for i, r in enumerate(rows):
            cells = "".join(f"<td>{esc(fmt(v, fmts.get(j, '{:.3f}')))}</td>" for j, v in enumerate(r))
            body.append(f'<tr class="{"best" if i == best_row else ""}">{cells}</tr>')
        self.parts.append(f'<div class="card"><table><tr>{h}</tr>{"".join(body)}</table></div>')
        w = [max(len(str(headers[j])), *(len(fmt(r[j], fmts.get(j, "{:.3f}"))) for r in rows)) if rows else len(str(headers[j]))
             for j in range(len(headers))]
        self.text.append("  " + "  ".join(str(headers[j]).ljust(w[j]) for j in range(len(headers))))
        for r in rows:
            self.text.append("  " + "  ".join(fmt(r[j], fmts.get(j, "{:.3f}")).ljust(w[j]) for j in range(len(r))))

    # --- charts
    def bars(self, labels, values, title="", fmt_s="{:.3f}", highlight=None, unit="", hl_label=None, horizontal=None,
             width=None):
        """Single-series bar chart. highlight: set of indexes drawn in the second color."""
        n = len(values)
        if not n:
            return
        highlight = set(highlight or ())
        horizontal = horizontal if horizontal is not None else (n <= 14 and max(len(str(x)) for x in labels) > 6)
        vmax = _nice_max(max([v for v in values if v is not None and v == v] + [0]))
        marks = []
        if horizontal:
            lw = min(260, 7 * max(len(str(x)) for x in labels) + 10)
            W = width or 760
            bh, gap = 18, 6
            H = 34 + n * (bh + gap) + 24
            pw = W - lw - 70
            marks.append(f'<text class="ttl" x="0" y="14">{esc(title)}</text>')
            for k in range(5):
                x = lw + pw * k / 4
                marks.append(f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{H - 22}" stroke="var(--grid)"/>'
                             f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle">{fmt(vmax * k / 4, fmt_s)}</text>')
            for i, (lab, v) in enumerate(zip(labels, values)):
                y = 30 + i * (bh + gap)
                vv = 0 if v is None or v != v else v
                w = max(0.0, pw * vv / vmax)
                col = "var(--hl)" if i in highlight else "var(--s1)"
                marks.append(f'<text x="{lw - 8}" y="{y + bh * 0.72:.1f}" text-anchor="end">{esc(lab)}</text>')
                marks.append(f'<rect class="m" x="{lw}" y="{y}" width="{max(w, 1):.1f}" height="{bh}" rx="4" fill="{col}">'
                             f'<title>{esc(lab)}: {fmt(v, fmt_s)}{unit}</title></rect>')
                marks.append(f'<text x="{lw + w + 6:.1f}" y="{y + bh * 0.72:.1f}">{fmt(v, fmt_s)}{esc(unit)}</text>')
        else:
            W = width or max(560, min(1150, 36 + n * 24))
            H = 230
            pl, pr, pt, pb = 46, 10, 28, 34
            pw, ph = W - pl - pr, H - pt - pb
            marks.append(f'<text class="ttl" x="0" y="14">{esc(title)}</text>')
            for k in range(5):
                y = pt + ph - ph * k / 4
                marks.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{W - pr}" y2="{y:.1f}" stroke="var(--grid)"/>'
                             f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(vmax * k / 4, fmt_s)}</text>')
            slot = pw / n
            bw = max(2.0, min(28.0, slot - 2))
            step = max(1, int(math.ceil(n / 40)))
            for i, (lab, v) in enumerate(zip(labels, values)):
                vv = 0 if v is None or v != v else v
                h = ph * max(0.0, vv) / vmax
                x = pl + i * slot + (slot - bw) / 2
                col = "var(--hl)" if i in highlight else "var(--s1)"
                marks.append(f'<rect class="m" x="{x:.1f}" y="{pt + ph - h:.1f}" width="{bw:.1f}" height="{max(h, 0.5):.1f}" '
                             f'rx="{min(4, bw / 3):.1f}" fill="{col}"><title>{esc(lab)}: {fmt(v, fmt_s)}{unit}</title></rect>')
                if i % step == 0:
                    marks.append(f'<text x="{x + bw / 2:.1f}" y="{H - pb + 14}" text-anchor="middle">{esc(lab)}</text>')
            marks.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{W - pr}" y2="{pt + ph}" stroke="var(--axis)"/>')
        leg = ""
        if highlight and hl_label:
            leg = (f'<div class="legend"><span><i style="background:var(--s1)"></i>{esc(hl_label[0])}</span>'
                   f'<span><i style="background:var(--hl)"></i>{esc(hl_label[1])}</span></div>')
        self.parts.append(f'<div class="card">{leg}<svg viewBox="0 0 {W} {H}" width="{W}">{"".join(marks)}</svg></div>')
        self.text.append(f"  {title}")
        mx = max([v for v in values if v is not None and v == v] + [1e-30])
        for i, (lab, v) in enumerate(zip(labels, values)):
            vv = 0 if v is None or v != v else v
            self.text.append(f"   {str(lab):>24s} {fmt(v, fmt_s):>9s} {'#' * int(round(30 * vv / mx)) if mx > 0 else ''}"
                             + ("  *" if i in highlight else ""))

    def lines(self, x, series, title="", fmt_s="{:.3f}", x_label="", y_min=None, y_max=None, markers=None, width=760):
        """series: list of (name, [y...]). markers: list of (x_value, text) vertical guides."""
        if not x or not series:
            return
        W, H = width, 260
        pl, pr, pt, pb = 52, 16, 30, 40
        pw, ph = W - pl - pr, H - pt - pb
        ys = [v for _, s in series for v in s if v is not None and v == v]
        lo = min(ys) if y_min is None else y_min
        hi = max(ys) if y_max is None else y_max
        if hi - lo < 1e-12:
            hi = lo + 1
        xs = [float(v) for v in x]
        x0, x1 = min(xs), max(xs)
        if x1 - x0 < 1e-12:
            x1 = x0 + 1
        px = lambda v: pl + pw * (float(v) - x0) / (x1 - x0)
        py = lambda v: pt + ph - ph * (v - lo) / (hi - lo)
        m = [f'<text class="ttl" x="0" y="14">{esc(title)}</text>']
        for k in range(5):
            val = lo + (hi - lo) * k / 4
            y = py(val)
            m.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{W - pr}" y2="{y:.1f}" stroke="var(--grid)"/>'
                     f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(val, fmt_s)}</text>')
        step = max(1, int(math.ceil(len(xs) / 12)))
        for i, v in enumerate(xs):
            if i % step == 0 or i == len(xs) - 1:
                m.append(f'<text x="{px(v):.1f}" y="{H - pb + 16}" text-anchor="middle">{esc(x[i])}</text>')
        if x_label:
            m.append(f'<text x="{pl + pw / 2:.1f}" y="{H - 4}" text-anchor="middle">{esc(x_label)}</text>')
        for xv, txt in (markers or []):
            xx = px(xv)
            m.append(f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt + ph}" stroke="var(--muted)" stroke-dasharray="4 4"/>'
                     f'<text x="{xx + 4:.1f}" y="{pt + 10}">{esc(txt)}</text>')
        for si, (name, s) in enumerate(series):
            col = f"var(--s{si % 8 + 1})"
            pts = [(px(xs[i]), py(v)) for i, v in enumerate(s) if v is not None and v == v]
            if len(pts) > 1:
                m.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="'
                         + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
            for i, v in enumerate(s):
                if v is None or v != v:
                    continue
                m.append(f'<circle class="m" cx="{px(xs[i]):.1f}" cy="{py(v):.1f}" r="4" fill="{col}" stroke="var(--surface)" '
                         f'stroke-width="2"><title>{esc(name)} @ {esc(x[i])}: {fmt(v, fmt_s)}</title></circle>')
        m.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{W - pr}" y2="{pt + ph}" stroke="var(--axis)"/>')
        leg = ""
        if len(series) > 1:
            leg = '<div class="legend">' + "".join(
                f'<span><i style="background:var(--s{i % 8 + 1})"></i>{esc(n)}</span>' for i, (n, _) in enumerate(series)) + "</div>"
        self.parts.append(f'<div class="card">{leg}<svg viewBox="0 0 {W} {H}" width="{W}">{"".join(m)}</svg></div>')
        self.text.append(f"  {title}")
        self.text.append("   x: " + " ".join(f"{str(v):>8s}" for v in x))
        for name, s in series:
            self.text.append(f"   {name[:20]:20s} " + " ".join(f"{fmt(v, fmt_s):>8s}" for v in s))

    def heatmap(self, rows, cols, M, title="", fmt_s="{:.2f}", vmin=None, vmax=None, show_values=True, reverse=False):
        """Sequential one-hue heatmap. reverse=True: small values dark (distances)."""
        nr, nc = len(rows), len(cols)
        if not nr or not nc:
            return
        vals = [v for r in M for v in r if v is not None and v == v]
        lo = min(vals) if vmin is None else vmin
        hi = max(vals) if vmax is None else vmax
        if hi - lo < 1e-12:
            hi = lo + 1
        lw = min(240, 7 * max(len(str(x)) for x in rows) + 10)
        cw = max(26, min(70, 7 * max(len(str(x)) for x in cols) + 8)) if show_values else 22
        ch = 22
        top = 30 + min(150, 6 * max(len(str(x)) for x in cols))
        W, H = lw + nc * cw + 10, top + nr * ch + 10
        m = [f'<text class="ttl" x="0" y="14">{esc(title)}</text>']
        for j, c in enumerate(cols):
            x = lw + j * cw + cw / 2
            m.append(f'<text transform="translate({x:.1f},{top - 6}) rotate(-45)">{esc(c)}</text>')
        for i, r in enumerate(rows):
            y = top + i * ch
            m.append(f'<text x="{lw - 6}" y="{y + 15}" text-anchor="end">{esc(r)}</text>')
            for j in range(nc):
                v = M[i][j]
                if v is None or v != v:
                    col, txtc = "var(--grid)", "var(--ink2)"
                else:
                    f = (v - lo) / (hi - lo)
                    f = 1 - f if reverse else f
                    k = max(0, min(len(SEQ) - 1, int(round(f * (len(SEQ) - 1)))))
                    col = SEQ[k]
                    txtc = "#ffffff" if k >= 3 else "#0b0b0b"
                m.append(f'<rect class="m" x="{lw + j * cw + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{col}">'
                         f'<title>{esc(r)} × {esc(cols[j])}: {fmt(v, fmt_s)}</title></rect>')
                if show_values:
                    m.append(f'<text x="{lw + j * cw + cw / 2:.1f}" y="{y + 15}" text-anchor="middle" '
                             f'style="fill:{txtc};font-size:10.5px">{fmt(v, fmt_s)}</text>')
        self.parts.append(f'<div class="card"><svg viewBox="0 0 {W} {H}" width="{W}">{"".join(m)}</svg></div>')
        self.text.append(f"  {title}")
        self.text.append("   " + " " * 22 + " ".join(f"{str(c)[:8]:>8s}" for c in cols))
        for i, r in enumerate(rows):
            self.text.append(f"   {str(r)[:22]:22s}" + " ".join(f"{fmt(v, fmt_s):>8s}" for v in M[i]))

    # --- output
    def html(self):
        body = "".join(self.parts)
        return (f'<div class="amr"><style>{CSS}</style><h1>{esc(self.title)}</h1>{body}'
                f'<p class="note" style="margin-top:28px">{esc(t("Отчёт AnimaMerger. Наведи курсор на столбик или точку — покажет точное значение.", "AnimaMerger report. Hover a bar or a point to see the exact value."))}</p></div>')

    def save(self, name=None, folder=None):
        folder = Path(folder) if folder else reports_dir()
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / f"{name or self.kind}_{stamp()}"
        page = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{esc(self.title)}</title><style>body{{margin:0;background:#f9f9f7}}"
                "@media (prefers-color-scheme:dark){body{background:#0d0d0d}}</style></head><body>"
                + self.html() + "</body></html>")
        Path(str(base) + ".html").write_text(page, encoding="utf-8")
        Path(str(base) + ".txt").write_text("\n".join(self.text) + "\n", encoding="utf-8")
        Path(str(base) + ".json").write_text(json.dumps(self.data, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
        log("\n" + "\n".join(self.text))
        log("\n" + t("Отчёт: ", "Report: ") + str(base) + ".html")
        emit_report(str(base) + ".html")
        return str(base) + ".html"
