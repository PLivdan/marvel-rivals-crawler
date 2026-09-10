"""TikZ/pgfplots figures for the revised lineup report, styled after Figure A2 of the headline report
(axis lines left, no tick marks, dashed zero line, open/filled circles joined by thin segments, italic
group labels, legend inside without a frame, document fonts). Drawn from SAVED outputs only (no refitting).
Inputs: results/lineup_selected_replacement.csv, results/lineup_selected_pairs_p50.csv,
results/lineup_tuning.json, results/lineup_baseline_dev.json, results/lineup_confirmation.json,
results/lineup_selected_summary_meta.json.
Outputs: results/figures/lineup_*.tikz (input by the report) and results/figures/lineup_*.pdf (the same
pictures compiled standalone at the report's 11pt size). Each picture is sized in two passes so that its
full extent, labels included, matches the report's text width (and a target height for the tall ones)."""
import json, os, subprocess, glob, re
import numpy as np, pandas as pd
PREFIX = "results/lineup_selected"; OUT = "results/figures"
R = pd.read_csv(f"{PREFIX}_replacement.csv"); P = pd.read_csv(f"{PREFIX}_pairs_p50.csv"); SM = json.load(open(f"{PREFIX}_summary_meta.json"))
TUN = json.load(open("results/lineup_tuning.json")); BASE = json.load(open("results/lineup_baseline_dev.json")); CONF = json.load(open("results/lineup_confirmation.json"))
TEXT_W = 468.0            # pt; the report's text width is 6.6 in = 475 pt
TALL_H = 566.0            # pt; leaves room for a one-line caption and three to four lines of notes
def esc(s):
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]: s = s.replace(a, b)
    return s
def fnum(x, d=2): return f"{x:+.{d}f}".replace("-", "$-$")
ROLES = [("Tank", "Tanks"), ("Damage", "Damage heroes"), ("Support", "Supports")]
COMMON = (r"scale only axis, axis lines=left, tick style={draw=none}, y axis line style={draw=none}, clip=false," "\n"
          r"  label style={font=\small}, tick label style={font=\footnotesize}, yticklabel style={font=\footnotesize}," "\n"
          r"  legend style={draw=none, fill=none, font=\footnotesize, cells={anchor=west}}, legend cell align=left,")

def grouped_layout(groups, name_col="name", label_h=1.0, label_at=0.5):
    """A2 geometry: a label row for each group, then one row per item, first item at the top."""
    rows, ticks, labels, seps, glabs, y = [], [], [], [], [], 0
    for glabel, block in groups:
        glabs.append((glabel, y + label_at)); y += label_h
        for r in block.itertuples():
            rows.append((r, y)); ticks.append(y); labels.append(esc(getattr(r, name_col))); y += 1
        seps.append(y)
    ymax = y + 0.4
    def Y(v): return ymax - v
    return rows, ticks, labels, seps[:-1], glabs, ymax, Y
def tick_opts(ticks, labels, Y):
    return "ytick={" + ",".join(f"{Y(t):.2f}" for t in ticks) + "}, yticklabels={" + ",".join("{" + l + "}" for l in labels) + "}"
def sep_lines(seps, Y, xmin, xmax):
    return "\n".join(rf"\draw[black!35, thin] (axis cs:{xmin},{Y(s):.2f}) -- (axis cs:{xmax},{Y(s):.2f});" for s in seps)
def group_labels(glabs, Y, x):
    return "\n".join(rf"\node[anchor=west, font=\footnotesize\itshape, fill=white, inner sep=1pt] at (axis cs:{x},{Y(y):.2f}) {{{lab}}};" for lab, y in glabs)

WRAP = r"""\documentclass[11pt,border=1pt]{standalone}
\usepackage{pgfplots,amsmath}\pgfplotsset{compat=newest}
\begin{document}\input{%s.tikz}\end{document}"""
def compile_measure(name):
    open(f"{OUT}/{name}.tex", "w").write(WRAP % name)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=OUT, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:]); raise SystemExit(f"{name}: standalone compile failed")
    for ext in ("aux", "log", "tex"): os.remove(f"{OUT}/{name}.{ext}")
    w, h = subprocess.run(["magick", "identify", "-density", "72", "-format", "%w %h", f"{OUT}/{name}.pdf"], capture_output=True, text=True).stdout.split()
    return float(w) - 2, float(h) - 2                      # minus the 1 pt border on each side
def fit(name, render, dims, target_w=TEXT_W, target_h=None, wkeys=("W",), hkey="H"):
    """Two passes: render with guessed axis-box sizes, measure the full picture, shrink or grow the boxes
    so the picture's outer size lands on the targets (labels do not change with box size)."""
    open(f"{OUT}/{name}.tikz", "w").write(render(dims).strip() + "\n"); w1, h1 = compile_measure(name); sizes = [(w1, h1)]
    for _ in range(3):
        w, h = sizes[-1]
        if abs(w - target_w) <= 1.5 and (target_h is None or abs(h - target_h) <= 1.5): break
        dw = (w - target_w) / len(wkeys)
        for k in wkeys: dims[k] -= dw
        if target_h is not None: dims[hkey] -= h - target_h
        open(f"{OUT}/{name}.tikz", "w").write(render(dims).strip() + "\n"); sizes.append(compile_measure(name))
    print(f"{name}: " + " -> ".join(f"{w:.0f}x{h:.0f}" for w, h in sizes) + f" pt (targets {target_w:.0f}x{target_h if target_h else 0:.0f}); boxes {dict((k, round(v)) for k, v in dims.items())}")

# ---------- Figure: common-reference replacement values, all 55 heroes, values and eligibility printed ------
groups = [(lab, R[R.role == ro].sort_values("common_ref_pp", ascending=False)) for ro, lab in ROLES]
rows, ticks, labels, seps, glabs, ymax, Y = grouped_layout(groups, label_h=0.8, label_at=0.35)
def render_replacement(d):
    XMIN, XMAX, COVX = -8, 8, 9.6
    pts = "\n".join(f"({r.common_ref_pp:.3f},{Y(y):.2f})" for r, y in rows)
    stems = "\n".join(rf"\draw[black!45, thin] (axis cs:0,{Y(y):.2f}) -- (axis cs:{r.common_ref_pp:.3f},{Y(y):.2f});" for r, y in rows)
    vals = "\n".join((rf"\node[anchor=west, font=\footnotesize] at (axis cs:{r.common_ref_pp + 0.22:.3f},{Y(y):.2f}) {{{fnum(r.common_ref_pp)}}};" if r.common_ref_pp >= 0 else
                      rf"\node[anchor=east, font=\footnotesize] at (axis cs:{r.common_ref_pp - 0.22:.3f},{Y(y):.2f}) {{{fnum(r.common_ref_pp)}}};") for r, y in rows)
    cov = "\n".join(rf"\node[anchor=east, font=\footnotesize, text=black!60] at (axis cs:{COVX},{Y(y):.2f}) {{{100*r.coverage:.0f}}};" for r, y in rows)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width={d['W']:.1f}pt, height={d['H']:.1f}pt, {COMMON}
  xmin={XMIN}, xmax={XMAX}, ymin=0, ymax={ymax:.2f}, xtick={{-8,-6,...,8}},
  xlabel={{Change in predicted win probability, percentage points}},
  {tick_opts(ticks, labels, Y)},
]
\draw[dashed, thin] (axis cs:0,0) -- (axis cs:0,{ymax:.2f});
{sep_lines(seps, Y, XMIN, XMAX)}
{stems}
\addplot[only marks, mark=*, mark size=1.7pt] coordinates {{
{pts}
}};
{group_labels(glabs, Y, XMIN + 0.15)}
{vals}
\node[anchor=south east, font=\footnotesize, text=black!60, align=right] at (axis cs:{COVX},{ymax - 0.45:.2f}) {{Eligible\\ slots (\%)}};
{cov}
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_replacement", render_replacement, {"W": 330.0, "H": 520.0}, target_h=TALL_H)

# ---------- Figure: change across rank environments, highlight (four largest increases, four largest decreases)
R["chg"] = R.obs_meta_pp_third2 - R.obs_meta_pp_third0
hi4 = R.sort_values("chg", ascending=False).head(4); lo4 = R.sort_values("chg").head(4)
sel = pd.concat([hi4, lo4]).sort_values("chg", ascending=False).reset_index(drop=True)
LEG_LO = "Lower third by lobby rank"; LEG_HI = "Upper third by lobby rank"
def dumbbell_block(rows, Y, values=True, off=0.3):
    segs = "\n".join(rf"\draw[thin] (axis cs:{r.obs_meta_pp_third0:.3f},{Y(y):.2f}) -- (axis cs:{r.obs_meta_pp_third2:.3f},{Y(y):.2f});" for r, y in rows)
    lo = "\n".join(f"({r.obs_meta_pp_third0:.3f},{Y(y):.2f})" for r, y in rows); hi = "\n".join(f"({r.obs_meta_pp_third2:.3f},{Y(y):.2f})" for r, y in rows)
    v = ""
    if values:
        v = "\n".join(rf"\node[anchor=east, font=\footnotesize] at (axis cs:{min(r.obs_meta_pp_third0, r.obs_meta_pp_third2) - off:.3f},{Y(y):.2f}) {{{fnum(min(r.obs_meta_pp_third0, r.obs_meta_pp_third2))}}};" "\n"
                      rf"\node[anchor=west, font=\footnotesize] at (axis cs:{max(r.obs_meta_pp_third0, r.obs_meta_pp_third2) + off:.3f},{Y(y):.2f}) {{{fnum(max(r.obs_meta_pp_third0, r.obs_meta_pp_third2))}}};" for r, y in rows)
    return segs, lo, hi, v
hrows = [(r, i + 1) for i, r in enumerate(sel.itertuples())]; hymax = len(sel) + 0.6
def HY(v): return hymax - v
def render_highlight(d):
    segs, lo, hi, v = dumbbell_block(hrows, HY)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width={d['W']:.1f}pt, height={d['H']:.1f}pt, {COMMON}
  xmin=-10, xmax=10, ymin=0.4, ymax={hymax:.2f}, xtick={{-8,-6,...,8}},
  xlabel={{Observed-incumbent replacement value, percentage points}},
  {tick_opts([y for _, y in hrows], [esc(r.name) for r, _ in hrows], HY)},
  legend style={{at={{(0.985,0.985)}}, anchor=north east}},
]
\draw[dashed, thin] (axis cs:0,0.4) -- (axis cs:0,{hymax:.2f});
{segs}
\addplot[only marks, mark=o, mark size=1.9pt] coordinates {{
{lo}
}};
\addlegendentry{{{LEG_LO}}}
\addplot[only marks, mark=*, mark size=1.9pt] coordinates {{
{hi}
}};
\addlegendentry{{{LEG_HI}}}
{v}
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_rank_highlight", render_highlight, {"W": 360.0, "H": 150.0})

# ---------- Figure: the same for all 55 heroes on one page (Figure A2 of the headline report, remade) ------
groups = [(lab, R[R.role == ro].sort_values("chg", ascending=False)) for ro, lab in ROLES]
rows_a, ticks_a, labels_a, seps_a, glabs_a, ymax_a, Ya = grouped_layout(groups, label_h=0.8, label_at=0.35)
def render_rank_all(d):
    XMIN, XMAX = -9, 9
    segs, lo, hi, _ = dumbbell_block(rows_a, Ya, values=False)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width={d['W']:.1f}pt, height={d['H']:.1f}pt, {COMMON}
  xmin={XMIN}, xmax={XMAX}, ymin=0, ymax={ymax_a:.2f}, xtick={{-8,-6,...,8}},
  xlabel={{Observed-incumbent replacement value, percentage points}},
  {tick_opts(ticks_a, labels_a, Ya)},
  legend style={{at={{(0.985,0.99)}}, anchor=north east}},
]
\draw[dashed, thin] (axis cs:0,0) -- (axis cs:0,{ymax_a:.2f});
{sep_lines(seps_a, Ya, XMIN, XMAX)}
{segs}
\addplot[only marks, mark=o, mark size=1.6pt] coordinates {{
{lo}
}};
\addlegendentry{{{LEG_LO}}}
\addplot[only marks, mark=*, mark size=1.6pt] coordinates {{
{hi}
}};
\addlegendentry{{{LEG_HI}}}
{group_labels(glabs_a, Ya, XMIN + 0.15)}
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_rank_all", render_rank_all, {"W": 360.0, "H": 520.0}, target_h=TALL_H)

# ---------- Figure: pair contrasts, three sections on one axis ---------------------------------------------
MIN = 500
A = P[(P.kind == "allied") & (P.support_dev >= MIN)].copy(); O = P[(P.kind == "opposing") & (P.support_dev >= MIN)].copy()
TU = A[A.is_teamup]; NT = A[~A.is_teamup]
tu_sel = pd.concat([TU.sort_values("did", ascending=False).head(10), TU.sort_values("did").head(5).sort_values("did", ascending=False)])
nt_sel = pd.concat([NT.sort_values("did", ascending=False).head(8), NT.sort_values("did").head(8).sort_values("did", ascending=False)])
O["a"] = O.did.abs(); o_sel = O.sort_values("a", ascending=False).head(14)
def orient(r): return (r.hero_a, r.hero_b, r.did) if r.did >= 0 else (r.hero_b, r.hero_a, -r.did)
tu_sel = tu_sel.assign(name=[f"{r.hero_a} + {r.hero_b}" for r in tu_sel.itertuples()], v=tu_sel.did)
nt_sel = nt_sel.assign(name=[f"{r.hero_a} + {r.hero_b}" for r in nt_sel.itertuples()], v=nt_sel.did)
o_or = [orient(r) for r in o_sel.itertuples()]
o_sel = o_sel.assign(name=[f"{a} against {b}" for a, b, _ in o_or], v=[v for _, _, v in o_or])
groups = [("Team-up-eligible allied pairs", tu_sel), ("Other allied pairs", nt_sel), ("Opposing matchups, A against B", o_sel)]
rows_p, ticks_p, labels_p, seps_p, glabs_p, ymax_p, Yp = grouped_layout(groups, label_h=1.9, label_at=0.8)
def render_pairs(d):
    XMIN, XMAX = -0.27, 0.36
    stems = "\n".join(rf"\draw[black!45, thin] (axis cs:0,{Yp(y):.2f}) -- (axis cs:{r.v:.4f},{Yp(y):.2f});" for r, y in rows_p)
    pts = "\n".join(f"({r.v:.4f},{Yp(y):.2f})" for r, y in rows_p)
    vals = "\n".join((rf"\node[anchor=west, font=\footnotesize] at (axis cs:{r.v + 0.008:.4f},{Yp(y):.2f}) {{{fnum(r.v, 3)} ({int(r.support_dev):,})}};" if r.v >= 0 else
                      rf"\node[anchor=east, font=\footnotesize] at (axis cs:{r.v - 0.008:.4f},{Yp(y):.2f}) {{{fnum(r.v, 3)} ({int(r.support_dev):,})}};") for r, y in rows_p)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width={d['W']:.1f}pt, height={d['H']:.1f}pt, {COMMON}
  xmin={XMIN}, xmax={XMAX}, ymin=0, ymax={ymax_p:.2f}, xtick={{-0.2,-0.1,...,0.3}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  xlabel={{Four-lineup contrast, log-odds (identifying matches in parentheses)}},
  {tick_opts(ticks_p, labels_p, Yp)},
]
\draw[dashed, thin] (axis cs:0,0) -- (axis cs:0,{ymax_p:.2f});
{sep_lines(seps_p, Yp, XMIN, XMAX)}
{stems}
\addplot[only marks, mark=*, mark size=1.7pt] coordinates {{
{pts}
}};
{group_labels(glabs_p, Yp, XMIN + 0.005)}
{vals}
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_pairs", render_pairs, {"W": 250.0, "H": 470.0}, target_h=500.0)

# ---------- Figure: prediction (development folds + internal temporal evaluation) ---------------------------
def ev(k): return next(e for e in TUN["evaluations"] if e["key"] == k and e["sub"] == 1.0)
red = TUN["reduced"]
series = [("Refitted headline specification", [f["logloss"] for f in BASE["folds"]], "dashed, mark=square*, mark size=1.8pt"),
          ("Heroes, composition, maps, rank imbalance", [f["logloss"] for f in ev(red["no_pairs_no_slopes_no_heromap"])["folds"]], "dotted, mark=triangle*, mark size=2.2pt"),
          ("+ rank slopes and hero-by-map", [f["logloss"] for f in ev(red["no_pairs"])["folds"]], "dashdotted, mark=diamond*, mark size=2.2pt"),
          ("+ shrunk allied and opposing pairs (unified)", [f["logloss"] for f in ev(red["selected"])["folds"]], "thick, mark=*, mark size=1.8pt")]
allv = [v for _, vs, _ in series for v in vs]; ylo, yhi = min(allv) - 0.0006, max(allv) + 0.0006
D = [CONF["D"]] + [t["D"] for t in CONF["by_third"]]; ns = [CONF["n"]] + [t["n"] for t in CONF["by_third"]]
ylabs = ["All matches", "Lower third", "Middle third", "Upper third"]
iid = CONF["ci_iid"]; boot = CONF["ci_boot"]
def render_prediction(d):
    plots = "\n".join(rf"\addplot[{sty}] coordinates {{(F1,{vs[0]:.5f}) (F2,{vs[1]:.5f}) (F3,{vs[2]:.5f})}};" "\n" rf"\addlegendentry{{{lab}}}" for lab, vs, sty in series)
    ytl = ",".join("{" + f"{l}\\\\($n$ = {n:,})" + "}" for l, n in zip(ylabs, ns))
    vals = "\n".join(rf"\node[anchor=east, font=\footnotesize] at (axis cs:0.0063,{4 - i}) {{{fnum(x, 4)}}};" for i, x in enumerate(D))
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  name=folds, scale only axis, width={d['W1']:.1f}pt, height={d['H']:.1f}pt, axis lines=left, tick style={{draw=none}}, clip=false,
  symbolic x coords={{F1,F2,F3}}, xtick={{F1,F2,F3}}, enlarge x limits=0.2,
  xticklabels={{{{Fold 1\\(train 60\%)}},{{Fold 2\\(train 73\%)}},{{Fold 3\\(train 87\%)}}}}, xticklabel style={{align=center, font=\footnotesize}},
  ylabel={{Validation log loss (lower is better)}}, ylabel style={{font=\small}},
  yticklabel style={{font=\footnotesize, /pgf/number format/fixed, /pgf/number format/fixed zerofill, /pgf/number format/precision=3}}, scaled y ticks=false,
  ymin={ylo:.4f}, ymax={yhi:.4f},
  title={{\small Chronological development folds}}, title style={{yshift=-3pt}},
  legend style={{at={{(0,-0.24)}}, anchor=north west, draw=none, fill=none, font=\footnotesize, cells={{anchor=west}}}}, legend cell align=left,
]
{plots}
\end{{axis}}
\begin{{axis}}[
  name=ev, at={{(folds.south east)}}, anchor=south west, xshift={d['GAP']:.1f}pt, scale only axis, width={d['W2']:.1f}pt, height={d['H']:.1f}pt,
  axis lines=left, tick style={{draw=none}}, y axis line style={{draw=none}}, clip=false,
  xmin=-0.0025, xmax=0.0065, ymin=0.4, ymax=4.6, ytick={{4,3,2,1}}, yticklabels={{{ytl}}}, yticklabel style={{align=right, font=\footnotesize}},
  xtick={{-0.002,0,0.002,0.004,0.006}}, xticklabel style={{font=\footnotesize, /pgf/number format/fixed, /pgf/number format/precision=3}}, scaled x ticks=false,
  xlabel={{Paired log-loss improvement over the refitted headline\\(positive = lower error)}}, xlabel style={{font=\small, align=center}},
  title={{\small Internal temporal evaluation}}, title style={{yshift=-3pt}},
  legend style={{at={{(1,-0.3)}}, anchor=north east, draw=none, fill=none, font=\footnotesize, cells={{anchor=west}}}}, legend cell align=left,
]
\draw[dashed, thin] (axis cs:0,0.4) -- (axis cs:0,4.6);
\draw[thin] (axis cs:{iid[0]:.5f},4) -- (axis cs:{iid[1]:.5f},4);
\draw[thin] (axis cs:{iid[0]:.5f},3.85) -- (axis cs:{iid[0]:.5f},4.15); \draw[thin] (axis cs:{iid[1]:.5f},3.85) -- (axis cs:{iid[1]:.5f},4.15);
\draw[line width=1.8pt] (axis cs:{boot[0]:.5f},4) -- (axis cs:{boot[1]:.5f},4);
\addplot[only marks, mark=*, mark size=2pt] coordinates {{({D[0]:.5f},4)}};
\addlegendentry{{Overall improvement}}
\addlegendimage{{line width=1.8pt}}
\addlegendentry{{Player-resampling 95\% interval}}
\addlegendimage{{thin}}
\addlegendentry{{Match-i.i.d.\ 95\% interval}}
\addplot[only marks, mark=o, mark size=2pt] coordinates {{({D[1]:.5f},3) ({D[2]:.5f},2) ({D[3]:.5f},1)}};
\addlegendentry{{By lobby-rank third (no interval)}}
{vals}
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_prediction", render_prediction, {"W1": 175.0, "W2": 140.0, "H": 150.0, "GAP": 78.0}, wkeys=("W1", "W2"))

# ---------- Figure: predicted against realised win rates on the evaluation slice (by hero, by composition) ------
FS = json.load(open("results/lineup_fit_scatter.json"))
HP = pd.DataFrame(FS["hero"]["points"]); SP = pd.DataFrame(FS["shape"]["points"])
def bounds(df, pad=0.01, step=0.05):
    lo = min(df.pred.min(), df.real.min()) - pad; hi = max(df.pred.max(), df.real.max()) + pad
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)
def label_nodes(d, lo, hi, keys=None, reserved=(), char_w=0.0105):
    """One node per labelled point, ported from the headline report: near candidates sit diagonally next to
    the point on the empty side of the 45-degree line; if every near candidate would overlap another point,
    another label, a reserved region or the frame, the label moves out and a thin leader joins it."""
    span = hi - lo
    norm = lambda x, y: ((x - lo) / span, (y - lo) / span)
    back = lambda nx, ny: (lo + nx * span, lo + ny * span)
    points = [norm(r.pred, r.real) for r in d.itertuples()]
    ANCHOR = {(-1, 1): "south east", (1, 1): "south west", (-1, -1): "north east", (1, -1): "north west",
              (0, 1): "south", (0, -1): "north", (-1, 0): "east", (1, 0): "west"}
    placed, out = [], []
    for r in d.sort_values("pred").itertuples():
        if keys is not None and r.key not in keys: continue
        hw = max(0.04, char_w * len(str(r.key)))
        def centre(px, py, sx, sy): return px + sx * hw, py + sy * 0.025
        def score(cx, cy, own):
            if not (0.0 <= cx - hw and cx + hw <= 1.0 and 0.0 <= cy - 0.025 and cy + 0.025 <= 1.0): return -1.0
            if any(x0 - hw <= cx <= x1 + hw and y0 - 0.03 <= cy <= y1 + 0.03 for x0, x1, y0, y1 in reserved): return 0.0
            best = 9.0
            for qx, qy in points:
                if (qx, qy) != own: best = min(best, (((cx - qx) / (hw + 0.02)) ** 2 + ((cy - qy) / 0.06) ** 2) ** 0.5)
            for qx, qy, qw in placed: best = min(best, (((cx - qx) / (hw + qw + 0.03)) ** 2 + ((cy - qy) / 0.06) ** 2) ** 0.5)
            return best
        nx, ny = norm(r.pred, r.real); own = (nx, ny); cands = []
        for sx, sy in ((-1, 1), (1, -1), (1, 1), (-1, -1)):
            px, py = nx + sx * 0.012, ny + sy * 0.008
            cands.append((score(*centre(px, py, sx, sy), own), False, sx, sy, (px, py)))
        default = cands[0] if ny >= nx else cands[1]
        if default[0] >= 1.0: best = default
        else:
            for radius in (0.10, 0.15, 0.20):
                for sx, sy in ANCHOR:
                    px, py = nx + sx * radius, ny + sy * radius
                    cands.append((score(*centre(px, py, sx, sy), own), True, sx, sy, (px, py)))
            best = max(cands, key=lambda c: (min(c[0], 1.3), not c[1], c[2] == -1, c[3] == 0))
        sc_, far, sx, sy, (px, py) = best
        cx, cy = centre(px, py, sx, sy); placed.append((cx, cy, hw)); lx, ly = back(px, py)
        if far: out.append(rf"\draw[very thin] (axis cs:{r.pred:.4f},{r.real:.4f}) -- (axis cs:{lx:.4f},{ly:.4f});")
        out.append(rf"\node[font=\scriptsize, anchor={ANCHOR[(sx, sy)]}, inner sep=0.6pt, fill=white, fill opacity=0.85, text opacity=1] at (axis cs:{lx:.4f},{ly:.4f}) {{{esc(r.key)}}};")
    return "\n".join(out)
HP["key"] = HP.name; SP["key"] = SP["shape"]; HP["resid"] = HP.real - HP.pred
H_LABELS = set(HP.sort_values("resid", key=abs, ascending=False).head(2).key) | {HP.loc[HP.pred.idxmax(), "key"], HP.loc[HP.pred.idxmin(), "key"]}
A_LO, A_HI = bounds(HP); A_LO -= 0.05; B_LO, B_HI = bounds(SP)
def render_fit(d):
    hu, hb, su = FS["hero"]["unified"], FS["hero"]["headline"], FS["shape"]["unified"]
    rowsA = "\n".join(f"{r.pred:.4f} {r.real:.4f}" for r in HP.itertuples()); rowsB = "\n".join(f"{r.pred:.4f} {r.real:.4f}" for r in SP.itertuples())
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  name=A, scale only axis, width={d['W1']:.1f}pt, height={d['W1']:.1f}pt, axis lines=left, tick style={{draw=none}}, clip=false,
  xmin={A_LO:.2f}, xmax={A_HI:.2f}, ymin={A_LO:.2f}, ymax={A_HI:.2f},
  title={{\small Panel A: by starting hero ({len(HP)} heroes)}}, title style={{yshift=-3pt}},
  xlabel={{Predicted win rate of teams starting the hero}}, ylabel={{Realised win rate on the evaluation slice}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize, /pgf/number format/fixed, /pgf/number format/precision=2}},
  legend style={{at={{(0.03,0.97)}}, anchor=north west, font=\footnotesize, draw=none, fill=none, cells={{anchor=west}}}},
]
\addplot[dashed, thin, domain={A_LO:.2f}:{A_HI:.2f}, samples=2] {{x}};
\addlegendentry{{$45^\circ$ line}}
\addplot[only marks, mark=*, mark size=1.7pt] table[x=p, y=a] {{
p a
{rowsA}
}};
\addlegendentry{{Unified model}}
{label_nodes(HP, A_LO, A_HI, H_LABELS, reserved=[(0.0, 0.45, 0.86, 1.0), (0.48, 1.0, 0.0, 0.30)])}
\node[anchor=south east, align=right, font=\scriptsize] at (rel axis cs:0.98,0.02)
  {{Unified model:\\ corr.\ {hu['corr']:.3f}, mean $|$gap$|$ {hu['mean_abs_gap_pp']:.2f} pp\\[2pt] Refitted headline:\\ corr.\ {hb['corr']:.3f}, mean $|$gap$|$ {hb['mean_abs_gap_pp']:.2f} pp}};
\end{{axis}}
\begin{{axis}}[
  name=B, at={{(A.south east)}}, anchor=south west, xshift={d['GAP']:.1f}pt, scale only axis, width={d['W2']:.1f}pt, height={d['W2']:.1f}pt,
  axis lines=left, tick style={{draw=none}}, clip=false,
  xmin={B_LO:.2f}, xmax={B_HI:.2f}, ymin={B_LO:.2f}, ymax={B_HI:.2f},
  title={{\small Panel B: by team composition ({len(SP)} shapes)}}, title style={{yshift=-3pt}},
  xlabel={{Predicted win rate of teams with the composition}}, ylabel={{Realised win rate on the evaluation slice}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize, /pgf/number format/fixed, /pgf/number format/precision=2}},
  legend style={{at={{(0.03,0.97)}}, anchor=north west, font=\footnotesize, draw=none, fill=none, cells={{anchor=west}}}},
]
\addplot[dashed, thin, domain={B_LO:.2f}:{B_HI:.2f}, samples=2] {{x}};
\addlegendentry{{$45^\circ$ line}}
\addplot[only marks, mark=*, mark size=1.7pt] table[x=p, y=a] {{
p a
{rowsB}
}};
\addlegendentry{{Unified model}}
{label_nodes(SP, B_LO, B_HI, reserved=[(0.0, 0.42, 0.86, 1.0), (0.55, 1.0, 0.0, 0.18)])}
\node[anchor=south east, align=right, font=\scriptsize] at (rel axis cs:0.98,0.02)
  {{corr.\ {su['corr']:.3f}\\ mean $|$gap$|$ {su['mean_abs_gap_pp']:.2f} pp}};
\end{{axis}}
\end{{tikzpicture}}"""
fit("lineup_fit", render_fit, {"W1": 175.0, "W2": 175.0, "GAP": 66.0}, wkeys=("W1", "W2"))

for stale in ["lineup_replacement_ts.pdf", "lineup_replacement_damage.pdf"]:
    if os.path.exists(f"{OUT}/{stale}"): os.remove(f"{OUT}/{stale}")
print("figures written:", sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/lineup_*")))
print("highlight rows:", sel[["name", "obs_meta_pp_third0", "obs_meta_pp_third2", "chg"]].round(2).to_string(index=False))
