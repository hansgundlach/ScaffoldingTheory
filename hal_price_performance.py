"""
Price vs Performance (logit-scaled) frontier plots from HAL benchmark data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox
import numpy as np
import re
from pathlib import Path
from scipy.special import logit, expit

BASE = Path(__file__).parent
HAL = BASE / "hal_data"

# ─── Behavior toggles (edit these to quickly change plot behavior) ────────────
SHOW_POINT_LABELS = False     # annotate each scatter point with the model name
SHOW_VECTOR_LABELS = True     # annotate scaffold-switch arrows with the model name
COST_AXIS_IN_DOLLARS = True   # format cost axis as $1000 instead of 10^3
EPOCH_STYLE = False           # also render Epoch-AI-styled vector plots (hal_vectors_epoch_*.png)

# ─── Plot configuration ──────────────────────────────────────────────────────
# All styling knobs live here. Bump these to make text/markers larger.
SAVE_DPI = 150

# Overview grid figure (4 rows x 2 cols of all benchmarks)
GRID_FIGSIZE = (17, 32)
GRID_SUPTITLE_FONTSIZE = 30
GRID_TITLE_FONTSIZE = 23
GRID_LABEL_FONTSIZE = 21
GRID_TICK_FONTSIZE = 18
GRID_ANNOT_FONTSIZE = 9
GRID_LEGEND_FONTSIZE = 16
GRID_POINT_SIZE = 90
GRID_LINEWIDTH = 2

# Individual per-benchmark frontier figures
IND_FIGSIZE = (13, 8)
IND_TITLE_FONTSIZE = 23
IND_LABEL_FONTSIZE = 23
IND_TICK_FONTSIZE = 20
IND_ANNOT_FONTSIZE = 12
IND_LEGEND_FONTSIZE = 18
IND_POINT_SIZE = 130
IND_LINEWIDTH = 2.5

# Scaffold-switch vector figures
VEC_FIGSIZE = (17, 11)
VEC_TITLE_FONTSIZE = 24
VEC_LABEL_FONTSIZE = 26
VEC_TICK_FONTSIZE = 23
VEC_ANNOT_FONTSIZE = 19
VEC_LEGEND_FONTSIZE = 21
VEC_LEGEND_TITLE_FONTSIZE = 22
VEC_ARROW_LW = 5.5
VEC_ARROW_MUTATION = 55
VEC_POINT_SIZE = 140
VEC_POINT_ALPHA = 0.45

# Point outlines (helps faint markers stand out on busy plots)
POINT_EDGE_COLOR = "#1a1a1a"
POINT_EDGE_WIDTH = 1.2
VEC_POINT_EDGE_WIDTH = 2.0


def parse_pct(s):
    """Extract first numeric value from a percentage string like '38.81%' or '61.00%(-7.00/+7.00)'."""
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)%", str(s))
    return float(m.group(1)) if m else np.nan


def parse_cost(s):
    """Extract dollar cost from strings like '$15.15', '$1,577.26', '$463.90 (-438.32/+438.32)'."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "")
    m = re.search(r"\$?([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


def clean_scaffold(s):
    """Remove 'Pareto optimal' suffix to get base scaffold name."""
    if pd.isna(s):
        return s
    return re.sub(r"\s*Pareto optimal\s*", "", str(s)).strip()


def load_hal(filename, model_col="Primary Model"):
    """Load a HAL CSV, parse accuracy and cost, drop empty rows."""
    df = pd.read_csv(HAL / filename, encoding="utf-8-sig")
    # Normalize column names (some have trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    if "Models" in df.columns and model_col not in df.columns:
        model_col = "Models"
    df = df.rename(columns={model_col: "Model"})
    df["accuracy"] = df["Accuracy"].apply(parse_pct)
    df["cost"] = df["Cost (USD)"].apply(parse_cost)
    df["scaffold_raw"] = df["Scaffold"]
    df["scaffold"] = df["Scaffold"].apply(clean_scaffold)
    df = df.dropna(subset=["accuracy", "cost"])
    df = df[df["accuracy"] > 0]  # can't logit 0
    df = df[df["accuracy"] < 100]  # can't logit 100
    return df


def compute_pareto(df):
    """Return subset of df on the Pareto frontier (lower cost, higher accuracy)."""
    df_sorted = df.sort_values("cost")
    pareto = []
    best_acc = -1
    for _, row in df_sorted.iterrows():
        if row["accuracy"] > best_acc:
            pareto.append(row)
            best_acc = row["accuracy"]
    return pd.DataFrame(pareto)


def logit_score(pct):
    """Convert percentage (0-100) to logit scale."""
    p = np.clip(pct / 100, 0.001, 0.999)
    return logit(p)


def make_pct_formatter():
    """Formatter that shows logit-axis ticks as percentages."""
    def fmt(x, pos):
        pct = expit(x) * 100
        if pct >= 10:
            return f"{pct:.0f}%"
        return f"{pct:.1f}%"
    return mticker.FuncFormatter(fmt)


def make_dollar_formatter():
    """Formatter that shows cost-axis ticks as dollars (e.g. $1000 instead of 10^3)."""
    def fmt(x, pos):
        if x >= 1:
            return f"${x:,.0f}"
        return f"${x:,.2f}"
    return mticker.FuncFormatter(fmt)


# Define which benchmarks to plot and their configs
BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("usaco.csv", "USACO", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
    ("scicode.csv", "SciCode", "Primary Model"),
    ("online_mine_2_web.csv", "Online Mind2Web", "Primary Model"),
]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ─── Consistent scaffold colors across figures ───────────────────────────────
# Only scaffolds that appear in more than one benchmark get a pinned color so
# they look the same in every figure (in this data that's "HAL Generalist Agent").
# Single-benchmark scaffolds just take the normal local palette — they need no
# cross-figure consistency, and may freely reuse colors between different graphs.
def build_shared_scaffold_colors():
    counts = {}
    for fname, title, mcol in BENCHMARKS:
        try:
            df = load_hal(fname, mcol)
        except Exception:
            continue
        for s in {str(x) for x in df["scaffold"].dropna().unique()}:
            counts[s] = counts.get(s, 0) + 1
    shared = sorted([s for s, c in counts.items() if c > 1],
                    key=lambda s: (0 if "generalist" in s.lower() else 1, s.lower()))
    palette = list(plt.cm.tab10.colors)
    return {s: palette[i % len(palette)] for i, s in enumerate(shared)}


SHARED_SCAFFOLD_COLORS = build_shared_scaffold_colors()


def panel_scaffold_colors(scaffolds):
    """Color map for one panel: pinned colors for shared scaffolds, local palette
    (skipping the pinned colors) for the rest."""
    reserved = set(SHARED_SCAFFOLD_COLORS.values())
    avail = [c for c in plt.cm.tab10.colors if c not in reserved]
    out, ai = {}, 0
    for s in scaffolds:
        if str(s) in SHARED_SCAFFOLD_COLORS:
            out[s] = SHARED_SCAFFOLD_COLORS[str(s)]
        else:
            out[s] = avail[ai % len(avail)] if avail else plt.cm.tab10.colors[ai % 10]
            ai += 1
    return out


fig, axes = plt.subplots(4, 2, figsize=GRID_FIGSIZE)
fig.suptitle("HAL Benchmarks: Per-Scaffold Pareto Frontiers\n(y-axis = logit of accuracy, x-axis = log cost)",
             fontsize=GRID_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995)

for idx, (fname, title, mcol) in enumerate(BENCHMARKS):
    ax = axes.flat[idx]
    df = load_hal(fname, mcol)

    scaffolds = sorted(df["scaffold"].unique())
    scaffold_colors = panel_scaffold_colors(scaffolds)

    for si, scaffold in enumerate(scaffolds):
        grp = df[df["scaffold"] == scaffold].copy()
        color = scaffold_colors[scaffold]
        marker = MARKERS[si % len(MARKERS)]
        ls = LINESTYLES[si % len(LINESTYLES)]
        logit_acc = logit_score(grp["accuracy"])

        # Scatter points
        ax.scatter(grp["cost"], logit_acc, c=[color], s=GRID_POINT_SIZE, alpha=0.6,
                   marker=marker, edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_WIDTH, zorder=3)

        # Label points with model name
        if SHOW_POINT_LABELS:
            for _, row in grp.iterrows():
                model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
                model_short = model_short.replace("Claude ", "C.").replace("Sonnet", "Son").replace("Opus", "Op")
                model_short = model_short.replace("GPT-", "G").replace("DeepSeek", "DS")
                model_short = model_short.replace("Gemini", "Gem").replace("High", "H")
                ax.annotate(model_short, (row["cost"], logit_score(row["accuracy"])),
                            textcoords="offset points", xytext=(4, 3), fontsize=GRID_ANNOT_FONTSIZE,
                            alpha=0.7, color=color)

        # Per-scaffold Pareto frontier
        pareto = compute_pareto(grp)
        if len(pareto) >= 2:
            pareto_sorted = pareto.sort_values("cost")
            ax.step(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                    where="post", color=color, linewidth=GRID_LINEWIDTH, linestyle=ls,
                    alpha=0.8, zorder=4, label=scaffold)
        else:
            # Still add to legend even with 1 point
            ax.scatter([], [], c=[color], marker=marker, s=GRID_POINT_SIZE, label=scaffold)

    ax.set_xscale("log")
    ax.set_xlabel("Cost (USD, log scale)", fontsize=GRID_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (logit scale)", fontsize=GRID_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE, fontweight="bold")
    ax.yaxis.set_major_formatter(make_pct_formatter())
    if COST_AXIS_IN_DOLLARS:
        ax.xaxis.set_major_formatter(make_dollar_formatter())
    ax.tick_params(axis="both", labelsize=GRID_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=GRID_LEGEND_FONTSIZE, loc="lower right",
              framealpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.savefig(BASE / "figures" / "hal_price_performance_frontier.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.savefig(BASE / "figures" / "hal_price_performance_frontier.pdf", bbox_inches="tight")
print("Saved: hal_price_performance_frontier.png/pdf")

# ─── Individual larger figures per benchmark ─────────────────────────────────
for fname, title, mcol in BENCHMARKS:
    fig_ind, ax = plt.subplots(figsize=IND_FIGSIZE)
    df = load_hal(fname, mcol)

    scaffolds = sorted(df["scaffold"].unique())
    scaffold_colors = panel_scaffold_colors(scaffolds)

    for si, scaffold in enumerate(scaffolds):
        grp = df[df["scaffold"] == scaffold].copy()
        color = scaffold_colors[scaffold]
        marker = MARKERS[si % len(MARKERS)]
        ls = LINESTYLES[si % len(LINESTYLES)]
        logit_acc = logit_score(grp["accuracy"])

        ax.scatter(grp["cost"], logit_acc, c=[color], s=IND_POINT_SIZE, alpha=0.65,
                   marker=marker, edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_WIDTH, zorder=3)

        if SHOW_POINT_LABELS:
            for _, row in grp.iterrows():
                model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
                ax.annotate(model_short, (row["cost"], logit_score(row["accuracy"])),
                            textcoords="offset points", xytext=(5, 4), fontsize=IND_ANNOT_FONTSIZE,
                            alpha=0.8, color=color)

        # Per-scaffold Pareto frontier
        pareto = compute_pareto(grp)
        if len(pareto) >= 2:
            pareto_sorted = pareto.sort_values("cost")
            ax.step(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                    where="post", color=color, linewidth=IND_LINEWIDTH, linestyle=ls,
                    alpha=0.85, zorder=4, label=f"{scaffold} frontier")
            ax.scatter(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                       marker=marker, s=IND_POINT_SIZE + 30, facecolors="none", edgecolors=color,
                       linewidths=2, zorder=5)
        else:
            ax.scatter([], [], c=[color], marker=marker, s=IND_POINT_SIZE, label=scaffold)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (USD, log scale)", fontsize=IND_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (logit scale, shown as %)", fontsize=IND_LABEL_FONTSIZE)
    ax.set_title(f"{title}: Per-Scaffold Price-Performance Frontier", fontsize=IND_TITLE_FONTSIZE, fontweight="bold")
    ax.yaxis.set_major_formatter(make_pct_formatter())
    if COST_AXIS_IN_DOLLARS:
        ax.xaxis.set_major_formatter(make_dollar_formatter())
    ax.tick_params(axis="both", labelsize=IND_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=IND_LEGEND_FONTSIZE, loc="lower right",
              framealpha=0.9)

    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    fig_ind.tight_layout()
    fig_ind.savefig(BASE / "figures" / f"hal_frontier_{safe_name}.png", dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: hal_frontier_{safe_name}.png")
    plt.close(fig_ind)

# ─── Figure 3: Scaffold-switch vectors ───────────────────────────────────────
# For each benchmark, draw arrows from (cost, logit_acc) on scaffold A to
# (cost, logit_acc) on scaffold B for every model that appears on both.

def normalize_model_name(s):
    """Strip 'High' suffix and date parentheticals to match base model across configs."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    # Remove " High" at end (before any parenthetical)
    s = re.sub(r"\s+High\b", "", s)
    # Remove parenthetical dates like "(September 2025)"
    s = re.sub(r"\s*\(.*?\)", "", s)
    return s.strip()


# Color each vector by the quadrant it heads toward:
#   accuracy change (dy) x cost change (dx)
QUADRANT_COLORS = {
    "more_acc_less_exp": "#2ca02c",   # green  — better & cheaper (best)
    "more_acc_more_exp": "#1f77b4",   # blue   — better but pricier
    "less_acc_less_exp": "#ff7f0e",   # orange — worse but cheaper
    "less_acc_more_exp": "#d62728",   # red    — worse & pricier (worst)
}
QUADRANT_LABELS = {
    "more_acc_less_exp": "More accurate · less expensive",
    "more_acc_more_exp": "More accurate · more expensive",
    "less_acc_less_exp": "Less accurate · less expensive",
    "less_acc_more_exp": "Less accurate · more expensive",
}


def classify_quadrant(dx, dy):
    """dx = change in log cost, dy = change in logit accuracy."""
    acc = "more_acc" if dy >= 0 else "less_acc"
    exp = "more_exp" if dx >= 0 else "less_exp"
    return f"{acc}_{exp}"


# ─── Epoch-AI-inspired theme ─────────────────────────────────────────────────
# Light canvas, minimal spines, horizontal-only gridlines, left-aligned bold
# title with a gray subtitle, muted categorical palette, and a source footer.
EPOCH_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#C7C7CF",
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "xtick.color": "#5A5A66",
    "ytick.color": "#5A5A66",
    "text.color": "#16161D",
    "axes.labelcolor": "#3A3A44",
}
# Quadrant arrow colors tuned to Epoch's muted palette (semantics preserved).
EPOCH_QUADRANT_COLORS = {
    "more_acc_less_exp": "#119DA4",   # teal  — better & cheaper (best)
    "more_acc_more_exp": "#2E5EAA",   # blue  — better but pricier
    "less_acc_less_exp": "#E1A730",   # amber — worse but cheaper
    "less_acc_more_exp": "#C5283D",   # red   — worse & pricier (worst)
}
EPOCH_SCAFFOLD_PALETTE = ["#3C6E9F", "#E07A3F", "#5BA88B", "#9B6BB0", "#C0556B", "#6C8EAD"]
EPOCH_TEXT_DARK = "#16161D"
EPOCH_TEXT_GRAY = "#5A5A66"
EPOCH_SOURCE = "Source: HAL benchmark data  ·  chart styled after Epoch AI"


def slugify(s):
    """Lowercase, alphanumeric-only slug for filenames."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def place_vector_labels(fig, ax, labels, obstacles_xy, fontsize, fontweight="bold"):
    """Greedy non-overlapping placement of model-name labels for the vector plots.

    For each label we test a ring of candidate offsets around its anchor (the
    arrow midpoint) — straight up/down first, then out at increasing radius —
    and keep the first position whose text box collides with neither an
    already-placed label nor a data point. A thin leader line, colored to match
    the arrow, is drawn from any displaced label back to its arrow midpoint, so
    it is always clear which arrow a name belongs to.

    labels:       list of {x, y, text, color, leader} dicts (anchor in data coords).
    obstacles_xy: (x, y) data points to avoid sitting on (the arrow endpoints).
    """
    if not labels:
        return
    fig.canvas.draw()  # establish a renderer + final axes geometry
    renderer = fig.canvas.get_renderer()
    # Freeze limits so off-axes labels can't rescale the view mid-placement.
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())
    to_disp = ax.transData.transform

    # A semi-transparent white box behind each label keeps it readable wherever
    # it lands — even on top of an arrow — and gives the placer an honest box to
    # pack against.
    label_box = dict(boxstyle="round,pad=0.22", fc="white", ec="0.65",
                     lw=0.5, alpha=0.82)

    def measure(dx, dy, lb):
        t = ax.annotate(lb["text"], (lb["x"], lb["y"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fontsize, color=lb["color"],
                        fontweight=fontweight, ha="center", va="center",
                        zorder=7, bbox=label_box)
        bb = t.get_window_extent(renderer).expanded(1.12, 1.45)
        return t, bb

    # Keep-out boxes: every faint endpoint marker, plus the legend if present.
    obstacles = []
    for ox, oy in obstacles_xy:
        px, py = to_disp((ox, oy))
        obstacles.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))
    leg = ax.get_legend()
    if leg is not None:
        obstacles.append(leg.get_window_extent(renderer).expanded(1.04, 1.04))

    # Candidate offsets in typographic points: rings of growing radius, with the
    # natural reading spots (above / below) tried before the diagonals & sides.
    angles = [90, 270, 40, 140, 320, 220, 0, 180]
    radii = [16, 26, 38, 52, 70, 92, 118]
    candidates = [(0.0, 0.0)] + [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a)))
                                 for r in radii for a in angles]

    placed = []
    for lb in labels:
        chosen = None
        for dx, dy in candidates:
            t, bb = measure(dx, dy, lb)
            t.remove()
            if not any(bb.overlaps(p) for p in placed) and \
               not any(bb.overlaps(o) for o in obstacles):
                chosen = (dx, dy, bb)
                break
        if chosen is None:               # no clear spot — accept the farthest ring
            dx, dy = candidates[-1]
            bb = None
        else:
            dx, dy, bb = chosen
        # Draw the final label (with its background box) plus a leader line back
        # to the arrow midpoint whenever it was nudged away from it. The leader is
        # a neutral dark gray so it stays distinct from the thick colored arrows.
        leader = (dict(arrowstyle="-", color="#333333", lw=1.3, alpha=0.85,
                       shrinkA=3, shrinkB=5) if float(np.hypot(dx, dy)) >= 8 else None)
        t = ax.annotate(lb["text"], (lb["x"], lb["y"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fontsize, color=lb["color"],
                        fontweight=fontweight, ha="center", va="center",
                        zorder=7, bbox=label_box, arrowprops=leader)
        placed.append(bb if bb is not None else
                      t.get_window_extent(renderer).expanded(1.12, 1.45))


def render_vector_figure(fname, title, mcol, style="default", pair=None):
    """Render one scaffold-switch vector plot. style is 'default' or 'epoch'.

    If `pair` is given as a (scaffold_a, scaffold_b) tuple, only that single
    scaffold pair is drawn (one figure per pair); otherwise every pair is
    drawn on the same figure.
    """
    epoch = style == "epoch"
    quad_colors = EPOCH_QUADRANT_COLORS if epoch else QUADRANT_COLORS

    df = load_hal(fname, mcol)
    df["model_norm"] = df["Model"].apply(normalize_model_name)

    if pair is not None:
        scaffolds = [s for s in pair if s in set(df["scaffold"].dropna())]
    else:
        scaffolds = sorted(df["scaffold"].dropna().unique())
    if len(scaffolds) < 2:
        return

    fig_v, ax = plt.subplots(figsize=VEC_FIGSIZE)

    if epoch:
        scaffold_colors = {s: EPOCH_SCAFFOLD_PALETTE[i % len(EPOCH_SCAFFOLD_PALETTE)]
                           for i, s in enumerate(scaffolds)}
    else:
        scaffold_colors = panel_scaffold_colors(scaffolds)

    label_color = EPOCH_TEXT_DARK if epoch else "black"

    # For each scaffold pair, find shared models and draw vectors.
    # Arrows are colored by the quadrant they head toward (not by pair).
    used_quadrants = set()
    used_transitions = []  # (s1, s2) pairs that actually had arrows, in order
    vec_labels = []        # model-name labels, placed later without overlaps
    vec_points = []        # arrow endpoints, used as label keep-out obstacles

    for i in range(len(scaffolds)):
        for j in range(i + 1, len(scaffolds)):
            s1, s2 = scaffolds[i], scaffolds[j]
            df1 = df[df["scaffold"] == s1].copy()
            df2 = df[df["scaffold"] == s2].copy()

            # Find models present in both scaffolds
            shared_models = set(df1["model_norm"]) & set(df2["model_norm"])
            if not shared_models:
                continue

            used_transitions.append((s1, s2))

            for model in sorted(shared_models):
                rows1 = df1[df1["model_norm"] == model]
                rows2 = df2[df2["model_norm"] == model]
                # If multiple rows per scaffold (e.g. High vs non-High), take the best accuracy
                r1 = rows1.loc[rows1["accuracy"].idxmax()]
                r2 = rows2.loc[rows2["accuracy"].idxmax()]

                x1, y1 = np.log10(r1["cost"]), logit_score(r1["accuracy"])
                x2, y2 = np.log10(r2["cost"]), logit_score(r2["accuracy"])

                quad = classify_quadrant(x2 - x1, y2 - y1)
                ac = quad_colors[quad]
                used_quadrants.add(quad)

                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color=ac,
                                            lw=VEC_ARROW_LW, alpha=0.9 if epoch else 0.85,
                                            mutation_scale=VEC_ARROW_MUTATION,
                                            connectionstyle="arc3,rad=0.05"),
                            zorder=4)

                vec_points.append((x1, y1))
                vec_points.append((x2, y2))

                # Collect the label; actual placement happens after layout so we
                # can resolve overlaps and draw leader lines to the right arrow.
                if SHOW_VECTOR_LABELS:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    vec_labels.append({"x": mx, "y": my, "text": model,
                                       "color": label_color, "leader": ac})

    # Also scatter all raw points faintly for context
    for scaffold in scaffolds:
        grp = df[df["scaffold"] == scaffold]
        color = scaffold_colors[scaffold]
        ax.scatter(np.log10(grp["cost"]), logit_score(grp["accuracy"]),
                   c=[color], s=VEC_POINT_SIZE, alpha=VEC_POINT_ALPHA, zorder=2, marker="o",
                   edgecolors=POINT_EDGE_COLOR, linewidths=VEC_POINT_EDGE_WIDTH)

    ax.set_xlabel("Cost per task (log₁₀ USD)", fontsize=VEC_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (logit scale, shown as %)", fontsize=VEC_LABEL_FONTSIZE)

    # Build a subtitle describing the direction the arrows point.
    # (Helvetica Neue lacks the "→" glyph, so spell it out in the Epoch variant.)
    arrow_sep = " to " if epoch else " → "
    if used_transitions:
        transition_strs = [f"{s1}{arrow_sep}{s2}" for s1, s2 in used_transitions]
        subtitle = "arrows point: " + ";  ".join(transition_strs)
    else:
        subtitle = "arrows show same model moving between scaffolds"

    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.tick_params(axis="both", labelsize=VEC_TICK_FONTSIZE)

    # Custom x-axis ticks in dollars (vector x-axis is log10, so dollar value = 10**x)
    if COST_AXIS_IN_DOLLARS:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, p: f"${10**x:,.0f}" if 10**x >= 1 else f"${10**x:,.2f}"))

    # Legend keyed by quadrant direction (only those that appear)
    legend_order = ["more_acc_less_exp", "more_acc_more_exp",
                    "less_acc_less_exp", "less_acc_more_exp"]
    legend_handles = [
        plt.Line2D([], [], color=quad_colors[q], lw=6, label=QUADRANT_LABELS[q])
        for q in legend_order if q in used_quadrants
    ]

    if epoch:
        # Left-aligned bold title + gray subtitle, Epoch-style header.
        ax.set_title(f"{title}: Scaffold-Switch Vectors", loc="left", pad=38,
                     fontsize=VEC_TITLE_FONTSIZE, fontweight="bold", color=EPOCH_TEXT_DARK)
        ax.text(0.0, 1.018, subtitle, transform=ax.transAxes,
                fontsize=VEC_TITLE_FONTSIZE * 0.62, color=EPOCH_TEXT_GRAY, va="bottom")
        # Minimal spines, horizontal-only gridlines.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#E6E6EA", linewidth=1.0)
        ax.grid(False, axis="x")
        if legend_handles:
            leg = ax.legend(handles=legend_handles, fontsize=VEC_LEGEND_FONTSIZE,
                            loc="lower right", frameon=False,
                            title="Direction of switch", title_fontsize=VEC_LEGEND_TITLE_FONTSIZE)
            leg.get_title().set_color(EPOCH_TEXT_GRAY)
        # Source footer, bottom-left.
        fig_v.text(0.01, 0.005, EPOCH_SOURCE, fontsize=VEC_TICK_FONTSIZE * 0.7,
                   color=EPOCH_TEXT_GRAY, style="italic", ha="left", va="bottom")
        suffix = "epoch_"
    else:
        ax.set_title(f"{title}: Scaffold-Switch Vectors\n({subtitle})",
                     fontsize=VEC_TITLE_FONTSIZE, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=VEC_LEGEND_FONTSIZE, loc="lower right",
                      framealpha=0.9, title="Direction of switch",
                      title_fontsize=VEC_LEGEND_TITLE_FONTSIZE)
        suffix = ""

    fig_v.tight_layout()
    # Resolve label overlaps + draw leader lines once the layout is final.
    place_vector_labels(fig_v, ax, vec_labels, vec_points,
                        fontsize=VEC_ANNOT_FONTSIZE,
                        fontweight="bold" if not epoch else "medium")
    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    if pair is not None:
        pair_slug = f"__{slugify(scaffolds[0])}_vs_{slugify(scaffolds[1])}"
    else:
        pair_slug = ""
    fig_v.savefig(BASE / "figures" / f"hal_vectors_{suffix}{safe_name}{pair_slug}.png",
                  dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: hal_vectors_{suffix}{safe_name}{pair_slug}.png")
    plt.close(fig_v)


def scaffold_pairs_with_shared_models(fname, mcol):
    """Return all (s1, s2) scaffold pairs that share at least one model."""
    df = load_hal(fname, mcol)
    df["model_norm"] = df["Model"].apply(normalize_model_name)
    scaffolds = sorted(df["scaffold"].dropna().unique())
    pairs = []
    for i in range(len(scaffolds)):
        for j in range(i + 1, len(scaffolds)):
            s1, s2 = scaffolds[i], scaffolds[j]
            m1 = set(df[df["scaffold"] == s1]["model_norm"])
            m2 = set(df[df["scaffold"] == s2]["model_norm"])
            if m1 & m2:
                pairs.append((s1, s2))
    return pairs


for fname, title, mcol in BENCHMARKS:
    for pair in scaffold_pairs_with_shared_models(fname, mcol):
        render_vector_figure(fname, title, mcol, style="default", pair=pair)
        if EPOCH_STYLE:
            with plt.rc_context(EPOCH_RC):
                render_vector_figure(fname, title, mcol, style="epoch", pair=pair)


# ─── Figure 4: Origin-centered scaffold-switch vectors ───────────────────────
# Same data as the scaffold-switch arrows, but every arrow is translated so it
# starts at the origin. (0, 0) is each model's price/accuracy on the *generalist*
# scaffold; the arrow points to (Δlog cost, Δlogit accuracy) after switching to a
# specialist scaffold. This collapses every benchmark onto a common reference so
# the *direction and magnitude* of the scaffold switch is directly comparable.

def get_reference_scaffold(df):
    """Pick the 'generalist' scaffold to anchor at the origin.

    Prefers a scaffold whose name contains 'generalist'; falls back to a generic
    browser agent ('Browser-Use') for benchmarks that lack a generalist entry.
    """
    scaffolds = list(df["scaffold"].dropna().unique())
    for s in scaffolds:
        if "generalist" in s.lower():
            return s
    for s in scaffolds:
        if "browser-use" in s.lower():
            return s
    return None


def origin_vectors(fname, mcol):
    """Compute origin-anchored switch vectors for one benchmark.

    Returns (ref_scaffold, list_of_vectors). Each vector is a dict with
    dx (Δlog10 cost), dy (Δlogit accuracy), quadrant, target scaffold, and model.
    """
    df = load_hal(fname, mcol)
    df["model_norm"] = df["Model"].apply(normalize_model_name)
    ref = get_reference_scaffold(df)
    if ref is None:
        return None, []

    ref_df = df[df["scaffold"] == ref]
    # Best-accuracy generalist point per model = origin reference.
    refmap = {}
    for model, rows in ref_df.groupby("model_norm"):
        r = rows.loc[rows["accuracy"].idxmax()]
        refmap[model] = (np.log10(r["cost"]), logit_score(r["accuracy"]))

    vectors = []
    for s in sorted(df["scaffold"].dropna().unique()):
        if s == ref:
            continue
        sdf = df[df["scaffold"] == s]
        for model, rows in sdf.groupby("model_norm"):
            if model not in refmap:
                continue
            r = rows.loc[rows["accuracy"].idxmax()]
            x0, y0 = refmap[model]
            dx = np.log10(r["cost"]) - x0
            dy = logit_score(r["accuracy"]) - y0
            vectors.append({"dx": dx, "dy": dy, "quad": classify_quadrant(dx, dy),
                            "target": s, "model": model})
    return ref, vectors


def make_cost_mult_formatter():
    """Format an x value (= Δlog10 cost) as a cost multiplier, e.g. 0.3→'×2'."""
    def fmt(x, pos):
        m = 10 ** x
        if m >= 1:
            return f"×{m:.2g}"
        return f"×{m:.2g}"
    return mticker.FuncFormatter(fmt)


def style_origin_axes(ax, vectors, label_fs, tick_fs):
    """Crosshairs at the origin, faint quadrant tints, symmetric framing."""
    ax.axhline(0, color="#555", lw=1.4, zorder=1)
    ax.axvline(0, color="#555", lw=1.4, zorder=1)
    ax.scatter([0], [0], c="black", s=70, zorder=6, marker="o")

    if vectors:
        max_x = max(abs(v["dx"]) for v in vectors)
        max_y = max(abs(v["dy"]) for v in vectors)
    else:
        max_x = max_y = 1.0
    px = max(max_x * 1.18, 0.1)
    py = max(max_y * 1.18, 0.1)
    ax.set_xlim(-px, px)
    ax.set_ylim(-py, py)

    # Faint quadrant tints matching the arrow color semantics.
    tint = 0.06
    ax.axhspan(0, py, xmin=0.5, xmax=1.0, color=QUADRANT_COLORS["more_acc_more_exp"], alpha=tint, zorder=0)
    ax.axhspan(0, py, xmin=0.0, xmax=0.5, color=QUADRANT_COLORS["more_acc_less_exp"], alpha=tint, zorder=0)
    ax.axhspan(-py, 0, xmin=0.5, xmax=1.0, color=QUADRANT_COLORS["less_acc_more_exp"], alpha=tint, zorder=0)
    ax.axhspan(-py, 0, xmin=0.0, xmax=0.5, color=QUADRANT_COLORS["less_acc_less_exp"], alpha=tint, zorder=0)

    ax.xaxis.set_major_formatter(make_cost_mult_formatter())
    ax.set_xlabel("Cost change vs. generalist (×, log scale)", fontsize=label_fs)
    ax.set_ylabel("Accuracy change vs. generalist (Δ logit)", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)


# Grid of origin-centered vector plots, one panel per benchmark.
fig_o, axes_o = plt.subplots(4, 2, figsize=GRID_FIGSIZE)
fig_o.suptitle("HAL Benchmarks: Origin-Centered Scaffold-Switch Vectors\n"
               "(origin = each model on the generalist scaffold; arrow = switch to a specialist scaffold)",
               fontsize=GRID_SUPTITLE_FONTSIZE * 0.85, fontweight="bold", y=0.997)

used_quadrants_all = set()
for idx, (fname, title, mcol) in enumerate(BENCHMARKS):
    ax = axes_o.flat[idx]
    ref, vectors = origin_vectors(fname, mcol)
    for v in vectors:
        ac = QUADRANT_COLORS[v["quad"]]
        used_quadrants_all.add(v["quad"])
        ax.annotate("", xy=(v["dx"], v["dy"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ac, lw=GRID_LINEWIDTH + 0.5,
                                    alpha=0.8, mutation_scale=22,
                                    connectionstyle="arc3,rad=0.04"), zorder=4)
    style_origin_axes(ax, vectors, GRID_LABEL_FONTSIZE, GRID_TICK_FONTSIZE)
    ref_note = f"\n(origin = {ref})" if ref else ""
    ax.set_title(f"{title}{ref_note}", fontsize=GRID_TITLE_FONTSIZE * 0.8, fontweight="bold")
    ax.grid(True, alpha=0.25)

legend_order = ["more_acc_less_exp", "more_acc_more_exp", "less_acc_less_exp", "less_acc_more_exp"]
legend_handles = [plt.Line2D([], [], color=QUADRANT_COLORS[q], lw=6, label=QUADRANT_LABELS[q])
                  for q in legend_order if q in used_quadrants_all]
if legend_handles:
    fig_o.legend(handles=legend_handles, fontsize=GRID_LEGEND_FONTSIZE, loc="lower center",
                 ncol=2, framealpha=0.9, title="Direction of switch",
                 title_fontsize=GRID_LEGEND_FONTSIZE)
plt.tight_layout(rect=[0, 0.03, 1, 0.975])
fig_o.savefig(BASE / "figures" / "hal_origin_vectors_grid.png", dpi=SAVE_DPI, bbox_inches="tight")
fig_o.savefig(BASE / "figures" / "hal_origin_vectors_grid.pdf", bbox_inches="tight")
print("Saved: hal_origin_vectors_grid.png/pdf")
plt.close(fig_o)

# Individual larger origin-centered figures per benchmark, with model labels.
for fname, title, mcol in BENCHMARKS:
    ref, vectors = origin_vectors(fname, mcol)
    if not vectors:
        continue
    fig_i, ax = plt.subplots(figsize=VEC_FIGSIZE)
    used_q = set()
    vec_labels, vec_points = [], []
    for v in vectors:
        ac = QUADRANT_COLORS[v["quad"]]
        used_q.add(v["quad"])
        ax.annotate("", xy=(v["dx"], v["dy"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ac, lw=VEC_ARROW_LW * 0.7,
                                    alpha=0.85, mutation_scale=VEC_ARROW_MUTATION * 0.7,
                                    connectionstyle="arc3,rad=0.04"), zorder=4)
        vec_points.append((v["dx"], v["dy"]))
        if SHOW_VECTOR_LABELS:
            vec_labels.append({"x": v["dx"], "y": v["dy"], "text": v["model"],
                               "color": "black", "leader": ac})
    style_origin_axes(ax, vectors, VEC_LABEL_FONTSIZE, VEC_TICK_FONTSIZE)
    targets = sorted({v["target"] for v in vectors})
    subtitle = f"origin = {ref};  arrows → " + ", ".join(targets)
    ax.set_title(f"{title}: Origin-Centered Scaffold-Switch Vectors\n({subtitle})",
                 fontsize=VEC_TITLE_FONTSIZE * 0.85, fontweight="bold")
    ax.grid(True, alpha=0.25)
    leg_handles = [plt.Line2D([], [], color=QUADRANT_COLORS[q], lw=6, label=QUADRANT_LABELS[q])
                   for q in legend_order if q in used_q]
    if leg_handles:
        ax.legend(handles=leg_handles, fontsize=VEC_LEGEND_FONTSIZE, loc="best",
                  framealpha=0.9, title="Direction of switch",
                  title_fontsize=VEC_LEGEND_TITLE_FONTSIZE)
    fig_i.tight_layout()
    place_vector_labels(fig_i, ax, vec_labels, vec_points, fontsize=VEC_ANNOT_FONTSIZE)
    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    fig_i.savefig(BASE / "figures" / f"hal_origin_vectors_{safe_name}.png",
                  dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: hal_origin_vectors_{safe_name}.png")
    plt.close(fig_i)


# ─── Figure 5: One origin-centered panel per scaffold switch ──────────────────
# Like Figure 4, but each panel shows exactly ONE scaffold switch (one benchmark,
# one reference→target pair) and its title names that switch. Mind2Web is excluded
# (it has no generalist reference scaffold).

switch_panels = []  # (benchmark_title, ref_scaffold, target_scaffold, vectors)
for fname, title, mcol in BENCHMARKS:
    if "mind2web" in title.lower().replace(" ", ""):
        continue
    ref, vectors = origin_vectors(fname, mcol)
    if not vectors:
        continue
    for target in sorted({v["target"] for v in vectors}):
        tv = [v for v in vectors if v["target"] == target]
        switch_panels.append((title, ref, target, tv))

n_panels = len(switch_panels)
ncols = 3
nrows = int(np.ceil(n_panels / ncols))
fig_s, axes_s = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)
fig_s.suptitle("Origin-Centered Scaffold Switches\n"
               "(origin = each model on the generalist scaffold; arrow = switch to the named scaffold)",
               fontsize=GRID_SUPTITLE_FONTSIZE * 0.85, fontweight="bold", y=0.998)

used_quadrants_switch = set()
for idx, (title, ref, target, vectors) in enumerate(switch_panels):
    ax = axes_s.flat[idx]
    # A "stub" is a near-zero-length switch (e.g. a model whose accuracy is
    # unchanged and cost barely moves) whose arrow would otherwise be hidden
    # under the origin dot. We keep stubs TRUE-TO-SCALE — just drawn above the
    # origin dot with an endpoint marker so the tiny arrow is visible without
    # exaggerating its magnitude. Every other arrow is left exactly as-is.
    panel_max_len = max((np.hypot(v["dx"], v["dy"]) for v in vectors), default=0.0)
    stub_thresh = 0.06 * panel_max_len
    for v in vectors:
        ac = QUADRANT_COLORS[v["quad"]]
        used_quadrants_switch.add(v["quad"])
        vlen = np.hypot(v["dx"], v["dy"])
        is_stub = panel_max_len > 0 and 0 < vlen < stub_thresh
        # Lift stubs above the origin dot (zorder 6) so they aren't occluded.
        z = 8 if is_stub else 4
        ax.annotate("", xy=(v["dx"], v["dy"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ac, lw=GRID_LINEWIDTH + 0.5,
                                    alpha=0.8, mutation_scale=27,
                                    connectionstyle="arc3,rad=0.04"), zorder=z)
        if is_stub:
            ax.scatter([v["dx"]], [v["dy"]], c=[ac], s=22, zorder=9,
                       edgecolors="white", linewidths=0.6)
    style_origin_axes(ax, vectors, GRID_LABEL_FONTSIZE * 0.85, GRID_TICK_FONTSIZE * 0.85)
    ax.set_title(f"{title}\n{ref} → {target}",
                 fontsize=GRID_TITLE_FONTSIZE * 0.7, fontweight="bold")
    ax.grid(True, alpha=0.25)

for idx in range(n_panels, nrows * ncols):
    axes_s.flat[idx].set_visible(False)

switch_legend_handles = [plt.Line2D([], [], color=QUADRANT_COLORS[q], lw=7, label=QUADRANT_LABELS[q])
                         for q in legend_order if q in used_quadrants_switch]
# Reserve a strip at the bottom for the legend, then anchor it inside that strip
# (below the bottom row of panels) so it never overlaps their x-axis labels.
plt.tight_layout(rect=[0, 0.075, 1, 0.975])
if switch_legend_handles:
    fig_s.legend(handles=switch_legend_handles, fontsize=GRID_LEGEND_FONTSIZE * 1.25,
                 loc="upper center", bbox_to_anchor=(0.5, 0.065), ncol=2, framealpha=0.9,
                 title="Direction of switch", title_fontsize=GRID_LEGEND_FONTSIZE * 1.35,
                 handlelength=2.2, columnspacing=2.0, borderpad=0.7, labelspacing=0.6)
fig_s.savefig(BASE / "figures" / "hal_origin_vectors_by_switch_grid.png", dpi=SAVE_DPI, bbox_inches="tight")
fig_s.savefig(BASE / "figures" / "hal_origin_vectors_by_switch_grid.pdf", bbox_inches="tight")
print(f"Saved: hal_origin_vectors_by_switch_grid.png/pdf ({n_panels} switch panels)")
plt.close(fig_s)


# ─── Figure 6: Mean switch arrow + confidence ellipse, all benchmarks overlaid ─
# One plot. For each benchmark we pool its origin-centered switch vectors (origin
# = generalist scaffold) and draw (a) a mean arrow from the origin to the centroid
# of the endpoints — "the typical switch" — and (b) an ellipse summarizing the
# cloud. Mind2Web is excluded (no generalist reference scaffold).
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# "ci" = 95% confidence ellipse on the mean (shrinks with n; "is the avg switch real?")
# "sd" = 1 standard-deviation spread of models ("how consistent is the switch?")
ELLIPSE_MODE = "ci"
CI_LEVEL = 0.95


def ellipse_params(points, mode=ELLIPSE_MODE, ci_level=CI_LEVEL):
    """Return (mean, (width, height, angle_deg)) for a covariance ellipse.

    Ellipse is None when there are too few points to estimate a covariance.
    """
    pts = np.asarray(points, dtype=float)
    mean = pts.mean(axis=0)
    if len(pts) < 3:
        return mean, None
    cov = np.cov(pts.T)
    if mode == "ci":
        cov = cov / len(pts)                       # covariance of the mean
        scale = np.sqrt(chi2.ppf(ci_level, df=2))  # 95% region ≈ 2.45 σ in 2-D
    else:                                          # "sd"
        scale = 1.0
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * scale * np.sqrt(np.clip(vals, 0, None))
    return mean, (width, height, angle)


overlay_benchmarks = [(f, t, m) for (f, t, m) in BENCHMARKS
                      if "mind2web" not in t.lower().replace(" ", "")]

fig_e, ax = plt.subplots(figsize=VEC_FIGSIZE)
cmap = plt.cm.tab10

# First pass: gather each benchmark's mean/ellipse/sample count.
entries = []  # (title, color, mean, ell, n, pts)
all_xy = []
for i, (fname, title, mcol) in enumerate(overlay_benchmarks):
    ref, vectors = origin_vectors(fname, mcol)
    if not vectors:
        continue
    pts = [(v["dx"], v["dy"]) for v in vectors]
    all_xy.extend(pts)
    mean, ell = ellipse_params(pts)
    entries.append((title, cmap(i % 10), mean, ell, len(pts), pts))

# Map sample count → ellipse fill opacity (more samples ⇒ darker blob).
ELLIPSE_ALPHA_RANGE = (0.10, 0.50)
ns = [n for *_, n, _ in entries]
n_min, n_max = min(ns), max(ns)
def alpha_for_n(n):
    if n_max == n_min:
        return ELLIPSE_ALPHA_RANGE[1]
    return float(np.interp(n, [n_min, n_max], ELLIPSE_ALPHA_RANGE))

for title, color, mean, ell, n, pts in entries:
    if ell is not None:
        w, h, ang = ell
        ax.add_patch(Ellipse(mean, w, h, angle=ang, facecolor=color, alpha=alpha_for_n(n),
                             edgecolor=color, lw=2.0, zorder=3))
    ax.annotate("", xy=(mean[0], mean[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=4.0,
                                mutation_scale=28, alpha=0.95,
                                connectionstyle="arc3,rad=0.0"), zorder=5)
    ax.scatter([mean[0]], [mean[1]], color=color, s=70, zorder=6,
               edgecolors="white", linewidths=1.0, label=f"{title} (n={n})")

# Symmetric framing around the origin from the pooled endpoints.
px = max(max(abs(x) for x, _ in all_xy) * 1.25, 0.1)
py = max(max(abs(y) for _, y in all_xy) * 1.25, 0.1)
ax.set_xlim(-px, px)
ax.set_ylim(-py, py)

# Faint quadrant tints + origin crosshairs (same semantics as the per-benchmark plots).
tint = 0.05
ax.axhspan(0, py, xmin=0.5, xmax=1.0, color=QUADRANT_COLORS["more_acc_more_exp"], alpha=tint, zorder=0)
ax.axhspan(0, py, xmin=0.0, xmax=0.5, color=QUADRANT_COLORS["more_acc_less_exp"], alpha=tint, zorder=0)
ax.axhspan(-py, 0, xmin=0.5, xmax=1.0, color=QUADRANT_COLORS["less_acc_more_exp"], alpha=tint, zorder=0)
ax.axhspan(-py, 0, xmin=0.0, xmax=0.5, color=QUADRANT_COLORS["less_acc_less_exp"], alpha=tint, zorder=0)
ax.axhline(0, color="#555", lw=1.4, zorder=1)
ax.axvline(0, color="#555", lw=1.4, zorder=1)
ax.scatter([0], [0], c="black", s=80, zorder=7, marker="o")

ax.xaxis.set_major_formatter(make_cost_mult_formatter())
ax.set_xlabel("Cost change vs. generalist (×, log scale)", fontsize=VEC_LABEL_FONTSIZE)
ax.set_ylabel("Accuracy change vs. generalist (Δ logit)", fontsize=VEC_LABEL_FONTSIZE)
ax.tick_params(axis="both", labelsize=VEC_TICK_FONTSIZE)
ax.grid(True, alpha=0.25)
ellipse_desc = ("95% confidence ellipse on the mean" if ELLIPSE_MODE == "ci"
                else "1 SD spread across models")
ax.set_title("Mean Scaffold-Switch Vector by Benchmark\n"
             f"(origin = generalist scaffold; arrow = mean switch; ellipse = {ellipse_desc})",
             fontsize=VEC_TITLE_FONTSIZE * 0.8, fontweight="bold")
ax.legend(fontsize=VEC_LEGEND_FONTSIZE * 0.8, loc="best", framealpha=0.9, title="Benchmark",
          title_fontsize=VEC_LEGEND_TITLE_FONTSIZE * 0.8)
fig_e.tight_layout()
fig_e.savefig(BASE / "figures" / "hal_origin_vectors_mean_ellipse_overlay.png",
              dpi=SAVE_DPI, bbox_inches="tight")
fig_e.savefig(BASE / "figures" / "hal_origin_vectors_mean_ellipse_overlay.pdf", bbox_inches="tight")
print("Saved: hal_origin_vectors_mean_ellipse_overlay.png/pdf")
plt.close(fig_e)


# ─── Figure 7: Spider/radar — generalist vs. best specialist scaffold ─────────
# One axis per benchmark. Inner web = max accuracy reachable on the HAL generalist
# scaffold; outer web = max accuracy reachable on the best specialist scaffold.
# The gap between the two webs is the capability headroom that scaffolding unlocks.
# Mind2Web is excluded (no generalist scaffold).

spider_benchmarks = [(f, t, m) for (f, t, m) in BENCHMARKS
                     if "mind2web" not in t.lower().replace(" ", "")]

spider_labels, gen_vals, spec_vals = [], [], []
for fname, title, mcol in spider_benchmarks:
    df = load_hal(fname, mcol)
    gen = get_reference_scaffold(df)
    spider_labels.append(title)
    gen_vals.append(float(df[df["scaffold"] == gen]["accuracy"].max()))
    spec_vals.append(float(df[df["scaffold"] != gen]["accuracy"].max()))

N = len(spider_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
ang_c = angles + angles[:1]                       # close the loop
gen_c = gen_vals + gen_vals[:1]
spec_c = spec_vals + spec_vals[:1]

GEN_COLOR = "#1f77b4"     # blue  — generalist scaffold
SPEC_COLOR = "#2ca02c"    # green — best specialist scaffold

fig_r = plt.figure(figsize=(13, 13))
ax = fig_r.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)   # first axis at top
ax.set_theta_direction(-1)       # clockwise

# Outer (specialist) web first, then generalist on top.
ax.plot(ang_c, spec_c, color=SPEC_COLOR, lw=3.5, zorder=4,
        label="Best specialist scaffold (max accuracy)")
ax.fill(ang_c, spec_c, color=SPEC_COLOR, alpha=0.15, zorder=2)
ax.plot(ang_c, gen_c, color=GEN_COLOR, lw=3.5, zorder=5,
        label="HAL generalist scaffold (max accuracy)")
ax.fill(ang_c, gen_c, color=GEN_COLOR, alpha=0.22, zorder=3)

# Markers + value labels at each vertex.
for ang, gv, sv in zip(angles, gen_vals, spec_vals):
    ax.scatter(ang, sv, color=SPEC_COLOR, s=70, zorder=6)
    ax.scatter(ang, gv, color=GEN_COLOR, s=70, zorder=7)
    ax.annotate(f"{sv:.0f}%", (ang, sv), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=13, fontweight="bold", color=SPEC_COLOR, zorder=8)
    ax.annotate(f"{gv:.0f}%", (ang, gv), textcoords="offset points", xytext=(0, -13),
                ha="center", fontsize=13, fontweight="bold", color=GEN_COLOR, zorder=8)

ax.set_xticks(angles)
ax.set_xticklabels(spider_labels, fontsize=16, fontweight="bold")
ax.tick_params(axis="x", pad=22)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=12, color="#666")
ax.grid(True, alpha=0.4)
ax.set_title("Scaffolding Headroom by Benchmark\n"
             "(max accuracy: HAL generalist scaffold vs. best specialist scaffold)",
             fontsize=22, fontweight="bold", pad=42)
ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), fontsize=15, framealpha=0.9)

fig_r.tight_layout()
fig_r.savefig(BASE / "figures" / "hal_spider_generalist_vs_specialist.png",
              dpi=SAVE_DPI, bbox_inches="tight")
fig_r.savefig(BASE / "figures" / "hal_spider_generalist_vs_specialist.pdf", bbox_inches="tight")
print("Saved: hal_spider_generalist_vs_specialist.png/pdf")
plt.close(fig_r)

plt.show()
