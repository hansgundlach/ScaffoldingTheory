"""
Price vs Performance (logit-scaled) frontier plots from HAL benchmark data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
VEC_ANNOT_FONTSIZE = 15
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

fig, axes = plt.subplots(4, 2, figsize=GRID_FIGSIZE)
fig.suptitle("HAL Benchmarks: Per-Scaffold Pareto Frontiers\n(y-axis = logit of accuracy, x-axis = log cost)",
             fontsize=GRID_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995)

for idx, (fname, title, mcol) in enumerate(BENCHMARKS):
    ax = axes.flat[idx]
    df = load_hal(fname, mcol)

    scaffolds = sorted(df["scaffold"].unique())
    cmap = plt.cm.tab10
    scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

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
    cmap = plt.cm.tab10
    scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

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
        cmap = plt.cm.tab10
        scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

    label_color = EPOCH_TEXT_DARK if epoch else "black"

    # For each scaffold pair, find shared models and draw vectors.
    # Arrows are colored by the quadrant they head toward (not by pair).
    used_quadrants = set()
    used_transitions = []  # (s1, s2) pairs that actually had arrows, in order

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

                # Label at midpoint
                if SHOW_VECTOR_LABELS:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.annotate(model, (mx, my),
                                fontsize=VEC_ANNOT_FONTSIZE, color=label_color, alpha=0.95,
                                fontweight="bold" if not epoch else "medium",
                                ha="center", va="bottom",
                                textcoords="offset points", xytext=(0, 5),
                                zorder=5)

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

plt.show()
