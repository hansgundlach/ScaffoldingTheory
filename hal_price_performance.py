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


# Define which benchmarks to plot and their configs
BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("usaco.csv", "USACO", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle("HAL Benchmarks: Per-Scaffold Pareto Frontiers\n(y-axis = logit of accuracy, x-axis = log cost)",
             fontsize=15, fontweight="bold")

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
        ax.scatter(grp["cost"], logit_acc, c=[color], s=50, alpha=0.6,
                   marker=marker, edgecolors="white", linewidths=0.3, zorder=3)

        # Label points with model name
        for _, row in grp.iterrows():
            model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
            model_short = model_short.replace("Claude ", "C.").replace("Sonnet", "Son").replace("Opus", "Op")
            model_short = model_short.replace("GPT-", "G").replace("DeepSeek", "DS")
            model_short = model_short.replace("Gemini", "Gem").replace("High", "H")
            ax.annotate(model_short, (row["cost"], logit_score(row["accuracy"])),
                        textcoords="offset points", xytext=(4, 3), fontsize=5,
                        alpha=0.7, color=color)

        # Per-scaffold Pareto frontier
        pareto = compute_pareto(grp)
        if len(pareto) >= 2:
            pareto_sorted = pareto.sort_values("cost")
            ax.step(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                    where="post", color=color, linewidth=2, linestyle=ls,
                    alpha=0.8, zorder=4, label=scaffold)
        else:
            # Still add to legend even with 1 point
            ax.scatter([], [], c=[color], marker=marker, s=50, label=scaffold)

    ax.set_xscale("log")
    ax.set_xlabel("Cost (USD, log scale)")
    ax.set_ylabel("Accuracy (logit scale)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right",
              framealpha=0.8)

plt.tight_layout()
plt.savefig(BASE / "figures" / "hal_price_performance_frontier.png", dpi=150, bbox_inches="tight")
plt.savefig(BASE / "figures" / "hal_price_performance_frontier.pdf", bbox_inches="tight")
print("Saved: hal_price_performance_frontier.png/pdf")

# ─── Individual larger figures per benchmark ─────────────────────────────────
for fname, title, mcol in BENCHMARKS:
    fig_ind, ax = plt.subplots(figsize=(12, 7))
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

        ax.scatter(grp["cost"], logit_acc, c=[color], s=70, alpha=0.65,
                   marker=marker, edgecolors="white", linewidths=0.4, zorder=3)

        for _, row in grp.iterrows():
            model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
            ax.annotate(model_short, (row["cost"], logit_score(row["accuracy"])),
                        textcoords="offset points", xytext=(5, 4), fontsize=7,
                        alpha=0.8, color=color)

        # Per-scaffold Pareto frontier
        pareto = compute_pareto(grp)
        if len(pareto) >= 2:
            pareto_sorted = pareto.sort_values("cost")
            ax.step(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                    where="post", color=color, linewidth=2.5, linestyle=ls,
                    alpha=0.85, zorder=4, label=f"{scaffold} frontier")
            ax.scatter(pareto_sorted["cost"], logit_score(pareto_sorted["accuracy"]),
                       marker=marker, s=100, facecolors="none", edgecolors=color,
                       linewidths=2, zorder=5)
        else:
            ax.scatter([], [], c=[color], marker=marker, s=70, label=scaffold)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (USD, log scale)", fontsize=12)
    ax.set_ylabel("Accuracy (logit scale, shown as %)", fontsize=12)
    ax.set_title(f"{title}: Per-Scaffold Price-Performance Frontier", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower right",
              framealpha=0.9)

    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    fig_ind.tight_layout()
    fig_ind.savefig(BASE / "figures" / f"hal_frontier_{safe_name}.png", dpi=150, bbox_inches="tight")
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


for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    df["model_norm"] = df["Model"].apply(normalize_model_name)

    scaffolds = sorted(df["scaffold"].dropna().unique())
    if len(scaffolds) < 2:
        continue

    fig_v, ax = plt.subplots(figsize=(13, 8))

    cmap = plt.cm.tab10
    scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

    # For each scaffold pair, find shared models and draw vectors
    # We'll draw from scaffold i to scaffold j for all i < j
    pair_idx = 0
    arrow_colors = plt.cm.Dark2(np.linspace(0, 1, 8))

    for i in range(len(scaffolds)):
        for j in range(i + 1, len(scaffolds)):
            s1, s2 = scaffolds[i], scaffolds[j]
            df1 = df[df["scaffold"] == s1].copy()
            df2 = df[df["scaffold"] == s2].copy()

            # Find models present in both scaffolds
            shared_models = set(df1["model_norm"]) & set(df2["model_norm"])
            if not shared_models:
                continue

            ac = arrow_colors[pair_idx % len(arrow_colors)]
            pair_idx += 1

            for model in sorted(shared_models):
                rows1 = df1[df1["model_norm"] == model]
                rows2 = df2[df2["model_norm"] == model]
                # If multiple rows per scaffold (e.g. High vs non-High), take the best accuracy
                r1 = rows1.loc[rows1["accuracy"].idxmax()]
                r2 = rows2.loc[rows2["accuracy"].idxmax()]

                x1, y1 = np.log10(r1["cost"]), logit_score(r1["accuracy"])
                x2, y2 = np.log10(r2["cost"]), logit_score(r2["accuracy"])

                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color=ac,
                                            lw=1.8, alpha=0.7,
                                            connectionstyle="arc3,rad=0.05"))

                # Label at midpoint
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(model, (mx, my),
                            fontsize=6.5, color="black", alpha=0.8,
                            ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3))

            # Invisible scatter for legend
            ax.plot([], [], color=ac, lw=2, label=f"{s1} → {s2}")

    # Also scatter all raw points faintly for context
    for scaffold in scaffolds:
        grp = df[df["scaffold"] == scaffold]
        color = scaffold_colors[scaffold]
        ax.scatter(np.log10(grp["cost"]), logit_score(grp["accuracy"]),
                   c=[color], s=40, alpha=0.3, zorder=2, marker="o",
                   edgecolors="none")

    ax.set_xlabel("Cost per task (log₁₀ USD)", fontsize=12)
    ax.set_ylabel("Accuracy (logit scale, shown as %)", fontsize=12)
    ax.set_title(f"{title}: Scaffold-Switch Vectors\n(arrows show same model moving between scaffolds)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(make_pct_formatter())

    # Custom x-axis ticks in dollars
    xticks = ax.get_xticks()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${10**x:.0f}" if 10**x >= 1 else f"${10**x:.2f}"))

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9,
              title="Scaffold transition", title_fontsize=10)

    fig_v.tight_layout()
    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    fig_v.savefig(BASE / "figures" / f"hal_vectors_{safe_name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: hal_vectors_{safe_name}.png")
    plt.close(fig_v)

plt.show()
