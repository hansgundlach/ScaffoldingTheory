"""
Rank-order preservation across scaffolds.

For each benchmark with 2+ scaffolds, show bump charts of how model rankings
change when switching scaffolds, plus compute rank correlation statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

BASE = Path(__file__).parent
HAL = BASE / "hal_data"


def parse_pct(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)%", str(s))
    return float(m.group(1)) if m else np.nan


def parse_cost(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "")
    m = re.search(r"\$?([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


def clean_scaffold(s):
    if pd.isna(s):
        return s
    return re.sub(r"\s*Pareto optimal\s*", "", str(s)).strip()


def normalize_model(s):
    """Normalize model name: strip 'High' and date parentheticals."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = re.sub(r"\s+High\b", "", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    return s.strip()


def load_hal(filename, model_col="Primary Model"):
    df = pd.read_csv(HAL / filename, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Models" in df.columns and model_col not in df.columns:
        model_col = "Models"
    df = df.rename(columns={model_col: "Model"})
    df["accuracy"] = df["Accuracy"].apply(parse_pct)
    df["cost"] = df["Cost (USD)"].apply(parse_cost)
    df["scaffold"] = df["Scaffold"].apply(clean_scaffold)
    df["model_norm"] = df["Model"].apply(normalize_model)
    df = df.dropna(subset=["accuracy", "model_norm"])
    # Take best accuracy per (scaffold, model_norm) — collapses High/non-High
    df = df.sort_values("accuracy", ascending=False).drop_duplicates(
        ["scaffold", "model_norm"], keep="first"
    )
    return df


BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("scicode.csv", "SciCode", "Primary Model"),
    ("online_mine_2_web.csv", "Online Mind2Web", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
]

# Model colors — consistent across plots
MODEL_COLORS = {}
ALL_MODELS = set()
for fname, _, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    ALL_MODELS.update(df["model_norm"].unique())
ALL_MODELS = sorted(ALL_MODELS)
cmap = plt.cm.tab20(np.linspace(0, 1, 20))
for i, m in enumerate(ALL_MODELS):
    MODEL_COLORS[m] = cmap[i % 20]

# Collect stats for summary
stats_rows = []

# ─── Per-benchmark bump charts ──────────────────────────────────────────────
for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    scaffolds = sorted(df["scaffold"].dropna().unique())
    if len(scaffolds) < 2:
        continue

    # For each scaffold pair with enough shared models
    for si in range(len(scaffolds)):
        for sj in range(si + 1, len(scaffolds)):
            s1, s2 = scaffolds[si], scaffolds[sj]
            df1 = df[df["scaffold"] == s1][["model_norm", "accuracy"]].copy()
            df2 = df[df["scaffold"] == s2][["model_norm", "accuracy"]].copy()

            shared = sorted(set(df1["model_norm"]) & set(df2["model_norm"]))
            if len(shared) < 4:
                continue

            df1 = df1[df1["model_norm"].isin(shared)].set_index("model_norm")
            df2 = df2[df2["model_norm"].isin(shared)].set_index("model_norm")

            # Rank: 1 = best
            df1["rank"] = df1["accuracy"].rank(ascending=False, method="min").astype(int)
            df2["rank"] = df2["accuracy"].rank(ascending=False, method="min").astype(int)

            n = len(shared)
            rho, rho_p = spearmanr(df1.loc[shared, "rank"], df2.loc[shared, "rank"])
            tau, tau_p = kendalltau(df1.loc[shared, "rank"], df2.loc[shared, "rank"])

            stats_rows.append({
                "benchmark": title, "scaffold_1": s1, "scaffold_2": s2,
                "n_shared": n, "spearman_rho": rho, "spearman_p": rho_p,
                "kendall_tau": tau, "kendall_p": tau_p,
            })

            # ── Bump chart ──
            fig, ax = plt.subplots(figsize=(8, max(5, n * 0.55)))

            x_left, x_right = 0, 1
            rank_inversions = 0

            for model in shared:
                r1 = df1.loc[model, "rank"]
                r2 = df2.loc[model, "rank"]
                color = MODEL_COLORS.get(model, "gray")

                if r1 != r2:
                    rank_inversions += 1

                # Draw connecting line
                # Color green if improved rank, red if worsened, gray if same
                if r2 < r1:
                    line_color = "tab:green"
                    line_alpha = 0.7
                elif r2 > r1:
                    line_color = "tab:red"
                    line_alpha = 0.7
                else:
                    line_color = "gray"
                    line_alpha = 0.4

                ax.plot([x_left, x_right], [r1, r2], color=line_color,
                        linewidth=2.5, alpha=line_alpha, zorder=2)

                # Dots at endpoints
                ax.scatter(x_left, r1, s=80, c=[color], zorder=3,
                           edgecolors="white", linewidths=0.5)
                ax.scatter(x_right, r2, s=80, c=[color], zorder=3,
                           edgecolors="white", linewidths=0.5)

                # Labels
                ax.text(x_left - 0.03, r1, f"{model} ({df1.loc[model, 'accuracy']:.1f}%)",
                        ha="right", va="center", fontsize=8.5, color=color, fontweight="bold")
                ax.text(x_right + 0.03, r2, f"{model} ({df2.loc[model, 'accuracy']:.1f}%)",
                        ha="left", va="center", fontsize=8.5, color=color, fontweight="bold")

            # Axis styling
            ax.set_xlim(-0.55, 1.55)
            ax.set_ylim(n + 0.5, 0.5)
            ax.set_xticks([x_left, x_right])
            ax.set_xticklabels([s1, s2], fontsize=11, fontweight="bold")
            ax.set_ylabel("Rank (1 = best)", fontsize=11)
            ax.set_yticks(range(1, n + 1))

            # Stats annotation
            stats_text = (f"n={n}   Spearman ρ={rho:.2f} (p={rho_p:.3f})"
                          f"   Kendall τ={tau:.2f} (p={tau_p:.3f})"
                          f"\n{rank_inversions}/{n} models changed rank")
            ax.set_title(f"{title}: Rank Order Across Scaffolds\n{stats_text}",
                         fontsize=12, fontweight="bold")

            # Legend for line colors
            legend_elements = [
                mpatches.Patch(color="tab:green", alpha=0.7, label="Rank improved"),
                mpatches.Patch(color="tab:red", alpha=0.7, label="Rank worsened"),
                mpatches.Patch(color="gray", alpha=0.4, label="Rank unchanged"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="lower center",
                      ncol=3, framealpha=0.9)

            ax.grid(True, axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            fig.tight_layout()
            safe = (f"{title}_{s1}_{s2}").lower().replace(" ", "_").replace("-", "_")
            safe = re.sub(r"[^a-z0-9_]", "", safe)
            fig.savefig(BASE / "figures" / f"rank_bump_{safe}.png",
                        dpi=150, bbox_inches="tight")
            print(f"Saved: rank_bump_{safe}.png")
            plt.close(fig)


# ─── Summary heatmap of rank correlations ───────────────────────────────────
stats_df = pd.DataFrame(stats_rows)

if len(stats_df) > 0:
    fig_s, ax_s = plt.subplots(figsize=(12, max(4, len(stats_df) * 0.5 + 1)))

    labels = [f"{r['benchmark']}\n{r['scaffold_1']} → {r['scaffold_2']}\n(n={r['n_shared']})"
              for _, r in stats_df.iterrows()]
    y_pos = range(len(stats_df))

    # Plot both correlations as grouped horizontal bars
    bar_h = 0.35
    bars_rho = ax_s.barh([y - bar_h/2 for y in y_pos], stats_df["spearman_rho"],
                          height=bar_h, color="tab:blue", alpha=0.75, label="Spearman ρ")
    bars_tau = ax_s.barh([y + bar_h/2 for y in y_pos], stats_df["kendall_tau"],
                          height=bar_h, color="tab:orange", alpha=0.75, label="Kendall τ")

    # Value labels — place outside the bar tip on the correct side so negative
    # bars don't cram their labels against the y-axis / category labels.
    def label_bars(bars, vals, ps, color):
        for bar, val, p in zip(bars, vals, ps):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            w = bar.get_width()
            if w >= 0:
                x, ha = w + 0.02, "left"
            else:
                x, ha = w - 0.02, "right"
            ax_s.text(x, bar.get_y() + bar.get_height()/2,
                      f"{val:.2f}{sig}", va="center", ha=ha, fontsize=9, color=color)

    label_bars(bars_rho, stats_df["spearman_rho"], stats_df["spearman_p"], "tab:blue")
    label_bars(bars_tau, stats_df["kendall_tau"], stats_df["kendall_p"], "tab:orange")

    ax_s.set_yticks(list(y_pos))
    ax_s.set_yticklabels(labels, fontsize=8)
    ax_s.set_xlabel("Rank Correlation", fontsize=12)
    ax_s.set_title("Rank-Order Preservation Across Scaffolds\n(* p<0.05, ** p<0.01, *** p<0.001)",
                    fontsize=13, fontweight="bold")
    ax_s.axvline(x=0, color="black", linewidth=0.8)
    ax_s.axvline(x=1, color="gray", linewidth=0.5, linestyle=":")
    ax_s.set_xlim(-0.55, 1.3)
    ax_s.legend(fontsize=10, loc="lower right")
    ax_s.grid(True, axis="x", alpha=0.3)
    ax_s.invert_yaxis()

    fig_s.tight_layout()
    fig_s.savefig(BASE / "figures" / "rank_correlation_summary.png", dpi=150, bbox_inches="tight")
    print("\nSaved: rank_correlation_summary.png")

    # Print summary
    print("\n" + "=" * 90)
    print("RANK CORRELATION SUMMARY")
    print("=" * 90)
    for _, r in stats_df.iterrows():
        print(f"  {r['benchmark']:25s}  {r['scaffold_1']:25s} → {r['scaffold_2']:25s}  "
              f"n={r['n_shared']:2d}  ρ={r['spearman_rho']:+.2f} (p={r['spearman_p']:.3f})  "
              f"τ={r['kendall_tau']:+.2f} (p={r['kendall_p']:.3f})")
