"""
Cost correlation across scaffolds.

For each benchmark with 2+ scaffolds, scatter plot of model cost on scaffold 1
vs cost on scaffold 2 (both log-transformed). Fit OLS line and report r².
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

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
    df = df.dropna(subset=["accuracy", "cost", "model_norm"])
    df = df[df["cost"] > 0]
    # Best accuracy per (scaffold, model_norm) — keeps cost of best-performing config
    df = df.sort_values("accuracy", ascending=False).drop_duplicates(
        ["scaffold", "model_norm"], keep="first"
    )
    return df


def make_dollar_formatter():
    def fmt(x, pos):
        v = 10 ** x
        if v >= 1000:
            return f"${v:.0f}"
        elif v >= 1:
            return f"${v:.1f}"
        else:
            return f"${v:.2f}"
    return mticker.FuncFormatter(fmt)


BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("scicode.csv", "SciCode", "Primary Model"),
    ("online_mine_2_web.csv", "Online Mind2Web", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
]

all_stats = []
all_pairs = []

for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    scaffolds = sorted(df["scaffold"].dropna().unique())
    if len(scaffolds) < 2:
        continue

    for si in range(len(scaffolds)):
        for sj in range(si + 1, len(scaffolds)):
            s1, s2 = scaffolds[si], scaffolds[sj]
            df1 = df[df["scaffold"] == s1][["model_norm", "cost"]].set_index("model_norm")
            df2 = df[df["scaffold"] == s2][["model_norm", "cost"]].set_index("model_norm")

            shared = sorted(set(df1.index) & set(df2.index))
            if len(shared) < 4:
                continue

            x_cost = df1.loc[shared, "cost"].values
            y_cost = df2.loc[shared, "cost"].values

            x_log = np.log10(x_cost)
            y_log = np.log10(y_cost)

            mask = np.isfinite(x_log) & np.isfinite(y_log)
            x_l, y_l = x_log[mask], y_log[mask]
            x_c, y_c = x_cost[mask], y_cost[mask]
            shared_masked = [s for s, m in zip(shared, mask) if m]

            if len(shared_masked) < 4:
                continue

            r_log, p_log = pearsonr(x_l, y_l)
            rho, rho_p = spearmanr(x_l, y_l)

            coeffs = np.polyfit(x_l, y_l, 1)
            slope, intercept = coeffs

            # Compute median cost ratio (scaffold 2 / scaffold 1)
            ratios = y_c / x_c
            median_ratio = np.median(ratios)

            all_stats.append({
                "benchmark": title, "scaffold_1": s1, "scaffold_2": s2,
                "n": len(shared_masked),
                "pearson_r_log": r_log, "pearson_p": p_log,
                "spearman_rho": rho, "spearman_p": rho_p,
                "slope": slope, "intercept": intercept,
                "median_cost_ratio": median_ratio,
            })

            all_pairs.append({
                "title": title, "s1": s1, "s2": s2,
                "models": shared_masked,
                "x_cost": x_c, "y_cost": y_c,
                "x_log": x_l, "y_log": y_l,
                "coeffs": coeffs, "r": r_log, "p": p_log,
                "slope": slope, "median_ratio": median_ratio,
            })

# ─── Individual scatter plots ───────────────────────────────────────────────
for pair in all_pairs:
    fig, ax = plt.subplots(figsize=(9, 8))

    x_l, y_l = pair["x_log"], pair["y_log"]
    models = pair["models"]

    ax.scatter(x_l, y_l, s=80, c="tab:purple", alpha=0.7, edgecolors="white",
               linewidths=0.5, zorder=3)

    for m, xv, yv in zip(models, x_l, y_l):
        ax.annotate(m, (xv, yv), textcoords="offset points", xytext=(5, 5),
                    fontsize=7.5, alpha=0.85)

    # OLS line
    x_range = np.linspace(x_l.min() - 0.3, x_l.max() + 0.3, 50)
    ax.plot(x_range, np.polyval(pair["coeffs"], x_range),
            color="tab:red", linewidth=2, linestyle="--", alpha=0.7,
            label=f"OLS: slope={pair['slope']:.2f}, r={pair['r']:.2f}")

    # y=x reference (same cost on both scaffolds)
    lims = [min(x_l.min(), y_l.min()) - 0.3, max(x_l.max(), y_l.max()) + 0.3]
    ax.plot(lims, lims, "k:", alpha=0.3, label="y = x (same cost)")

    ax.set_xlabel(f"Cost on {pair['s1']} (log₁₀ USD)", fontsize=11)
    ax.set_ylabel(f"Cost on {pair['s2']} (log₁₀ USD)", fontsize=11)

    sig = "***" if pair["p"] < 0.001 else "**" if pair["p"] < 0.01 else "*" if pair["p"] < 0.05 else ""
    ax.set_title(
        f"{pair['title']}: Cost Correlation Across Scaffolds\n"
        f"Pearson r={pair['r']:.2f}{sig} (p={pair['p']:.3f}), "
        f"slope={pair['slope']:.2f}, median cost ratio={pair['median_ratio']:.1f}x, "
        f"n={len(models)}",
        fontsize=11, fontweight="bold")

    ax.xaxis.set_major_formatter(make_dollar_formatter())
    ax.yaxis.set_major_formatter(make_dollar_formatter())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    safe = f"{pair['title']}_{pair['s1']}_{pair['s2']}".lower().replace(" ", "_").replace("-", "_")
    safe = re.sub(r"[^a-z0-9_]", "", safe)
    fig.savefig(BASE / "figures" / f"cost_corr_{safe}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: cost_corr_{safe}.png")
    plt.close(fig)

# ─── Multi-panel summary ────────────────────────────────────────────────────
n_pairs = len(all_pairs)
ncols = min(4, n_pairs)
nrows = (n_pairs + ncols - 1) // ncols
fig_m, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
if n_pairs == 1:
    axes = np.array([axes])
axes = axes.flat

for idx, pair in enumerate(all_pairs):
    ax = axes[idx]
    x_l, y_l = pair["x_log"], pair["y_log"]

    ax.scatter(x_l, y_l, s=50, c="tab:purple", alpha=0.7, edgecolors="white",
               linewidths=0.3, zorder=3)

    x_range = np.linspace(x_l.min() - 0.3, x_l.max() + 0.3, 50)
    ax.plot(x_range, np.polyval(pair["coeffs"], x_range),
            color="tab:red", linewidth=1.5, linestyle="--", alpha=0.7)

    lims = [min(x_l.min(), y_l.min()) - 0.3, max(x_l.max(), y_l.max()) + 0.3]
    ax.plot(lims, lims, "k:", alpha=0.3)

    sig = "***" if pair["p"] < 0.001 else "**" if pair["p"] < 0.01 else "*" if pair["p"] < 0.05 else ""
    ax.set_title(f"{pair['title']}\n{pair['s1']} → {pair['s2']}\n"
                 f"r={pair['r']:.2f}{sig}  slope={pair['slope']:.2f}  "
                 f"ratio={pair['median_ratio']:.1f}x  n={len(pair['models'])}",
                 fontsize=8, fontweight="bold")
    ax.xaxis.set_major_formatter(make_dollar_formatter())
    ax.yaxis.set_major_formatter(make_dollar_formatter())
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)

for idx in range(n_pairs, len(list(axes))):
    axes[idx].set_visible(False)

fig_m.suptitle("Cost Correlation Across Scaffolds (log₁₀-log₁₀)\n"
               "Red dashed = OLS fit, black dotted = y=x (same cost)",
               fontsize=13, fontweight="bold")
fig_m.tight_layout()
fig_m.savefig(BASE / "figures" / "cost_correlation_summary.png", dpi=150, bbox_inches="tight")
print("\nSaved: cost_correlation_summary.png")

# ─── Print summary table ────────────────────────────────────────────────────
stats_df = pd.DataFrame(all_stats)
print("\n" + "=" * 120)
print("COST CORRELATION SUMMARY (log10-transformed costs)")
print("=" * 120)
print(f"{'Benchmark':25s}  {'Scaffold 1':25s}  {'Scaffold 2':25s}  "
      f"n   Pearson r   Slope   Med.Ratio  Interpretation")
print("-" * 120)
for _, r in stats_df.iterrows():
    if r["slope"] > 1.1:
        interp = "expensive models get disproportionately more expensive"
    elif r["slope"] < 0.9:
        interp = "cost differences compress (cheap models cost relatively more)"
    else:
        interp = "roughly multiplicative shift"

    if abs(r["pearson_r_log"]) < 0.3:
        interp = "WEAK/NO cost correlation"

    shift = "cheaper" if r["median_cost_ratio"] < 1 else "more expensive"

    sig = "***" if r["pearson_p"] < 0.001 else "**" if r["pearson_p"] < 0.01 else "*" if r["pearson_p"] < 0.05 else ""
    print(f"  {r['benchmark']:25s}  {r['scaffold_1']:25s}  {r['scaffold_2']:25s}  "
          f"{r['n']:2d}  r={r['pearson_r_log']:+.2f}{sig:3s}  "
          f"β={r['slope']:+.2f}   {r['median_cost_ratio']:5.1f}x     "
          f"S2 is {r['median_cost_ratio']:.1f}x {shift}; {interp}")
