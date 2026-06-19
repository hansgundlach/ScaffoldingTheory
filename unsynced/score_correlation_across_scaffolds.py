"""
Score correlation across scaffolds.

For each benchmark with 2+ scaffolds, scatter plot of model score on scaffold 1
vs score on scaffold 2 (both logit-transformed). Fit OLS line and report r².
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re
from pathlib import Path
from scipy.special import logit, expit
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).parent
HAL = BASE / "hal_data"


def parse_pct(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)%", str(s))
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
    df["scaffold"] = df["Scaffold"].apply(clean_scaffold)
    df["model_norm"] = df["Model"].apply(normalize_model)
    df = df.dropna(subset=["accuracy", "model_norm"])
    df = df[df["accuracy"] > 0]
    df = df[df["accuracy"] < 100]
    # Best accuracy per (scaffold, model_norm)
    df = df.sort_values("accuracy", ascending=False).drop_duplicates(
        ["scaffold", "model_norm"], keep="first"
    )
    return df


def logit_score(pct):
    p = np.clip(pct / 100, 0.001, 0.999)
    return logit(p)


def make_pct_formatter():
    def fmt(x, pos):
        pct = expit(x) * 100
        if pct >= 10:
            return f"{pct:.0f}%"
        return f"{pct:.1f}%"
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
all_pairs = []  # collect for multi-panel figure

for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    scaffolds = sorted(df["scaffold"].dropna().unique())
    if len(scaffolds) < 2:
        continue

    for si in range(len(scaffolds)):
        for sj in range(si + 1, len(scaffolds)):
            s1, s2 = scaffolds[si], scaffolds[sj]
            df1 = df[df["scaffold"] == s1][["model_norm", "accuracy"]].set_index("model_norm")
            df2 = df[df["scaffold"] == s2][["model_norm", "accuracy"]].set_index("model_norm")

            shared = sorted(set(df1.index) & set(df2.index))
            if len(shared) < 4:
                continue

            x_pct = df1.loc[shared, "accuracy"].values
            y_pct = df2.loc[shared, "accuracy"].values
            x_logit = logit_score(x_pct)
            y_logit = logit_score(y_pct)

            mask = np.isfinite(x_logit) & np.isfinite(y_logit)
            x_l, y_l = x_logit[mask], y_logit[mask]
            shared_masked = [s for s, m in zip(shared, mask) if m]

            r_logit, p_logit = pearsonr(x_l, y_l)
            rho, rho_p = spearmanr(x_l, y_l)

            # OLS fit in logit space
            coeffs = np.polyfit(x_l, y_l, 1)
            slope, intercept = coeffs

            all_stats.append({
                "benchmark": title, "scaffold_1": s1, "scaffold_2": s2,
                "n": len(shared_masked),
                "pearson_r_logit": r_logit, "pearson_p": p_logit,
                "spearman_rho": rho, "spearman_p": rho_p,
                "slope": slope, "intercept": intercept,
            })

            all_pairs.append({
                "title": title, "s1": s1, "s2": s2,
                "models": shared_masked,
                "x_pct": x_pct[mask], "y_pct": y_pct[mask],
                "x_logit": x_l, "y_logit": y_l,
                "coeffs": coeffs, "r": r_logit, "p": p_logit,
                "slope": slope,
            })

# ─── Individual scatter plots ───────────────────────────────────────────────
for pair in all_pairs:
    fig, ax = plt.subplots(figsize=(9, 8))

    x_l, y_l = pair["x_logit"], pair["y_logit"]
    models = pair["models"]

    ax.scatter(x_l, y_l, s=80, c="tab:blue", alpha=0.7, edgecolors="white",
               linewidths=0.5, zorder=3)

    for m, xv, yv in zip(models, x_l, y_l):
        ax.annotate(m, (xv, yv), textcoords="offset points", xytext=(5, 5),
                    fontsize=7.5, alpha=0.85)

    # OLS line
    x_range = np.linspace(x_l.min() - 0.2, x_l.max() + 0.2, 50)
    ax.plot(x_range, np.polyval(pair["coeffs"], x_range),
            color="tab:red", linewidth=2, linestyle="--", alpha=0.7,
            label=f"OLS: slope={pair['slope']:.2f}, r={pair['r']:.2f}")

    # y=x reference
    lims = [min(x_l.min(), y_l.min()) - 0.3, max(x_l.max(), y_l.max()) + 0.3]
    ax.plot(lims, lims, "k:", alpha=0.3, label="y = x (identical scores)")

    ax.set_xlabel(f"Score on {pair['s1']} (logit scale, shown as %)", fontsize=11)
    ax.set_ylabel(f"Score on {pair['s2']} (logit scale, shown as %)", fontsize=11)

    sig = "***" if pair["p"] < 0.001 else "**" if pair["p"] < 0.01 else "*" if pair["p"] < 0.05 else ""
    ax.set_title(
        f"{pair['title']}: Score Correlation Across Scaffolds\n"
        f"Pearson r={pair['r']:.2f}{sig} (p={pair['p']:.3f}), "
        f"slope={pair['slope']:.2f}, n={len(models)}",
        fontsize=12, fontweight="bold")

    ax.xaxis.set_major_formatter(make_pct_formatter())
    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    safe = f"{pair['title']}_{pair['s1']}_{pair['s2']}".lower().replace(" ", "_").replace("-", "_")
    safe = re.sub(r"[^a-z0-9_]", "", safe)
    fig.savefig(BASE / "figures" / f"score_corr_{safe}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: score_corr_{safe}.png")
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
    x_l, y_l = pair["x_logit"], pair["y_logit"]

    ax.scatter(x_l, y_l, s=50, c="tab:blue", alpha=0.7, edgecolors="white",
               linewidths=0.3, zorder=3)

    # OLS line
    x_range = np.linspace(x_l.min() - 0.2, x_l.max() + 0.2, 50)
    ax.plot(x_range, np.polyval(pair["coeffs"], x_range),
            color="tab:red", linewidth=1.5, linestyle="--", alpha=0.7)

    # y=x
    lims = [min(x_l.min(), y_l.min()) - 0.3, max(x_l.max(), y_l.max()) + 0.3]
    ax.plot(lims, lims, "k:", alpha=0.3)

    sig = "***" if pair["p"] < 0.001 else "**" if pair["p"] < 0.01 else "*" if pair["p"] < 0.05 else ""
    ax.set_title(f"{pair['title']}\n{pair['s1']} → {pair['s2']}\n"
                 f"r={pair['r']:.2f}{sig}  slope={pair['slope']:.2f}  n={len(pair['models'])}",
                 fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(make_pct_formatter())
    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)

# Hide unused axes
for idx in range(n_pairs, len(list(axes))):
    axes[idx].set_visible(False)

fig_m.suptitle("Score Correlation Across Scaffolds (logit-logit)\nRed dashed = OLS fit, black dotted = y=x",
               fontsize=13, fontweight="bold")
fig_m.tight_layout()
fig_m.savefig(BASE / "figures" / "score_correlation_summary.png", dpi=150, bbox_inches="tight")
print("\nSaved: score_correlation_summary.png")

# ─── Print summary table ────────────────────────────────────────────────────
stats_df = pd.DataFrame(all_stats)
print("\n" + "=" * 110)
print("SCORE CORRELATION SUMMARY (logit-transformed scores)")
print("=" * 110)
print(f"{'Benchmark':25s}  {'Scaffold 1':25s}  {'Scaffold 2':25s}  n   Pearson r   Slope   Interpretation")
print("-" * 110)
for _, r in stats_df.iterrows():
    # Interpret slope
    if r["slope"] > 1.1:
        interp = "scaffold 2 amplifies differences"
    elif r["slope"] < 0.9:
        interp = "scaffold 2 compresses differences"
    else:
        interp = "roughly parallel shift"
    if abs(r["pearson_r_logit"]) < 0.3:
        interp = "WEAK/NO correlation"

    sig = "***" if r["pearson_p"] < 0.001 else "**" if r["pearson_p"] < 0.01 else "*" if r["pearson_p"] < 0.05 else ""
    print(f"  {r['benchmark']:25s}  {r['scaffold_1']:25s}  {r['scaffold_2']:25s}  "
          f"{r['n']:2d}  r={r['pearson_r_logit']:+.2f}{sig:3s}  "
          f"β={r['slope']:+.2f}   {interp}")
