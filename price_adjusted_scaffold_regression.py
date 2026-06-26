"""
Price-adjusted variance decomposition: how much of the variation in
*cost-efficient* agent performance is driven by the scaffold vs. the model?

This is the price-aware companion to anova_analysis.py. Instead of decomposing
logit(accuracy), we use a single efficiency metric that folds price in:

    y = logit(accuracy) / log(cost)          ("logit performance per log dollar")

and regress it on scaffold dummies (one dummy per scaffold in the pooled data).
The headline number is the *adjusted* R^2 of that regression: the combined
explanatory power of all the scaffold dummies for y. We report the same number
for model dummies so the two can be compared on equal footing (adjusted R^2
already penalises the model factor for having more levels than the scaffold
factor, so the comparison is fair despite the differing dummy counts).

Caveats this script makes explicit, because they matter for interpretation:

  * Nesting. In HAL most scaffolds are benchmark-specific (CORE-Agent only on
    CORE-bench, the SciCode agents only on SciCode, ...), whereas models recur
    across benchmarks. A raw pooled `y ~ C(scaffold)` therefore lets the
    scaffold dummies soak up *benchmark-difficulty* variance and overstates the
    scaffold. We counter this two ways: (a) incremental adjusted R^2 after
    benchmark fixed effects, and (b) a 3-factor Shapley decomposition
    (benchmark / model / scaffold) that cleanly separates the benchmark term.
  * The metric. log(cost) flips sign below $1, so y flips sign for the four
    sub-$1 runs; a run priced near $1 also inflates |y|. The log base is
    irrelevant to every R^2 here (it only rescales y). We report a robustness
    pass that drops cost < $1.

Outputs: price_adjusted_scaffold_results.csv and a small adj-R^2 bar figure.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
import statsmodels.formula.api as smf

BASE = Path(__file__).parent
HAL = BASE / "hal_data"
FIG = BASE / "figures"
SAVE_DPI = 150


# ─── Data loading (conventions shared with anova_analysis.py / hal_price_performance.py)
def parse_pct(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)%", str(s))
    return float(m.group(1)) if m else np.nan


def parse_cost(s):
    """Dollar cost from '$15.15', '$1,577.26', '$463.90 (-438.32/+438.32)'."""
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


def load_hal(filename, title, model_col="Primary Model"):
    df = pd.read_csv(HAL / filename, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Models" in df.columns and model_col not in df.columns:
        model_col = "Models"
    df = df.rename(columns={model_col: "Model"})
    df["accuracy"] = df["Accuracy"].apply(parse_pct)
    df["cost"] = df["Cost (USD)"].apply(parse_cost)
    df["scaffold"] = df["Scaffold"].apply(clean_scaffold)
    df["model_norm"] = df["Model"].apply(normalize_model)
    df["benchmark"] = title
    df = df.dropna(subset=["accuracy", "cost", "scaffold", "model_norm"])
    df = df[(df["accuracy"] > 0) & (df["accuracy"] < 100)]
    # Collapse High / non-High duplicates exactly as anova_analysis.py does:
    # best accuracy per (scaffold, model) within a benchmark.
    df = df.sort_values("accuracy", ascending=False).drop_duplicates(
        ["benchmark", "scaffold", "model_norm"], keep="first"
    )
    return df[["benchmark", "scaffold", "model_norm", "accuracy", "cost"]]


BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("scicode.csv", "SciCode", "Primary Model"),
    ("online_mine_2_web.csv", "Online Mind2Web", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
    ("usaco.csv", "USACO", "Primary Model"),
    ("assist_bench.csv", "AssistBench", "Primary Model"),
]


def build_dataset():
    frames = []
    for fname, title, mcol in BENCHMARKS:
        p = HAL / fname
        if not p.exists() or p.stat().st_size == 0:
            continue
        frames.append(load_hal(fname, title, mcol))
    d = pd.concat(frames, ignore_index=True)
    # Efficiency metric: logit performance per log dollar.
    p = (d["accuracy"] / 100.0).clip(0.01, 0.99)        # same clip as anova_analysis.py
    d["logit_acc"] = logit(p)
    d["log_cost"] = np.log(d["cost"])                    # log base is irrelevant to R^2
    d["y"] = d["logit_acc"] / d["log_cost"]
    return d


# ─── Helpers ─────────────────────────────────────────────────────────────────
def fit(formula, data):
    m = smf.ols(formula, data=data).fit()
    return {"r2": m.rsquared, "adj_r2": m.rsquared_adj, "df_model": int(m.df_model)}


def shapley_3(data, factors):
    """3-factor Shapley decomposition of full-model R^2 (LMG / Shapley-regression).

    factors: dict label -> formula term, e.g. {'benchmark':'C(benchmark)', ...}.
    Returns label -> share of total variance; shares + residual sum to 1.
    """
    labels = list(factors)
    from itertools import combinations
    from math import factorial

    n = len(labels)

    def r2_of(subset):
        if not subset:
            return 0.0
        rhs = " + ".join(factors[l] for l in subset)
        return smf.ols(f"y ~ {rhs}", data=data).fit().rsquared

    # Cache R^2 for every coalition.
    cache = {}
    for k in range(n + 1):
        for combo in combinations(labels, k):
            cache[frozenset(combo)] = r2_of(combo)

    shap = {}
    for i in labels:
        rest = [l for l in labels if l != i]
        total = 0.0
        for k in range(len(rest) + 1):
            for combo in combinations(rest, k):
                T = frozenset(combo)
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                total += w * (cache[T | {i}] - cache[T])
        shap[i] = total
    shap["residual"] = 1.0 - cache[frozenset(labels)]
    return shap, cache[frozenset(labels)]




# ─── Analysis run over a sample ──────────────────────────────────────────────
def analyze(data):
    """All specs for one sample. DV is y = logit(accuracy)/log(cost) throughout."""
    out = {"n": len(data)}
    s = fit("y ~ C(scaffold)", data)
    m = fit("y ~ C(model_norm)", data)
    b = fit("y ~ C(benchmark)", data)
    bs = fit("y ~ C(benchmark) + C(scaffold)", data)
    bm = fit("y ~ C(benchmark) + C(model_norm)", data)
    out.update(
        scaffold_r2=s["r2"], scaffold_adj=s["adj_r2"], scaffold_k=s["df_model"],
        model_r2=m["r2"], model_adj=m["adj_r2"], model_k=m["df_model"],
        bench_adj=b["adj_r2"],
        inc_scaffold=bs["adj_r2"] - b["adj_r2"],
        inc_model=bm["adj_r2"] - b["adj_r2"],
    )
    shap, r2_full = shapley_3(
        data,
        {"benchmark": "C(benchmark)", "model": "C(model_norm)", "scaffold": "C(scaffold)"},
    )
    out.update(
        shap_benchmark=shap["benchmark"], shap_model=shap["model"],
        shap_scaffold=shap["scaffold"], shap_resid=shap["residual"], r2_full=r2_full,
    )
    return out


# ─── Build samples ───────────────────────────────────────────────────────────
d = build_dataset()
full = d
primary = d[d["cost"] >= 1.0].copy()                 # drop the pole at cost=$1
# Winsorise y to its 2nd/98th percentile on the full sample (alternative to dropping)
lo, hi = d["y"].quantile([0.02, 0.98])
wins = d.copy()
wins["y"] = wins["y"].clip(lo, hi)

print("=" * 78)
print("PRICE-ADJUSTED PERFORMANCE   y = logit(accuracy) / log(cost)")
print(f"pooled | {d['benchmark'].nunique()} benchmarks | {d['scaffold'].nunique()} "
      f"scaffolds | {d['model_norm'].nunique()} models")
print("NOTE: y has a pole at cost=$1; the cost>=$1 sample is the primary spec.")
print("=" * 78)

samples = [("FULL (n=%d)" % len(full), full),
           ("PRIMARY  cost>=$1 (n=%d)" % len(primary), primary),
           ("WINSORISED 2/98pct (n=%d)" % len(wins), wins)]

results = {name: analyze(data) for name, data in samples}

for name, r in results.items():
    print(f"\n--- {name} ---")
    print(f"  y ~ scaffold dummies   adjR2={r['scaffold_adj']:+.3f}  (R2={r['scaffold_r2']:.3f}, {r['scaffold_k']} dummies)")
    print(f"  y ~ model dummies      adjR2={r['model_adj']:+.3f}  (R2={r['model_r2']:.3f}, {r['model_k']} dummies)")
    print(f"  incremental adjR2 over benchmark FE:  scaffold {r['inc_scaffold']:+.3f} | model {r['inc_model']:+.3f}")
    print(f"  3-factor Shapley (R2_full={r['r2_full']:.3f}):  "
          f"benchmark {r['shap_benchmark']:.1%} | model {r['shap_model']:.1%} | "
          f"scaffold {r['shap_scaffold']:.1%} | resid {r['shap_resid']:.1%}")

# ─── Persist results table ───────────────────────────────────────────────────
res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "sample"})
out_csv = BASE / "price_adjusted_scaffold_results.csv"
res.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# ─── Figure: scaffold vs model adj-R^2 across samples ────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 3.8))
names = list(results)
x = np.arange(len(names))
w = 0.36
scaf = [results[n]["scaffold_adj"] for n in names]
mod = [results[n]["model_adj"] for n in names]
ax.bar(x - w / 2, scaf, w, color="#DD8452", label="Scaffold dummies")
ax.bar(x + w / 2, mod, w, color="#4C72B0", label="Model dummies")
for xi in x:
    ax.text(xi - w / 2, scaf[xi] + (0.005 if scaf[xi] >= 0 else -0.02),
            f"{scaf[xi]:.2f}", ha="center", va="bottom" if scaf[xi] >= 0 else "top", fontsize=9)
    ax.text(xi + w / 2, mod[xi] + (0.005 if mod[xi] >= 0 else -0.02),
            f"{mod[xi]:.2f}", ha="center", va="bottom" if mod[xi] >= 0 else "top", fontsize=9)
ax.axhline(0, color="#888", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=10)
ax.set_ylabel("Adjusted $R^2$\n$y=\\mathrm{logit(acc)}/\\log(\\mathrm{cost})$", fontsize=10)
ax.set_title("Combined explanatory power for price-adjusted performance:\n"
             "scaffold vs. model (adjusted $R^2$)", fontsize=12)
ax.legend(fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
out_fig = FIG / "price_adjusted_scaffold_adjr2.png"
fig.savefig(out_fig, dpi=SAVE_DPI, bbox_inches="tight")
print(f"Saved {out_fig}")
