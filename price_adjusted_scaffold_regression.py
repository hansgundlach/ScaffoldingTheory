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


def shapley(data, factors, dv="y"):
    """Shapley decomposition of full-model R^2 (LMG / Shapley-regression method).

    Each factor's share is the average, over every ordering in which the factors
    are entered, of the increase in R^2 it produces when added. Equivalently it
    is the weighted sum, over every coalition T not containing factor i, of the
    marginal gain R^2(T+i) - R^2(T), with combinatorial weights that count how
    many orderings put i right after T. The weights sum to 1, so the shares are
    order-independent and add up exactly to the full-model R^2; the leftover
    1 - R^2(full) is the residual. This generalises to any number of factors.

    factors: dict label -> formula term, e.g. {'benchmark':'C(benchmark)', ...}.
    dv:      name of the dependent variable column.
    Returns (shares dict incl. 'residual', full-model R^2).
    """
    labels = list(factors)
    from itertools import combinations
    from math import factorial

    n = len(labels)

    def r2_of(subset):
        if not subset:
            return 0.0
        rhs = " + ".join(factors[l] for l in subset)
        return smf.ols(f"{dv} ~ {rhs}", data=data).fit().rsquared

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
    shap, r2_full = shapley(
        data,
        {"benchmark": "C(benchmark)", "model": "C(model_norm)", "scaffold": "C(scaffold)"},
        dv="y",
    )
    out.update(
        shap_benchmark=shap["benchmark"], shap_model=shap["model"],
        shap_scaffold=shap["scaffold"], shap_resid=shap["residual"], r2_full=r2_full,
    )
    return out


# ─── Build samples ───────────────────────────────────────────────────────────
d = build_dataset()

# Export the exact set of (normalised) models entering the analysis, with how
# many systems and benchmarks each appears in. Kept in sync because it is derived
# straight from the analysis dataframe `d`.
models = (
    d.groupby("model_norm")
    .agg(
        n_systems=("model_norm", "size"),
        n_benchmarks=("benchmark", "nunique"),
        n_scaffolds=("scaffold", "nunique"),
        benchmarks=("benchmark", lambda s: "; ".join(sorted(s.unique()))),
    )
    .reset_index()
    .rename(columns={"model_norm": "model"})
    .sort_values(["n_systems", "model"], ascending=[False, True])
    .reset_index(drop=True)
)
models_csv = BASE / "models_in_analysis.csv"
models.to_csv(models_csv, index=False)
print("Saved %s (%d models)\n" % (models_csv, len(models)))

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


# ═══ ALTERNATIVE SPECIFICATION: price as a covariate (no pole) ════════════════
# Here the dependent variable is logit(accuracy) and log(cost) enters as a
# control term rather than a denominator. Dividing is what created the pole at
# cost=$1; as a covariate log(cost) is perfectly well behaved, so this spec uses
# the FULL sample (n=185). "Combined effect of the scaffold dummies" becomes the
# incremental adjusted R^2 from adding them on top of price (and, in the second
# block, price + benchmark). Same fair-comparison logic as before: adjusted R^2
# nets out the differing dummy counts.
print("\n" + "=" * 78)
print("COVARIATE SPEC   DV = logit(accuracy),  log(cost) as a control  (n=%d)" % len(d))
print("=" * 78)


def incr(base_terms, add_term):
    """adjusted R^2 of base, of base+add, and the increment from adding add."""
    base = fit("logit_acc ~ " + " + ".join(base_terms), d)["adj_r2"]
    both = fit("logit_acc ~ " + " + ".join(base_terms + [add_term]), d)["adj_r2"]
    return base, both, both - base

# Net of price only.
b_price = fit("logit_acc ~ log_cost", d)["adj_r2"]
_, as_p, inc_s_p = incr(["log_cost"], "C(scaffold)")
_, am_p, inc_m_p = incr(["log_cost"], "C(model_norm)")
print("\nlogit_acc ~ log_cost                       adjR2=%.3f" % b_price)
print("           + scaffold dummies               adjR2=%.3f  (incremental %+.3f)" % (as_p, inc_s_p))
print("           + model dummies                  adjR2=%.3f  (incremental %+.3f)" % (am_p, inc_m_p))

# Net of price AND benchmark (handles the scaffold-nested-in-benchmark confound).
b_pb = fit("logit_acc ~ log_cost + C(benchmark)", d)["adj_r2"]
_, as_pb, inc_s_pb = incr(["log_cost", "C(benchmark)"], "C(scaffold)")
_, am_pb, inc_m_pb = incr(["log_cost", "C(benchmark)"], "C(model_norm)")
print("\nlogit_acc ~ log_cost + benchmark           adjR2=%.3f" % b_pb)
print("           + scaffold dummies               adjR2=%.3f  (incremental %+.3f)" % (as_pb, inc_s_pb))
print("           + model dummies                  adjR2=%.3f  (incremental %+.3f)" % (am_pb, inc_m_pb))

# 4-factor Shapley: split logit(accuracy) variance across price/benchmark/model/scaffold.
shap_c, r2c = shapley(
    d,
    {"price": "log_cost", "benchmark": "C(benchmark)",
     "model": "C(model_norm)", "scaffold": "C(scaffold)"},
    dv="logit_acc",
)
print("\n4-factor Shapley on logit(accuracy)  (R2_full=%.3f):" % r2c)
for k in ["price", "benchmark", "model", "scaffold", "residual"]:
    print("  %-10s %5.1f%%" % (k, 100 * shap_c[k]))
print("  model:scaffold ratio = %.2fx" % (shap_c["model"] / shap_c["scaffold"]))

# Cleanest model-vs-scaffold comparison: each factor entered LAST, i.e. net of
# price + benchmark AND the other factor (semipartial / unique contribution).
# as_pb = adjR2(price+benchmark+scaffold); am_pb = adjR2(price+benchmark+model).
both_adj = fit("logit_acc ~ log_cost + C(benchmark) + C(model_norm) + C(scaffold)", d)["adj_r2"]
uniq_scaffold = both_adj - am_pb     # scaffold added last
uniq_model = both_adj - as_pb        # model added last
print("\nUnique contribution, each added LAST (net of price + benchmark + other), "
      "full adjR2=%.3f:" % both_adj)
print("  scaffold  %+.3f" % uniq_scaffold)
print("  model     %+.3f   -> scaffold/model = %.2fx" % (uniq_model, uniq_scaffold / uniq_model))

# Persist covariate results.
cov = pd.DataFrame(
    [
        ("logit_acc ~ log_cost", b_price, np.nan),
        ("  + scaffold", as_p, inc_s_p),
        ("  + model", am_p, inc_m_p),
        ("logit_acc ~ log_cost + benchmark", b_pb, np.nan),
        ("  + scaffold", as_pb, inc_s_pb),
        ("  + model", am_pb, inc_m_pb),
        ("full (price+benchmark+model+scaffold)", both_adj, np.nan),
        ("  scaffold unique (added last)", both_adj, uniq_scaffold),
        ("  model unique (added last)", both_adj, uniq_model),
    ],
    columns=["spec", "adj_r2", "incremental_adj_r2"],
)
for k in ["price", "benchmark", "model", "scaffold", "residual"]:
    cov["shap_" + k] = shap_c[k]
cov_csv = BASE / "price_covariate_scaffold_results.csv"
cov.to_csv(cov_csv, index=False)
print("\nSaved %s" % cov_csv)

# ─── Figure: incremental adj R^2 (scaffold vs model), net of price / +benchmark
fig2, ax2 = plt.subplots(figsize=(7.4, 3.8))
groups = ["net of price", "net of price\n+ benchmark"]
scaf_inc = [inc_s_p, inc_s_pb]
mod_inc = [inc_m_p, inc_m_pb]
x = np.arange(len(groups))
w = 0.36
ax2.bar(x - w / 2, scaf_inc, w, color="#DD8452", label="Scaffold dummies")
ax2.bar(x + w / 2, mod_inc, w, color="#4C72B0", label="Model dummies")
for xi in x:
    ax2.text(xi - w / 2, scaf_inc[xi] + 0.005, f"{scaf_inc[xi]:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.text(xi + w / 2, mod_inc[xi] + 0.005, f"{mod_inc[xi]:.2f}", ha="center", va="bottom", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=10)
ax2.set_ylabel("Incremental adjusted $R^2$\non logit(accuracy)", fontsize=10)
ax2.set_title("Price as a covariate (n=%d): added explanatory power\nof scaffold vs. model dummies" % len(d), fontsize=12)
ax2.legend(fontsize=9, framealpha=0.95)
ax2.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
plt.tight_layout()
out_fig2 = FIG / "price_covariate_scaffold_adjr2.png"
fig2.savefig(out_fig2, dpi=SAVE_DPI, bbox_inches="tight")
print("Saved %s" % out_fig2)

# ─── Figure: conservative, non-Shapley comparison (net of price + benchmark) ──
# Benchmark + price entered first and absorb the shared variance; scaffold/model
# dummies added afterward. The scaffold-unfavourable way to keep score.
fig3, ax3 = plt.subplots(figsize=(5.6, 4.0))
bars = ["Scaffold", "Model"]
vals = [inc_s_pb, inc_m_pb]
colors = ["#DD8452", "#4C72B0"]
xb = np.arange(len(bars))
ax3.bar(xb, vals, width=0.6, color=colors)
for xi, v in zip(xb, vals):
    ax3.text(xi, v + 0.004, f"+{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_xticks(xb)
ax3.set_xticklabels(bars, fontsize=12)
ax3.set_ylim(0, max(vals) * 1.25)
ax3.set_ylabel("Added explanatory power\n(incremental adjusted $R^2$)", fontsize=11)
ax3.set_title("Drivers of agent performance, net of price and benchmark\n"
              "(benchmark absorbs shared variance; scaffold added later)", fontsize=11.5)
ax3.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)
ax3.annotate(f"DV = logit(accuracy);  n = {len(d)} systems across "
             f"{d['benchmark'].nunique()} HAL benchmarks",
             xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
             fontsize=8, color="#666")
plt.tight_layout()
out_fig3 = FIG / "scaffold_vs_model_net_of_price_benchmark.png"
fig3.savefig(out_fig3, dpi=SAVE_DPI, bbox_inches="tight")
print("Saved %s" % out_fig3)
