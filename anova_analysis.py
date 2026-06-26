"""
Shapley ANOVA: how much of the variation in agent accuracy is due to the
*model* vs. the *scaffold*?

For each benchmark we fit OLS models on logit-transformed accuracy and use a
Shapley-value decomposition of the regression R^2 (the LMG / Shapley-regression
method). Each factor's contribution is the average of its marginal increase in
R^2 over every ordering in which factors are added. With two factors this is:

    Shapley(model)    = 1/2 * R2(model)    + 1/2 * [R2(both) - R2(scaffold)]
    Shapley(scaffold) = 1/2 * R2(scaffold) + 1/2 * [R2(both) - R2(model)]

Unlike Type-II sum-of-squares, Shapley shares are order-independent and sum
exactly to the full-model R^2; the residual is 1 - R2(both). The headline
figure is a stacked bar chart per benchmark splitting variance into
model / scaffold / residual.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from scipy.special import logit
import statsmodels.formula.api as smf

BASE = Path(__file__).parent
HAL = BASE / "hal_data"
FIG = BASE / "figures"

SAVE_DPI = 150


# ─── Data loading (shared with rank_order_analysis.py) ───────────────────────
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
    s = re.sub(r"\s*\(.*?\)", "", s).strip()        # drop the "(August 2025)" date stamp
    # Keep each reasoning-effort level distinct (High / Medium / Low). Runs that
    # carry no effort label are HAL's default high-effort runs, so tag them
    # "High" as well — gives every model an explicit effort/version label. The
    # run grouping is identical to stripping "High", so results are unchanged.
    if not re.search(r"\b(High|Medium|Low)\b", s):
        s = f"{s} High"
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
    df = df.dropna(subset=["accuracy", "model_norm", "scaffold"])
    # Best accuracy per (scaffold, model) — collapse High/non-High duplicates
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
    ("usaco.csv", "USACO", "Primary Model"),
]


def run_anova(df):
    """Shapley decomposition of R^2 on logit(accuracy). Returns share per factor."""
    d = df.copy()
    p = (d["accuracy"] / 100.0).clip(0.01, 0.99)
    d["y"] = logit(p)

    # Need 2+ levels on each factor and more rows than parameters
    if d["model_norm"].nunique() < 2 or d["scaffold"].nunique() < 2:
        return None
    n_params = d["model_norm"].nunique() + d["scaffold"].nunique() - 1
    if len(d) <= n_params + 1:
        return None

    # R^2 for each coalition of factors
    r2_model = smf.ols("y ~ C(model_norm)", data=d).fit().rsquared
    r2_scaffold = smf.ols("y ~ C(scaffold)", data=d).fit().rsquared
    r2_both = smf.ols("y ~ C(model_norm) + C(scaffold)", data=d).fit().rsquared

    # Shapley values (average marginal contribution over both orderings)
    shap_model = 0.5 * r2_model + 0.5 * (r2_both - r2_scaffold)
    shap_scaffold = 0.5 * r2_scaffold + 0.5 * (r2_both - r2_model)
    resid = 1.0 - r2_both
    if r2_both <= 0:
        return None

    return {
        "n": len(d),
        "n_models": d["model_norm"].nunique(),
        "n_scaffolds": d["scaffold"].nunique(),
        "r2_full": r2_both,
        "eta_model": shap_model,
        "eta_scaffold": shap_scaffold,
        "eta_resid": resid,
    }


rows = []
for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    res = run_anova(df)
    if res is None:
        print(f"  skip {title} (insufficient data)")
        continue
    res["benchmark"] = title
    rows.append(res)
    print(f"  {title:26s}  model={res['eta_model']:.0%}  "
          f"scaffold={res['eta_scaffold']:.0%}  resid={res['eta_resid']:.0%}  (n={res['n']})")

res_df = pd.DataFrame(rows)
# Order by scaffold share so the most scaffold-driven benchmarks sit on top
res_df = res_df.sort_values("eta_scaffold", ascending=True).reset_index(drop=True)

# ─── Stacked horizontal bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 0.62 * len(res_df) + 1.8))

y = np.arange(len(res_df))
c_model = "#4C72B0"
c_scaffold = "#DD8452"
c_resid = "#CCCCCC"

ax.barh(y, res_df["eta_model"], color=c_model, label="Model")
ax.barh(y, res_df["eta_scaffold"], left=res_df["eta_model"],
        color=c_scaffold, label="Scaffold")
ax.barh(y, res_df["eta_resid"],
        left=res_df["eta_model"] + res_df["eta_scaffold"],
        color=c_resid, label="Residual")

# In-bar percentage labels for model & scaffold
for i, r in res_df.iterrows():
    if r["eta_model"] > 0.06:
        ax.text(r["eta_model"] / 2, i, f"{r['eta_model']:.0%}",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if r["eta_scaffold"] > 0.06:
        ax.text(r["eta_model"] + r["eta_scaffold"] / 2, i, f"{r['eta_scaffold']:.0%}",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(res_df["benchmark"], fontsize=11)
ax.set_xlim(0, 1)
ax.set_xlabel("Share of explained variance in accuracy (logit scale)", fontsize=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Shapley variance decomposition: model vs. scaffold\n"
             "(Shapley-$R^2$ on logit accuracy)", fontsize=13)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
ax.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out = FIG / "anova_variance_decomposition.png"
fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
print(f"\nSaved {out}")

res_df.to_csv(BASE / "anova_results.csv", index=False)
print(f"Saved {BASE / 'anova_results.csv'}")
