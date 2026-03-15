"""
GPQA-Diamond (base capability) vs Agentic Benchmark (scaffolded) performance.

For each model that appears in both GPQA-Diamond (Epoch) and a HAL benchmark,
plot GPQA score (x) vs agentic score (y), colored by scaffold.
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
EPOCH = BASE / "epoch_benchmark_data"

# ─── Utility functions (shared with hal_price_performance.py) ────────────────

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

def load_hal(filename, model_col="Primary Model"):
    df = pd.read_csv(HAL / filename, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Models" in df.columns and model_col not in df.columns:
        model_col = "Models"
    df = df.rename(columns={model_col: "Model"})
    df["accuracy"] = df["Accuracy"].apply(parse_pct)
    df["cost"] = df["Cost (USD)"].apply(parse_cost)
    df["scaffold"] = df["Scaffold"].apply(clean_scaffold)
    df = df.dropna(subset=["accuracy", "cost"])
    df = df[df["accuracy"] > 0]
    df = df[df["accuracy"] < 100]
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


# ─── Load GPQA-Diamond ──────────────────────────────────────────────────────
gpqa = pd.read_csv(EPOCH / "gpqa_diamond.csv")
gpqa["gpqa_pct"] = gpqa["mean_score"] * 100

# For models with multiple thinking-budget variants, keep best score per base model
# We'll create a base_model_id for matching
def gpqa_base_id(model_version):
    """Strip thinking budget suffixes like _32K, _64K, _16K, _high, _medium, etc."""
    s = str(model_version)
    # Remove trailing _NNK (thinking budget)
    s = re.sub(r"_\d+K$", "", s)
    # Remove trailing _high, _medium, _low, _xhigh
    s = re.sub(r"_(xhigh|high|medium|low)$", "", s)
    # Remove org prefix
    s = re.sub(r"^(openai|fireworks)/", "", s)
    return s

gpqa["base_id"] = gpqa["Model version"].apply(gpqa_base_id)

# For each base_id, keep the row with the default/base thinking budget
# Priority: no suffix > _16K > _32K > _64K (closer to "standard" eval)
def suffix_priority(model_version):
    s = str(model_version)
    if re.search(r"_\d+K$", s) or re.search(r"_(xhigh|high|medium|low)$", s):
        # Has suffix — extract it
        m = re.search(r"_(\d+K|xhigh|high|medium|low)$", s)
        suffix = m.group(1) if m else ""
        priority = {"16K": 2, "32K": 3, "27K": 3, "59K": 4, "64K": 5,
                     "low": 1, "medium": 2, "high": 3, "xhigh": 4}
        return priority.get(suffix, 3)
    return 0  # no suffix = base = highest priority

gpqa["_prio"] = gpqa["Model version"].apply(suffix_priority)
gpqa_dedup = gpqa.sort_values("_prio").drop_duplicates("base_id", keep="first")

# Also keep a version with best score (for "High" HAL models)
gpqa_best = gpqa.sort_values("gpqa_pct", ascending=False).drop_duplicates("base_id", keep="first")


# ─── HAL model name → GPQA base_id mapping ──────────────────────────────────
# HAL names: "Claude Sonnet 4.5 (September 2025)", "o4-mini High (April 2025)", etc.
# GPQA base_ids: "claude-sonnet-4-5-20250929", "o4-mini-2025-04-16", etc.

# Explicit mapping table
HAL_TO_GPQA = {
    # Claude models
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Claude Sonnet 4.5 (September 2025)": "claude-sonnet-4-5-20250929",
    "Claude Sonnet 4.5 High (September 2025)": "claude-sonnet-4-5-20250929",
    "Claude Opus 4.5": "claude-opus-4-5-20251101",
    "Claude Opus 4.5 (November 2025)": "claude-opus-4-5-20251101",
    "Claude Opus 4.5 High (November 2025)": "claude-opus-4-5-20251101",
    "Claude Opus 4.1": "claude-opus-4-1-20250805",
    "Claude Opus 4.1 (August 2025)": "claude-opus-4-1-20250805",
    "Claude Opus 4.1 High (August 2025)": "claude-opus-4-1-20250805",
    "Claude Opus 4": "claude-opus-4-20250514",
    "Claude Opus 4 (May 2025)": "claude-opus-4-20250514",
    "Claude Opus 4 High (May 2025)": "claude-opus-4-20250514",
    "Claude Sonnet 4": "claude-sonnet-4-20250514",
    "Claude Sonnet 4 (May 2025)": "claude-sonnet-4-20250514",
    "Claude Sonnet 4 High (May 2025)": "claude-sonnet-4-20250514",
    "Claude-3.7 Sonnet": "claude-3-7-sonnet-20250219",
    "Claude-3.7 Sonnet (February 2025)": "claude-3-7-sonnet-20250219",
    "Claude-3.7 Sonnet High (February 2025)": "claude-3-7-sonnet-20250219",
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
    "Claude Haiku 4.5 (October 2025)": "claude-haiku-4-5-20251001",
    "Claude Haiku 4.5 High (October 2025)": "claude-haiku-4-5-20251001",
    # OpenAI models
    "o4-mini High (April 2025)": "o4-mini-2025-04-16",
    "o4-mini Low (April 2025)": "o4-mini-2025-04-16",
    "o3 Medium (April 2025)": "o3-2025-04-16",
    "GPT-5 Medium (August 2025)": "gpt-5-2025-08-07",
    "GPT-4.1 (April 2025)": "gpt-4.1-2025-04-14",
    # DeepSeek
    "DeepSeek R1 (January 2025)": "DeepSeek-R1",
    "DeepSeek R1 (May 2025)": "deepseek-reasoner",
    "DeepSeek V3 (March 2025)": None,  # No GPQA entry
    "DeepSeek V3.1 (August 2025)": None,
    # Google
    "Gemini 2.0 Flash (February 2025)": None,  # No clear GPQA match
    "Gemini 2.0 Flash High (February 2025)": None,
    "Gemini 2.5 Pro Preview (March 2025)": "gemini-2.5-pro-exp-03-25",
    "Gemini 3 Pro Preview High (November 2025)": "gemini-3-pro-preview",
    # OSS
    "GPT-OSS-120B (August 2025)": "openai/gpt-oss-120b",
    "GPT-OSS-120B High (August 2025)": "openai/gpt-oss-120b",
}

# For "High" HAL models, we want the higher-thinking-budget GPQA score
HAL_HIGH_MODELS = {k for k in HAL_TO_GPQA if "High" in str(k)}


def get_gpqa_score(hal_model_name):
    """Look up GPQA score for a HAL model name. Returns (gpqa_pct, gpqa_model_version, matched)."""
    gpqa_id = HAL_TO_GPQA.get(str(hal_model_name).strip())
    if gpqa_id is None:
        return np.nan, None, False

    # For "High" HAL models, use best GPQA score for that base model
    if hal_model_name in HAL_HIGH_MODELS:
        lookup = gpqa_best
    else:
        lookup = gpqa_dedup

    match = lookup[lookup["base_id"] == gpqa_id]
    if match.empty:
        # Try case-insensitive
        match = lookup[lookup["base_id"].str.lower() == gpqa_id.lower()]
    if match.empty:
        return np.nan, gpqa_id, False

    row = match.iloc[0]
    return row["gpqa_pct"], row["Model version"], True


# ─── Build match report ─────────────────────────────────────────────────────
match_report = []
match_report.append("# GPQA-Diamond ↔ HAL Benchmark Model Matching Report\n")
match_report.append("This document describes how HAL benchmark model names were matched to")
match_report.append("Epoch GPQA-Diamond model IDs for the GPQA vs Agentic performance plots.\n")
match_report.append("## Matching approach\n")
match_report.append("1. HAL uses human-readable names like `Claude Sonnet 4.5 (September 2025)`")
match_report.append("2. Epoch GPQA uses machine IDs like `claude-sonnet-4-5-20250929_32K`")
match_report.append("3. An explicit mapping table (`HAL_TO_GPQA`) maps HAL names → GPQA base model IDs")
match_report.append("4. GPQA models have multiple thinking-budget variants (_16K, _32K, _64K, etc.)")
match_report.append("5. For standard HAL entries, we use the **base** (no suffix) GPQA score")
match_report.append("6. For HAL entries marked **High**, we use the **best** GPQA score across budgets")
match_report.append("7. Some HAL models have no GPQA match (DeepSeek V3, Gemini 2.0 Flash) — these are excluded\n")
match_report.append("## Known issues and limitations\n")
match_report.append("- **Thinking budget mismatch**: HAL 'High' likely uses a specific thinking budget;")
match_report.append("  we use best-available GPQA score as a proxy, which may overestimate base capability.")
match_report.append("- **GPQA is a static QA benchmark**, not an agentic one — it measures reasoning")
match_report.append("  capability without tool use, multi-step planning, or environment interaction.")
match_report.append("- **HAL 'High' vs standard**: The same base model appears with different thinking")
match_report.append("  budgets in HAL. We map both to the same GPQA base model but use different GPQA")
match_report.append("  score variants (base vs best).")
match_report.append("- **Missing matches**: DeepSeek V3/V3.1, Gemini 2.0 Flash have no GPQA-Diamond entry.\n")
match_report.append("## Full matching table\n")
match_report.append("| HAL Model Name | GPQA Base ID | GPQA Version Used | GPQA Score | Matched? |")
match_report.append("|---|---|---|---|---|")

BENCHMARKS = [
    ("swe_bench_mini_verified.csv", "SWE-bench Mini Verified", "Primary Model"),
    ("gaia.csv", "GAIA", "Primary Model"),
    ("core_bench_hard.csv", "CORE-bench Hard", "Primary Model"),
    ("tau_bench_airline.csv", "TAU-bench Airline", "Primary Model"),
    ("usaco.csv", "USACO", "Primary Model"),
    ("sci_agent_bench.csv", "SciAgentBench", "Models"),
]

# Collect all unique HAL model names across all benchmarks
all_hal_models = set()
for fname, _, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)
    all_hal_models.update(df["Model"].dropna().unique())

for hal_name in sorted(all_hal_models):
    gpqa_pct, gpqa_ver, matched = get_gpqa_score(hal_name)
    gpqa_id = HAL_TO_GPQA.get(str(hal_name).strip(), "NOT IN MAP")
    if gpqa_id is None:
        gpqa_id = "(no GPQA entry)"
    match_report.append(
        f"| {hal_name} | {gpqa_id} | {gpqa_ver or '—'} | "
        f"{f'{gpqa_pct:.1f}%' if not np.isnan(gpqa_pct) else '—'} | "
        f"{'Yes' if matched else 'No'} |"
    )


# ─── Generate plots ─────────────────────────────────────────────────────────
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

per_benchmark_stats = []

for fname, title, mcol in BENCHMARKS:
    df = load_hal(fname, mcol)

    # Attach GPQA scores
    results = df["Model"].apply(get_gpqa_score)
    df["gpqa_pct"] = [r[0] for r in results]
    df["gpqa_version"] = [r[1] for r in results]
    df["gpqa_matched"] = [r[2] for r in results]

    df_matched = df[df["gpqa_matched"]].copy()
    if len(df_matched) < 3:
        print(f"Skipping {title}: only {len(df_matched)} matched points")
        continue

    scaffolds = sorted(df_matched["scaffold"].unique())
    cmap = plt.cm.tab10
    scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for si, scaffold in enumerate(scaffolds):
        grp = df_matched[df_matched["scaffold"] == scaffold]
        color = scaffold_colors[scaffold]
        marker = MARKERS[si % len(MARKERS)]

        gpqa_logit = logit_score(grp["gpqa_pct"])
        agentic_logit = logit_score(grp["accuracy"])

        ax.scatter(gpqa_logit, agentic_logit, c=[color], s=70, alpha=0.7,
                   marker=marker, edgecolors="white", linewidths=0.4,
                   label=scaffold, zorder=3)

        # Label each point with model name
        for _, row in grp.iterrows():
            model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
            model_short = model_short.replace("Claude ", "C.").replace("High", "H")
            ax.annotate(model_short, (logit_score(row["gpqa_pct"]), logit_score(row["accuracy"])),
                        textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                        alpha=0.8, color=color)

        # Fit and draw trend line per scaffold if enough points
        if len(grp) >= 4:
            x = gpqa_logit.values
            y = agentic_logit.values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 4:
                coeffs = np.polyfit(x[mask], y[mask], 1)
                x_range = np.linspace(x[mask].min(), x[mask].max(), 50)
                ax.plot(x_range, np.polyval(coeffs, x_range),
                        color=color, linestyle="--", alpha=0.5, linewidth=1.5)

    # Diagonal reference line (y = x would mean agentic = GPQA)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k:", alpha=0.3, label="y = x (agentic = GPQA)")

    ax.set_xlabel("GPQA-Diamond Score (logit scale, shown as %)", fontsize=12)
    ax.set_ylabel(f"{title} Score (logit scale, shown as %)", fontsize=12)
    ax.set_title(f"GPQA-Diamond vs {title}\n(each point = one model on one scaffold)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(make_pct_formatter())
    ax.yaxis.set_major_formatter(make_pct_formatter())
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower right",
              framealpha=0.9)

    fig.tight_layout()
    safe_name = title.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    fig.savefig(BASE / "figures" / f"gpqa_vs_{safe_name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: gpqa_vs_{safe_name}.png  ({len(df_matched)} matched points)")
    plt.close(fig)

    # Stats for report
    per_benchmark_stats.append(f"\n### {title}\n")
    per_benchmark_stats.append(f"- {len(df_matched)} matched points out of {len(df)} total HAL entries")
    per_benchmark_stats.append(f"- {len(df_matched['scaffold'].unique())} scaffolds: {', '.join(scaffolds)}")
    unmatched = df[~df["gpqa_matched"]]["Model"].unique()
    if len(unmatched) > 0:
        per_benchmark_stats.append(f"- Unmatched HAL models: {', '.join(sorted(set(str(m) for m in unmatched)))}")

# ─── Write match report ─────────────────────────────────────────────────────
match_report.append("\n## Per-benchmark matching statistics\n")
match_report.extend(per_benchmark_stats)

report_path = BASE / "gpqa_matching_report.md"
report_path.write_text("\n".join(match_report))
print(f"\nSaved matching report: {report_path.name}")

# ─── Anthropic-only CORE-bench Hard vs GPQA-Diamond ─────────────────────────
df = load_hal("core_bench_hard.csv", "Primary Model")
results = df["Model"].apply(get_gpqa_score)
df["gpqa_pct"] = [r[0] for r in results]
df["gpqa_version"] = [r[1] for r in results]
df["gpqa_matched"] = [r[2] for r in results]

# Filter to Anthropic models only (Claude*)
df_anth = df[df["gpqa_matched"] & df["Model"].str.contains("Claude", case=False, na=False)].copy()

scaffolds = sorted(df_anth["scaffold"].unique())
cmap = plt.cm.tab10
scaffold_colors = {s: cmap(i % 10) for i, s in enumerate(scaffolds)}

fig, ax = plt.subplots(figsize=(12, 8))

for si, scaffold in enumerate(scaffolds):
    grp = df_anth[df_anth["scaffold"] == scaffold]
    color = scaffold_colors[scaffold]
    marker = MARKERS[si % len(MARKERS)]

    gpqa_logit = logit_score(grp["gpqa_pct"])
    agentic_logit = logit_score(grp["accuracy"])

    ax.scatter(gpqa_logit, agentic_logit, c=[color], s=80, alpha=0.7,
               marker=marker, edgecolors="white", linewidths=0.5,
               label=scaffold, zorder=3)

    for _, row in grp.iterrows():
        model_short = re.sub(r"\s*\(.*?\)", "", str(row["Model"]))
        model_short = model_short.replace("Claude ", "").replace("High", "H")
        ax.annotate(model_short, (logit_score(row["gpqa_pct"]), logit_score(row["accuracy"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8,
                    alpha=0.85, color=color)

    # Trend line per scaffold
    if len(grp) >= 3:
        x = gpqa_logit.values
        y = agentic_logit.values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_range = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(x_range, np.polyval(coeffs, x_range),
                    color=color, linestyle="--", alpha=0.5, linewidth=1.5)

lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "k:", alpha=0.3, label="y = x (agentic = GPQA)")

ax.set_xlabel("GPQA-Diamond Score (logit scale, shown as %)", fontsize=12)
ax.set_ylabel("CORE-bench Hard Score (logit scale, shown as %)", fontsize=12)
ax.set_title("GPQA-Diamond vs CORE-bench Hard — Anthropic Models Only\n(each point = one Claude model on one scaffold)",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(make_pct_formatter())
ax.yaxis.set_major_formatter(make_pct_formatter())
ax.grid(True, alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="lower right", framealpha=0.9)

fig.tight_layout()
fig.savefig(BASE / "figures" / "gpqa_vs_core_bench_hard_anthropic_only.png", dpi=150, bbox_inches="tight")
print(f"Saved: gpqa_vs_core_bench_hard_anthropic_only.png  ({len(df_anth)} points)")
