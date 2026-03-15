"""
Compare Anthropic self-reported benchmark scores vs external scaffold evaluations.

Data sources:
- anthropic_agentic_benchmarks_reconstructed.csv: Anthropic's own reported scores
- epoch_benchmark_data/: Epoch AI standardized evaluations (various scaffolds)
- hal_data/: HAL benchmark leaderboard data (various scaffolds)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent

# ─── Load Anthropic self-reported data ───────────────────────────────────────
anthropic = pd.read_csv(BASE / "anthropic_agentic_benchmarks_reconstructed.csv")
anthropic["date"] = pd.to_datetime(anthropic["date"])
# Only keep base score_pct rows (not high-compute variants)
anthropic_base = anthropic[anthropic["metric_name"] == "score_pct"].copy()

# ─── Load Epoch SWE-bench Verified ──────────────────────────────────────────
epoch_swe = pd.read_csv(BASE / "epoch_benchmark_data" / "swe_bench_verified.csv")
epoch_swe["Release date"] = pd.to_datetime(epoch_swe["Release date"], errors="coerce")
epoch_swe["score_pct"] = epoch_swe["mean_score"] * 100
epoch_swe_claude = epoch_swe[epoch_swe["Organization"] == "Anthropic"].copy()
# Deduplicate: keep best score per model version
epoch_swe_claude = epoch_swe_claude.sort_values("score_pct", ascending=False).drop_duplicates("Model version")

# ─── Load Epoch OSWorld ─────────────────────────────────────────────────────
epoch_os = pd.read_csv(BASE / "epoch_benchmark_data" / "os_world_external.csv")
epoch_os["Release date"] = pd.to_datetime(epoch_os["Release date"], errors="coerce")
epoch_os.rename(columns={"Score": "score_pct"}, inplace=True)
epoch_os_claude = epoch_os[epoch_os["Organization"] == "Anthropic"].copy()
# Take best score per model (across step counts)
epoch_os_claude = epoch_os_claude.sort_values("score_pct", ascending=False).drop_duplicates("Model version")

# ─── Load Epoch TerminalBench ───────────────────────────────────────────────
epoch_tb = pd.read_csv(BASE / "epoch_benchmark_data" / "terminalbench_external.csv")
epoch_tb["Release date"] = pd.to_datetime(epoch_tb["Release date"], errors="coerce")
epoch_tb["score_pct"] = epoch_tb["Accuracy mean"] * 100
epoch_tb_claude = epoch_tb[epoch_tb["Model Org"] == "Anthropic"].copy()
# Best scaffold per model
epoch_tb_claude_best = epoch_tb_claude.sort_values("score_pct", ascending=False).drop_duplicates("Model version")

# ─── Helper: map model names to short labels and release dates ──────────────
MODEL_ORDER = [
    ("claude-3-5-sonnet-20240620", "3.5 Sonnet v1", "2024-06-21"),
    ("claude-3-5-sonnet-20241022", "3.5 Sonnet v2", "2024-10-22"),
    ("claude-3-7-sonnet-20250219", "3.7 Sonnet", "2025-02-24"),
    ("claude-opus-4-20250514", "Opus 4", "2025-05-22"),
    ("claude-sonnet-4-20250514", "Sonnet 4", "2025-05-22"),
    ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "2025-09-29"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5", "2025-10-15"),
    ("claude-opus-4-5-20251101", "Opus 4.5", "2025-11-24"),
    ("claude-opus-4-6", "Opus 4.6", "2026-02-05"),
    ("claude-sonnet-4-6", "Sonnet 4.6", "2026-02-17"),
]

def short_name(model_id):
    for mid, short, _ in MODEL_ORDER:
        if mid in str(model_id):
            return short
    return str(model_id)[:20]

# ─── Figure 1: SWE-bench Verified comparison ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Anthropic Self-Reported vs External Scaffold Benchmark Scores\n(Claude models only)",
             fontsize=14, fontweight="bold")

# --- SWE-bench ---
ax = axes[0, 0]
ax.set_title("SWE-bench Verified")

# Anthropic self-reported
swe_anth = anthropic_base[anthropic_base["benchmark_name"].str.contains("SWE-bench", case=False)].copy()
swe_anth = swe_anth.sort_values("date")
swe_anth["label"] = swe_anth["model_name"].apply(short_name)
ax.plot(swe_anth["date"], swe_anth["metric_value"], "s-", color="tab:red",
        markersize=8, label="Anthropic self-reported", linewidth=2, zorder=5)
for _, row in swe_anth.iterrows():
    ax.annotate(row["label"], (row["date"], row["metric_value"]),
                textcoords="offset points", xytext=(5, 8), fontsize=7, color="tab:red")

# Epoch standardized scaffold
swe_ext = epoch_swe_claude.dropna(subset=["Release date"]).sort_values("Release date")
swe_ext["label"] = swe_ext["Model version"].apply(short_name)
ax.plot(swe_ext["Release date"], swe_ext["score_pct"], "o--", color="tab:blue",
        markersize=8, label="Epoch standardized scaffold", linewidth=2, zorder=5)
for _, row in swe_ext.iterrows():
    ax.annotate(row["label"], (row["Release date"], row["score_pct"]),
                textcoords="offset points", xytext=(5, -12), fontsize=7, color="tab:blue")

ax.set_ylabel("Score (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)

# --- OSWorld ---
ax = axes[0, 1]
ax.set_title("OSWorld")

os_anth = anthropic_base[anthropic_base["benchmark_name"].str.contains("OSWorld", case=False)].copy()
os_anth = os_anth.sort_values("date")
os_anth["label"] = os_anth["model_name"].apply(short_name)
ax.plot(os_anth["date"], os_anth["metric_value"], "s-", color="tab:red",
        markersize=8, label="Anthropic self-reported", linewidth=2, zorder=5)
for _, row in os_anth.iterrows():
    ax.annotate(row["label"], (row["date"], row["metric_value"]),
                textcoords="offset points", xytext=(5, 8), fontsize=7, color="tab:red")

os_ext = epoch_os_claude.dropna(subset=["Release date"]).sort_values("Release date")
os_ext["label"] = os_ext["Model version"].apply(short_name)
ax.plot(os_ext["Release date"], os_ext["score_pct"], "o--", color="tab:blue",
        markersize=8, label="Epoch external (best config)", linewidth=2, zorder=5)
for _, row in os_ext.iterrows():
    ax.annotate(row["label"], (row["Release date"], row["score_pct"]),
                textcoords="offset points", xytext=(5, -12), fontsize=7, color="tab:blue")

ax.set_ylabel("Score (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)

# --- Terminal-Bench ---
ax = axes[1, 0]
ax.set_title("Terminal-Bench (v2)")

tb_anth = anthropic_base[anthropic_base["benchmark_name"].str.contains("Terminal", case=False)].copy()
tb_anth = tb_anth.sort_values("date")
tb_anth["label"] = tb_anth["model_name"].apply(short_name)
ax.plot(tb_anth["date"], tb_anth["metric_value"], "s-", color="tab:red",
        markersize=8, label="Anthropic self-reported", linewidth=2, zorder=5)
for _, row in tb_anth.iterrows():
    ax.annotate(row["label"], (row["date"], row["metric_value"]),
                textcoords="offset points", xytext=(5, 8), fontsize=7, color="tab:red")

# Epoch: show best scaffold AND spread
tb_ext_best = epoch_tb_claude_best.dropna(subset=["Release date"]).copy()
# Clean model version names
tb_ext_best["base_model"] = tb_ext_best["Model version"].str.replace(r"_.*", "", regex=True)
tb_ext_best = tb_ext_best.sort_values("Release date").drop_duplicates("base_model")
tb_ext_best["label"] = tb_ext_best["base_model"].apply(short_name)
ax.plot(tb_ext_best["Release date"], tb_ext_best["score_pct"], "o--", color="tab:blue",
        markersize=8, label="Epoch best external scaffold", linewidth=2, zorder=5)
for _, row in tb_ext_best.iterrows():
    ax.annotate(f"{row['label']}\n({row['Agent']})", (row["Release date"], row["score_pct"]),
                textcoords="offset points", xytext=(5, -12), fontsize=6, color="tab:blue")

# Also show scaffold spread as a range
tb_claude_all = epoch_tb_claude.dropna(subset=["Release date"]).copy()
tb_claude_all["base_model"] = tb_claude_all["Model version"].str.replace(r"_.*", "", regex=True)
for model, grp in tb_claude_all.groupby("base_model"):
    rd = grp["Release date"].iloc[0]
    ax.plot([rd, rd], [grp["score_pct"].min(), grp["score_pct"].max()],
            color="tab:blue", alpha=0.3, linewidth=3)

ax.set_ylabel("Score (%)")
ax.set_xlabel("Model release date")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)

# --- TAU-bench / τ2-bench ---
ax = axes[1, 1]
ax.set_title("TAU-bench / τ2-bench (retail)")

tau_anth = anthropic_base[anthropic_base["benchmark_name"].str.contains("bench retail", case=False)].copy()
tau_anth = tau_anth.sort_values("date")
tau_anth["label"] = tau_anth["model_name"].apply(short_name)
ax.plot(tau_anth["date"], tau_anth["metric_value"], "s-", color="tab:red",
        markersize=8, label="Anthropic self-reported", linewidth=2, zorder=5)
for _, row in tau_anth.iterrows():
    ax.annotate(row["label"], (row["date"], row["metric_value"]),
                textcoords="offset points", xytext=(5, 8), fontsize=7, color="tab:red")

ax.set_ylabel("Score (%)")
ax.set_xlabel("Model release date")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(BASE / "figures" / "anthropic_vs_external_benchmarks.png", dpi=150, bbox_inches="tight")
plt.savefig(BASE / "figures" / "anthropic_vs_external_benchmarks.pdf", bbox_inches="tight")
print("Saved figure 1: anthropic_vs_external_benchmarks")

# ─── Figure 2: Scaffold spread on Terminal-Bench (all orgs for context) ─────
fig2, ax2 = plt.subplots(figsize=(14, 7))
ax2.set_title("Terminal-Bench v2: Scaffold Spread per Claude Model\n(each point = a different scaffold/agent)",
              fontsize=13, fontweight="bold")

# Plot all Claude scaffolds
colors_map = {}
color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
for i, (mid, sname, _) in enumerate(MODEL_ORDER):
    colors_map[sname] = color_cycle[i % 10]

tb_claude_all = epoch_tb_claude.dropna(subset=["Release date"]).copy()
tb_claude_all["short"] = tb_claude_all["Model version"].str.replace(r"_.*", "", regex=True).apply(short_name)

for sname, grp in tb_claude_all.groupby("short"):
    color = colors_map.get(sname, "gray")
    ax2.scatter(grp["score_pct"], [sname] * len(grp), c=[color], s=60, alpha=0.7, zorder=3)
    # Label top scaffold
    best = grp.loc[grp["score_pct"].idxmax()]
    ax2.annotate(best["Agent"], (best["score_pct"], sname),
                textcoords="offset points", xytext=(5, 5), fontsize=7, color=color)

# Overlay Anthropic self-reported
for _, row in tb_anth.iterrows():
    sname = short_name(row["model_name"])
    ax2.scatter(row["metric_value"], sname, marker="*", s=200, c="red",
                zorder=5, edgecolors="black", linewidths=0.5)

ax2.axvline(x=50, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Score (%)")
ax2.scatter([], [], marker="*", s=200, c="red", edgecolors="black", label="Anthropic self-reported")
ax2.scatter([], [], marker="o", s=60, c="tab:blue", label="External scaffold (Epoch)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(BASE / "figures" / "terminalbench_scaffold_spread.png", dpi=150, bbox_inches="tight")
print("Saved figure 2: terminalbench_scaffold_spread")

# ─── Figure 3: Score gap (Anthropic reported - Epoch measured) ──────────────
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_title("Score Gap: Anthropic Self-Reported minus Epoch Standardized Score\n(SWE-bench Verified, Claude models)",
              fontsize=13, fontweight="bold")

# Match models between anthropic and epoch SWE-bench
gaps = []
for _, arow in swe_anth.iterrows():
    model = arow["model_name"]
    anth_score = arow["metric_value"]
    # Find in epoch data
    match = epoch_swe_claude[epoch_swe_claude["Model version"].apply(lambda x: model in str(x) or str(x) in model)]
    if match.empty:
        # Try fuzzy
        for _, erow in epoch_swe_claude.iterrows():
            if short_name(model) == short_name(erow["Model version"]):
                match = pd.DataFrame([erow])
                break
    if not match.empty:
        ext_score = match.iloc[0]["score_pct"]
        gaps.append({
            "model": short_name(model),
            "date": arow["date"],
            "anthropic": anth_score,
            "epoch": ext_score,
            "gap": anth_score - ext_score
        })

if gaps:
    gaps_df = pd.DataFrame(gaps).sort_values("date")
    bars = ax3.bar(gaps_df["model"], gaps_df["gap"], color=["tab:red" if g > 0 else "tab:blue" for g in gaps_df["gap"]])
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_ylabel("Score gap (percentage points)")
    ax3.set_xlabel("Model")
    for bar, row in zip(bars, gaps_df.itertuples()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"A:{row.anthropic:.1f}\nE:{row.epoch:.1f}", ha="center", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(BASE / "figures" / "swebench_score_gap.png", dpi=150, bbox_inches="tight")
print("Saved figure 3: swebench_score_gap")

# ─── Print summary table ────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: SWE-bench Verified — Anthropic vs Epoch Scores")
print("="*80)
if gaps:
    for g in gaps:
        print(f"  {g['model']:20s}  Anthropic: {g['anthropic']:5.1f}%  Epoch: {g['epoch']:5.1f}%  Gap: {g['gap']:+5.1f}pp")

print("\n" + "="*80)
print("SUMMARY: Terminal-Bench — Anthropic vs Best External Scaffold")
print("="*80)
for _, row in tb_anth.iterrows():
    model = row["model_name"]
    sname = short_name(model)
    anth_score = row["metric_value"]
    ext_match = tb_ext_best[tb_ext_best["label"] == sname]
    if not ext_match.empty:
        ext_score = ext_match.iloc[0]["score_pct"]
        agent = ext_match.iloc[0]["Agent"]
        print(f"  {sname:20s}  Anthropic: {anth_score:5.1f}%  Best external: {ext_score:5.1f}% ({agent})  Gap: {anth_score - ext_score:+5.1f}pp")
    else:
        # Check all claude TB entries
        tb_all_model = epoch_tb_claude[epoch_tb_claude["Model version"].str.contains(model.split("-2025")[0].split("-2026")[0], case=False, na=False)]
        if not tb_all_model.empty:
            best = tb_all_model.loc[tb_all_model["score_pct"].idxmax()]
            print(f"  {sname:20s}  Anthropic: {anth_score:5.1f}%  Best external: {best['score_pct']:5.1f}% ({best['Agent']})  Gap: {anth_score - best['score_pct']:+5.1f}pp")
        else:
            print(f"  {sname:20s}  Anthropic: {anth_score:5.1f}%  (no external match found)")

print("\n" + "="*80)
print("SUMMARY: OSWorld — Anthropic vs Best External Score")
print("="*80)
for _, row in os_anth.iterrows():
    model = row["model_name"]
    sname = short_name(model)
    anth_score = row["metric_value"]
    ext_match = epoch_os_claude[epoch_os_claude["Model version"].apply(lambda x: short_name(x) == sname)]
    if not ext_match.empty:
        ext_score = ext_match.iloc[0]["score_pct"]
        agent_info = ext_match.iloc[0].get("Agent", "")
        print(f"  {sname:20s}  Anthropic: {anth_score:5.1f}%  External best: {ext_score:5.1f}%  Gap: {anth_score - ext_score:+5.1f}pp")
    else:
        print(f"  {sname:20s}  Anthropic: {anth_score:5.1f}%  (no external match found)")

plt.show()
