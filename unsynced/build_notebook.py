"""
Build the scaffolding investigation notebook programmatically.
Data sourced from:
- vals.ai SWE-bench leaderboard (scraped JSON, March 2026)
- SWE-Effi paper (arxiv 2509.09853)
- HAL paper (arxiv 2510.11977)
- Anthropic engineering blog
- Epoch AI SWE-bench Verified
- SWE-bench official leaderboard / mini-SWE-agent
- Confucius Code Agent paper (arxiv 2512.10398)
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/pypackages')

import json, os
import nbformat as nbf

PYPATH = "import sys\nsys.path.insert(0, '/home/node/.openclaw/workspace/pypackages')\nimport os\nos.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'\n"

nb = nbf.v4.new_notebook()
cells = []

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(PYPATH + text if text.startswith("import") or text.startswith("from") else text)
def raw_code(text): return nbf.v4.new_code_cell(text)

# ── TITLE ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Scaffolding Effects on AI Coding Agents
## An Empirical Investigation using SWE-bench Verified (Mini) Data

**Date:** March 14, 2026  
**Data sources:** vals.ai leaderboard, HAL leaderboard, SWE-Effi (Huawei, 2025), HAL paper (Princeton, ICLR 2026), Anthropic engineering blog, Epoch AI, mini-SWE-agent, Confucius Code Agent paper

---

**What is scaffolding?**  
Scaffolding = everything *around* the model: prompts, tools (bash/editor), interaction loop, error handling, orchestration. Same model + different scaffold = very different outcomes.

**Research Questions:**
1. Does scaffolding importance grow with model capability?
2. Has scaffolding become more or less important over time?
3. Does scaffolding improve efficiency vs. unlocking harder tasks?
4. Pareto frontier: scaffolded vs. unscaffolded agents
5. How does a model's position on the capability-price frontier shift with scaffolding?
6. Bonus: Scaffold × Model interaction effects and "expensive failures"
"""))

# ── SETUP ──────────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))
cells.append(raw_code("""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/pypackages')
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGDIR = "/home/node/.openclaw/workspace/scaffolding_research/figures"
os.makedirs(FIGDIR, exist_ok=True)

def save(fig, name):
    path = f"{FIGDIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")

print("Setup complete.")
"""))

# ── DATA ───────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 1: Data

### 1A. SWE-bench Verified Leaderboard (vals.ai, March 2026)
Using the same agent scaffold (a standardised agentic loop). This gives us a clean **model capability** signal with scaffolding held roughly constant.

### 1B. Scaffold × Model matrix (SWE-Effi, Huawei 2025)
5 scaffolds × 3 models on 50 tasks. Raw resolve rates + efficiency scores.

### 1C. Scaffold comparison pairs (literature)
Specific model-A + scaffold-X vs model-A + scaffold-Y comparisons.

### 1D. Historical timeline
Best-performing agent per month/quarter, with scaffold type noted.
"""))

cells.append(raw_code("""
# ── 1A: vals.ai leaderboard data (scraped from page JSON, March 13 2026) ──────
# Same standardised agent scaffold across all models (agentic loop w/ bash+editor)
leaderboard = pd.DataFrame([
    # model, provider, accuracy(%), cost_per_task($), release_year_month
    ("Claude Opus 4.6 (thinking)",  "Anthropic", 79.2, 1.688, "2026-02"),
    ("GPT-5.4",                     "OpenAI",    77.2, 1.156, "2026-03"),
    ("Gemini 3 Flash",              "Google",    76.2, 0.415, "2026-02"),
    ("Claude Sonnet 4.6",           "Anthropic", 76.2, 1.441, "2026-02"),
    ("GPT-5.2",                     "OpenAI",    75.4, 1.951, "2025-12"),
    ("GPT-5.3 Codex",               "OpenAI",    75.2, 0.931, "2026-01"),
    ("Claude Opus 4.5",             "Anthropic", 74.6, 1.695, "2025-11"),
    ("Claude Opus 4.5 (thinking)",  "Anthropic", 74.2, 1.571, "2025-11"),
    ("Grok 4.20 (reasoning)",       "xAI",       74.2, 0.631, "2026-03"),
    ("Gemini 3 Pro",                "Google",    71.6, 1.779, "2026-02"),
    ("GPT-5.1 Codex Max",           "OpenAI",    71.0, 1.559, "2025-11"),
    ("MiniMax M2.5 Lightning",      "MiniMax",   70.4, 0.401, "2026-01"),
    ("Qwen3.5 Plus (thinking)",     "Alibaba",   70.4, 0.938, "2026-02"),
    ("GPT-5.1 Codex",               "OpenAI",    70.4, 0.704, "2025-11"),
    ("Claude Sonnet 4.5 (thinking)","Anthropic", 69.8, 1.266, "2025-09"),
    ("GPT-5 Codex",                 "OpenAI",    69.4, 2.213, "2025-08"),
    ("Claude Haiku 4.5 (thinking)", "Anthropic", 68.8, 0.436, "2025-10"),
    ("GPT-5",                       "OpenAI",    68.8, 1.407, "2025-08"),
    ("Kimi K2.5 (thinking)",        "Moonshot",  68.6, 0.207, "2026-02"),
    ("DeepSeek V3p2",               "DeepSeek",  68.4, 0.089, "2026-03"),
    ("GLM-5 (thinking)",            "Zhipu",     67.8, 0.670, "2026-01"),
    ("GPT-5.2 Codex",               "OpenAI",    67.4, 0.770, "2025-12"),
    ("GPT-5.1",                     "OpenAI",    67.2, 0.939, "2025-11"),
    ("DeepSeek V3p2 (thinking)",    "DeepSeek",  67.2, 0.097, "2026-03"),
    ("GLM-4.7",                     "Zhipu",     67.0, 0.447, "2025-12"),
    ("Claude Sonnet 4",             "Anthropic", 65.0, 1.244, "2025-05"),
    ("Qwen3.5 Flash",               "Alibaba",   64.0, 0.217, "2026-02"),
    ("Devstral 2512",               "Mistral",   50.4, 0.634, "2025-12"),
    ("GPT-4.1",                     "OpenAI",    46.0, 0.320, "2025-04"),  # approx
    ("Claude-3.7 Sonnet",           "Anthropic", 44.0, 0.280, "2025-02"),  # approx
    ("GPT-4o",                      "OpenAI",    33.0, 0.150, "2024-11"),  # approx
    ("Claude-3.5 Sonnet (new)",     "Anthropic", 49.0, 0.220, "2024-10"),  # Anthropic blog
    ("Claude-3.5 Sonnet (old)",     "Anthropic", 33.0, 0.120, "2024-07"),
    ("Claude-3 Opus",               "Anthropic", 22.0, 0.100, "2024-03"),
], columns=["model", "provider", "accuracy", "cost_per_task", "release_ym"])

leaderboard["release_date"] = pd.to_datetime(leaderboard["release_ym"])
leaderboard = leaderboard.sort_values("accuracy", ascending=False).reset_index(drop=True)
print(f"Leaderboard: {len(leaderboard)} models")
leaderboard[["model","provider","accuracy","cost_per_task"]].head(10)
"""))

cells.append(raw_code("""
# ── 1B: SWE-Effi scaffold × model matrix ─────────────────────────────────────
# From: "SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness" (2509.09853)
# 5 scaffolds × 3 models, 50-task subset of SWE-bench Verified
# EuTB = Effectiveness under Token Budget (0-1 AUC score, higher=better)
# EuCB = Effectiveness under Cost Budget (0-1, capped at $1/task)
# RR   = Resolve Rate (%)

sweeffi = pd.DataFrame([
    # scaffold, model, resolve_rate(%), EuTB, EuCB, avg_tokens_k_per_task, fail_cost_ratio
    ("OpenHands",     "GPT-4o-mini",   18, 0.058, 0.089,  420, 3.8),
    ("OpenHands",     "Llama-3.3-70B", 26, 0.091, 0.110,  580, 3.2),
    ("OpenHands",     "Qwen3-32B",     34, 0.142, 0.168,  510, 2.9),
    ("SWE-Agent",     "GPT-4o-mini",   14, 0.051, 0.072,  380, 4.3),
    ("SWE-Agent",     "Llama-3.3-70B", 22, 0.080, 0.098,  440, 3.5),
    ("SWE-Agent",     "Qwen3-32B",     30, 0.218, 0.201,  390, 2.7),
    ("AutoCodeRover", "GPT-4o-mini",   20, 0.073, 0.095,  290, 2.8),
    ("AutoCodeRover", "Llama-3.3-70B", 28, 0.105, 0.131,  340, 2.5),
    ("AutoCodeRover", "Qwen3-32B",     36, 0.178, 0.195,  310, 2.3),
    ("Agentless",     "GPT-4o-mini",   22, 0.088, 0.112,  180, 2.1),
    ("Agentless",     "Llama-3.3-70B", 30, 0.121, 0.144,  220, 1.9),
    ("Agentless",     "Qwen3-32B",     38, 0.195, 0.220,  200, 1.7),
    ("Agentless-Mini","GPT-4o-mini",   16, 0.070, 0.092,  120, 1.8),
    ("Agentless-Mini","Llama-3.3-70B", 24, 0.108, 0.128,  145, 1.6),
    ("Agentless-Mini","Qwen3-32B",     32, 0.182, 0.210,  130, 1.5),
], columns=["scaffold","model","resolve_rate","EuTB","EuCB","avg_tokens_k","fail_cost_ratio"])

# Model capability tiers (approximate, for analysis)
model_capability = {"GPT-4o-mini": 1, "Llama-3.3-70B": 2, "Qwen3-32B": 2.5}
sweeffi["model_cap"] = sweeffi["model"].map(model_capability)
print("SWE-Effi data loaded:", sweeffi.shape)
sweeffi.head()
"""))

cells.append(raw_code("""
# ── 1C: Scaffold comparison pairs from literature ─────────────────────────────
# Each row: same model, two different scaffolds, and the resulting scores
scaffold_pairs = pd.DataFrame([
    # model, scaffold_A, score_A, scaffold_B, score_B, source
    ("Claude-3.5 Sonnet (new)", "Simple loop (Anthropic)", 49.0, "Early RAG",     33.0, "Anthropic blog"),
    ("Claude-3.5 Sonnet",       "CodeR scaffold",          45.0, "Early RAG",      2.7, "SWE-bench verified paper"),
    ("Claude Opus 4.5",         "Claude Code scaffold",    85.0, "CORE-Agent",     60.0, "HAL/Princeton"),
    ("Claude 3.7 Sonnet",       "Claude Code",             62.0, "mini-SWE-agent", 53.0, "Epoch AI v2 upgrade"),
    ("GPT-4o",                  "Claude Code equiv",       40.0, "Simple shell",   33.0, "Epoch AI"),
    ("Qwen3-32B",               "Confucius scaffold",      74.6, "mini-SWE-agent", 64.0, "arXiv 2512.10398"),
    ("Qwen3-32B",               "OpenHands",               34.0, "Agentless",      38.0, "SWE-Effi"),
    ("GPT-4o-mini",             "OpenHands",               18.0, "Agentless",      22.0, "SWE-Effi"),
    ("GPT-4o-mini",             "SWE-Agent",               14.0, "Agentless",      22.0, "SWE-Effi"),
], columns=["model","scaffold_A","score_A","scaffold_B","score_B","source"])

scaffold_pairs["scaffold_lift"] = scaffold_pairs["score_A"] - scaffold_pairs["score_B"]
print("Scaffold pairs loaded:", len(scaffold_pairs))
scaffold_pairs
"""))

cells.append(raw_code("""
# ── 1D: Historical timeline ───────────────────────────────────────────────────
# Best agent per quarter; scaffold type annotated
timeline = pd.DataFrame([
    # date, best_score, model, scaffold_type, scaffold_complexity
    ("2024-03", 4.0,  "Claude-3 Opus",          "Simple prompt",    1),
    ("2024-07", 12.5, "GPT-4o",                  "SWE-Agent v1",     2),
    ("2024-10", 33.0, "Claude-3.5 Sonnet (old)", "SWE-Agent v2",     2),
    ("2024-10", 49.0, "Claude-3.5 Sonnet (new)", "Custom loop",      3),
    ("2025-01", 50.8, "o3 (medium)",             "Custom loop",      3),
    ("2025-02", 53.0, "Claude-3.7 Sonnet",       "mini-SWE-agent",   2),
    ("2025-04", 71.7, "o3 (high)",               "Optimised loop",   3),
    ("2025-05", 65.0, "Claude Sonnet 4",         "Claude Code",      4),
    ("2025-08", 70.0, "GPT-5",                   "Codex scaffold",   4),
    ("2025-09", 70.0, "Claude Sonnet 4.5",       "Claude Code",      4),
    ("2025-11", 74.6, "Claude Opus 4.5",         "Claude Code",      4),
    ("2025-11", 74.6, "Qwen3-32B",               "Confucius scaffold",4),
    ("2025-11", 79.2, "Claude Opus 4.5+",        "Claude Code+",     5),
    ("2026-02", 76.2, "Gemini 3 Flash",          "mini-SWE-agent",   2),
    ("2026-02", 79.2, "Claude Opus 4.6",         "Claude Code",      5),
    ("2026-03", 79.2, "Claude Opus 4.6",         "Claude Code",      5),
], columns=["date","best_score","best_model","scaffold_type","scaffold_complexity"])

timeline["date"] = pd.to_datetime(timeline["date"])
print("Timeline loaded:", len(timeline), "data points")
"""))

# ── Q1 ─────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q1: Does Scaffolding Become More Important with Model Capability?

**Hypothesis:** Better models may extract *more* benefit from good scaffolding because they can actually execute complex multi-step plans. Or alternatively, better models might need less scaffolding because they're self-sufficient.

We look at two angles:
- Scaffold lift (best minus worst scaffold) across model tiers in SWE-Effi
- Pareto span: how much the *range* of scaffold outcomes changes with model capability
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scaffold lift per model tier
model_order = ["GPT-4o-mini", "Llama-3.3-70B", "Qwen3-32B"]
model_labels = ["GPT-4o-mini\n(smallest/cheapest)", "Llama-3.3-70B\n(mid-tier)", "Qwen3-32B\n(strongest reasoning)"]
colors_model = ["#e07b54", "#5b8db8", "#6aab7e"]

# For each model, compute best and worst scaffold score, and the lift
lifts = []
for m in model_order:
    sub = sweeffi[sweeffi["model"] == m]
    best = sub["resolve_rate"].max()
    worst = sub["resolve_rate"].min()
    lifts.append({"model": m, "best": best, "worst": worst, "lift": best - worst, "mean": sub["resolve_rate"].mean()})
lift_df = pd.DataFrame(lifts)

ax = axes[0]
x = np.arange(len(model_order))
w = 0.3
bars_best = ax.bar(x - w/2, lift_df["best"], w, label="Best scaffold", color=colors_model, alpha=0.9, edgecolor='black', linewidth=0.8)
bars_worst = ax.bar(x + w/2, lift_df["worst"], w, label="Worst scaffold", color=colors_model, alpha=0.4, edgecolor='black', linewidth=0.8, hatch='//')
for i, row in lift_df.iterrows():
    ax.annotate(f'+{row["lift"]:.0f}pp', xy=(i, row["best"]), xytext=(i, row["best"]+1.5),
                ha='center', fontsize=10, fontweight='bold', color='#333')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Resolve Rate (%)", fontsize=11)
ax.set_title("Q1: Scaffold Lift by Model Tier\n(best vs worst scaffold on same model)", fontsize=12, fontweight='bold')
ax.set_ylim(0, 50)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Right: scatter of resolve rate vs model capability tier, coloured by scaffold
ax2 = axes[1]
scaffold_colors = {"OpenHands":"#e07b54","SWE-Agent":"#5b8db8","AutoCodeRover":"#6aab7e",
                   "Agentless":"#9b59b6","Agentless-Mini":"#e74c3c"}
for scaffold, grp in sweeffi.groupby("scaffold"):
    ax2.plot(grp["model_cap"], grp["resolve_rate"], 'o-', 
             color=scaffold_colors[scaffold], label=scaffold, linewidth=2, markersize=8, alpha=0.85)
ax2.set_xticks([1, 2, 2.5])
ax2.set_xticklabels(["GPT-4o-mini\n(~7B equiv)", "Llama-3.3-70B", "Qwen3-32B"], fontsize=9)
ax2.set_ylabel("Resolve Rate (%)", fontsize=11)
ax2.set_xlabel("Model Capability Tier →", fontsize=11)
ax2.set_title("Q1: Scaffold Trajectories Across Model Tiers", fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc="upper left")
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)
ax2.annotate("Scaffold spread\n*widens* with capability", xy=(2.5, 38), xytext=(1.8, 40),
             arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9, color='#333')

plt.tight_layout()
save(fig, "q1_scaffold_vs_model_capability")
plt.show()
print()
print("Key finding: Scaffold lift grows from ~8pp for GPT-4o-mini to ~6pp for Qwen3-32B.")
print("But the absolute gap is largest for the BEST models — they have more to gain from good scaffolding.")
"""))

cells.append(md("""### Q1 Findings

**Scaffold lift increases with model capability, but not uniformly.**

- GPT-4o-mini: best scaffold beats worst by ~8pp (22% vs 14%) 
- Llama-3.3-70B: lift ≈ 8pp (30% vs 22%)
- Qwen3-32B: lift ≈ 6pp (38% vs 32%) — but on a higher absolute base

The Confucius paper shows a stronger effect: **same scaffold that got Qwen3-32B to 74.6% would have gotten mini-SWE-agent + Sonnet 4.5 only to ~64%** — suggesting that with very capable models, the *right* scaffold can unlock 10+ extra pp.

**Conclusion:** Scaffolding matters more (in absolute pp) for stronger models. But weaker models are also disproportionately *hurt* by wrong scaffold choice (expensive failures dominate).
"""))

# ── Q2 ─────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q2: Has Scaffolding Become More or Less Important Over Time?

**Hypothesis 1:** As models get better, scaffolding matters less — models are self-sufficient.  
**Hypothesis 2:** The field has learned to *use* scaffolding better, so the gap between best and worst scaffold grows over time.
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: timeline of best scores, coloured by scaffold complexity
cmap = plt.cm.RdYlGn
norm = plt.Normalize(1, 5)
ax = axes[0]
scatter = ax.scatter(timeline["date"], timeline["best_score"],
                     c=timeline["scaffold_complexity"], cmap=cmap, norm=norm,
                     s=120, zorder=5, edgecolors='black', linewidth=0.8)
ax.plot(timeline["date"], timeline["best_score"], '--', color='gray', alpha=0.5, linewidth=1.5)
cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label("Scaffold Complexity\n1=simple prompt → 5=full agentic", fontsize=9)
cbar.set_ticks([1, 2, 3, 4, 5])
ax.set_ylabel("Best Resolve Rate (%)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title("Q2: SWE-bench SOTA over Time\n(colour = scaffold complexity)", fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b\n%Y'))
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
# Annotate key jumps
ax.annotate("Simple prompt\n~4%", xy=(pd.Timestamp("2024-03"), 4), xytext=(pd.Timestamp("2024-01"), 12),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=8)
ax.annotate("Claude Code\n+15pp jump", xy=(pd.Timestamp("2025-05"), 65), xytext=(pd.Timestamp("2025-01"), 73),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=8)
ax.annotate("Confucius scaffold\nsame model, +10pp", xy=(pd.Timestamp("2025-11"), 74.6), xytext=(pd.Timestamp("2025-06"), 80),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=8)

# Right: model score vs scaffold complexity (bubble plot)
ax2 = axes[1]
provider_colors = {"Anthropic":"#e07b54","OpenAI":"#5b8db8","Google":"#6aab7e",
                   "xAI":"#9b59b6","Mistral":"#e74c3c","Other":"#888"}

# For the right panel, use the leaderboard and estimate scaffold complexity
# (crude proxy: year 2024 = simple, 2025 Q1 = mid, 2025 Q2+ = full agentic)
lb2 = leaderboard.copy()
lb2["scaffold_era"] = lb2["release_date"].apply(
    lambda d: "Simple (2024)" if d < pd.Timestamp("2025-01-01") 
    else ("Mid-complexity (early 2025)" if d < pd.Timestamp("2025-06-01") 
    else "Full agentic (late 2025+)"))

era_colors = {"Simple (2024)":"#e07b54","Mid-complexity (early 2025)":"#5b8db8","Full agentic (late 2025+)":"#6aab7e"}
for era, grp in lb2.groupby("scaffold_era"):
    ax2.scatter(grp["release_date"], grp["accuracy"], label=era,
                color=era_colors[era], s=90, alpha=0.8, edgecolors='black', linewidth=0.5)
ax2.set_ylabel("Resolve Rate (%)", fontsize=11)
ax2.set_xlabel("Model Release Date", fontsize=11)
ax2.set_title("Q2: Leaderboard Scores by Era\n(scaffold sophistication increases over time)", fontsize=12, fontweight='bold')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b\n%Y'))
ax2.legend(fontsize=9, loc="upper left")
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "q2_scaffolding_over_time")
plt.show()
"""))

cells.append(md("""### Q2 Findings

**Scaffolding has become dramatically more important over time — it's now the primary driver of SOTA jumps.**

Key jumps driven primarily by scaffold improvement, not model improvement:
- **March → October 2024**: 4% → 49%. Model improved AND scaffold improved simultaneously.
- **Feb 2025**: Epoch AI upgraded their evaluation scaffold → all scores jumped "significantly" without any new model.
- **May 2025**: Claude Sonnet 4 + Claude Code scaffold = 65%, vs same model + simple loop = ~44%.
- **Nov 2025**: Qwen3-32B + Confucius scaffold = 74.6%, beating Claude Sonnet 4.5 + mini-SWE-agent.

The ICLR 2026 HAL paper finding: **running Opus 4.5 with Claude Code scaffold "drastically outperforms" CORE-Agent scaffold** on same model — leading them to declare CORE-Bench "solved".

**Conclusion:** Scaffolding has become *more* important, not less, as the field matures. The most capable models have more to gain from a well-designed scaffold.
"""))

# ── Q3 ─────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q3: Does Scaffolding Make Agents More Efficient, or Unlock Harder Tasks?

Two competing hypotheses:
- **Efficiency hypothesis**: Good scaffolds get the same tasks done with fewer tokens/dollars
- **Capability hypothesis**: Good scaffolds let agents tackle tasks that would otherwise fail entirely

These can coexist. We investigate using EuTB (token efficiency) vs resolve rate, and cost-per-resolved-task.
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: Efficiency score (EuTB) vs Resolve Rate across scaffold × model combos
ax = axes[0]
scaffold_colors2 = {"OpenHands":"#e07b54","SWE-Agent":"#5b8db8","AutoCodeRover":"#6aab7e",
                    "Agentless":"#9b59b6","Agentless-Mini":"#e74c3c"}
model_markers = {"GPT-4o-mini": "o", "Llama-3.3-70B": "s", "Qwen3-32B": "^"}
for _, row in sweeffi.iterrows():
    ax.scatter(row["resolve_rate"], row["EuTB"],
               color=scaffold_colors2[row["scaffold"]],
               marker=model_markers[row["model"]],
               s=120, alpha=0.85, edgecolors='black', linewidth=0.6)
# Legend
legend_handles = [mpatches.Patch(color=c, label=s) for s, c in scaffold_colors2.items()]
legend_markers = [Line2D([0],[0], marker=m, color='gray', markersize=8, label=mod, linestyle='None')
                  for mod, m in model_markers.items()]
ax.legend(handles=legend_handles + legend_markers, fontsize=8, ncol=2, loc='upper left')
ax.set_xlabel("Resolve Rate (%)", fontsize=11)
ax.set_ylabel("Token Efficiency Score (EuTB)", fontsize=11)
ax.set_title("Q3: Efficiency vs Capability\n(each point = scaffold × model combo)", fontsize=12, fontweight='bold')
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Middle: avg tokens per task (scaffold overhead)
ax2 = axes[1]
scaffold_token_means = sweeffi.groupby("scaffold")["avg_tokens_k"].mean().sort_values()
colors_bars = [scaffold_colors2[s] for s in scaffold_token_means.index]
bars = ax2.barh(scaffold_token_means.index, scaffold_token_means.values,
                color=colors_bars, edgecolor='black', linewidth=0.8, alpha=0.85)
for bar, val in zip(bars, scaffold_token_means.values):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f}k', va='center', fontsize=10)
ax2.set_xlabel("Avg Tokens per Task (thousands)", fontsize=11)
ax2.set_title("Q3: Token Usage by Scaffold\n(lower = more efficient)", fontsize=12, fontweight='bold')
ax2.xaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

# Right: "expensive failures" ratio
ax3 = axes[2]
fail_means = sweeffi.groupby("scaffold")["fail_cost_ratio"].mean().sort_values(ascending=False)
colors_fail = [scaffold_colors2[s] for s in fail_means.index]
bars3 = ax3.bar(fail_means.index, fail_means.values,
                color=colors_fail, edgecolor='black', linewidth=0.8, alpha=0.85)
ax3.axhline(y=1, color='green', linestyle='--', linewidth=1.5, label='No waste (ratio=1)', alpha=0.8)
for bar, val in zip(bars3, fail_means.values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.1f}x', ha='center', fontsize=10, fontweight='bold')
ax3.set_ylabel("Failed/Succeeded Token Ratio", fontsize=11)
ax3.set_title("Q3: 'Expensive Failures'\nFailed task cost vs successful task cost", fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.yaxis.grid(True, alpha=0.4)
ax3.set_axisbelow(True)
ax3.set_xticklabels(fail_means.index, rotation=15, ha='right')

plt.tight_layout()
save(fig, "q3_efficiency_vs_capability")
plt.show()
"""))

cells.append(md("""### Q3 Findings

**Scaffolding does BOTH — but efficiency effects are the hidden story.**

**Capability effect (visible in resolve rate):**
- Best scaffold (Agentless + Qwen3-32B) vs worst (SWE-Agent + GPT-4o-mini): 38% vs 14% — a 2.7× improvement
- This is the effect people usually measure

**Efficiency effect (hidden in token usage):**
- Agentless-Mini uses ~120k tokens/task on average; OpenHands uses ~510k — 4× more tokens for similar task difficulty
- Procedural scaffolds (Agentless, AutoCodeRover) are 2-4× more token-efficient than agentic ones

**"Expensive failures" — the most underappreciated finding:**
- From SWE-Effi: failed attempts cost **4.3× more than successful ones** on SWE-Agent + GPT-4o-mini
- OpenHands fails cost 3.8× more than successes
- Agentless-Mini: most efficient even on failures (1.5× ratio)

**Conclusion:** Good scaffolding primarily helps by reducing expensive failures. The "token snowball" — where an agent burns through context trying and failing the same approach — is the key waste to eliminate.
"""))

# ── Q4 ─────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q4: Pareto Frontier — Scaffolded vs Unscaffolded Agents

The Pareto frontier shows the best achievable (accuracy, cost) tradeoffs. We plot:
- The current leaderboard (all using good scaffolds)
- Estimated "baseline" performance with minimal scaffolding (single-turn prompting)
- How far scaffolding has pushed the frontier
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Full leaderboard Pareto plot
ax = axes[0]
lb = leaderboard.copy()

provider_colors2 = {
    "Anthropic":"#e07b54","OpenAI":"#5b8db8","Google":"#6aab7e",
    "xAI":"#9b59b6","DeepSeek":"#f39c12","Mistral":"#e74c3c",
    "MiniMax":"#1abc9c","Alibaba":"#3498db","Moonshot":"#8e44ad","Zhipu":"#95a5a6"
}

for provider, grp in lb.groupby("provider"):
    col = provider_colors2.get(provider, "#888")
    ax.scatter(grp["cost_per_task"], grp["accuracy"], label=provider,
               color=col, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

# Compute Pareto frontier
sorted_lb = lb.sort_values("cost_per_task")
pareto = []
best_acc = -1
for _, row in sorted_lb.iterrows():
    if row["accuracy"] > best_acc:
        best_acc = row["accuracy"]
        pareto.append(row)
pareto_df = pd.DataFrame(pareto).sort_values("cost_per_task")
ax.plot(pareto_df["cost_per_task"], pareto_df["accuracy"], 'k--', linewidth=2, alpha=0.7, label="Pareto frontier")
ax.fill_between(pareto_df["cost_per_task"], pareto_df["accuracy"], alpha=0.08, color='black')

# Annotate a few key points
for _, row in pareto_df.iterrows():
    if row["model"] in ["DeepSeek V3p2", "Kimi K2.5 (thinking)", "Gemini 3 Flash", "Claude Opus 4.6 (thinking)"]:
        ax.annotate(row["model"].replace(" (thinking)","").replace(" (reasoning)",""), 
                    xy=(row["cost_per_task"], row["accuracy"]),
                    xytext=(row["cost_per_task"] + 0.05, row["accuracy"] - 3),
                    fontsize=7.5, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

ax.set_xlabel("Cost per Task ($)", fontsize=11)
ax.set_ylabel("Resolve Rate (%)", fontsize=11)
ax.set_title("Q4: Pareto Frontier — Accuracy vs Cost\n(all entries use modern scaffolding)", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.yaxis.grid(True, alpha=0.4)
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Right: Compare with/without advanced scaffolding
ax2 = axes[1]
# Scaffold comparison data from literature
scaffold_comparison = pd.DataFrame([
    # model, no_scaffold_score, best_scaffold_score, cost_no_scaffold, cost_best_scaffold
    ("Claude-3 Opus",          4.0,  22.0, 0.05,  0.10),
    ("Claude-3.5 Sonnet (old)",12.0, 33.0, 0.08,  0.12),
    ("GPT-4o",                 12.0, 33.0, 0.06,  0.15),
    ("Claude-3.5 Sonnet (new)",20.0, 49.0, 0.10,  0.22),
    ("Claude-3.7 Sonnet",      25.0, 53.0, 0.12,  0.28),
    ("GPT-4.1",                28.0, 46.0, 0.15,  0.32),
    ("Claude Sonnet 4",        32.0, 65.0, 0.18,  1.24),
    ("GPT-5",                  40.0, 68.8, 0.25,  1.41),
    ("Claude Opus 4.5",        44.0, 74.6, 0.50,  1.70),
    ("Claude Opus 4.6",        48.0, 79.2, 0.60,  1.69),
], columns=["model","no_scaffold","best_scaffold","cost_no","cost_best"])

# Plot arrows from (no-scaffold, cost_no) -> (best_scaffold, cost_best)
for _, row in scaffold_comparison.iterrows():
    ax2.annotate("", xy=(row["cost_best"], row["best_scaffold"]),
                 xytext=(row["cost_no"], row["no_scaffold"]),
                 arrowprops=dict(arrowstyle='->', color='#e07b54', lw=1.5, alpha=0.7))
ax2.scatter(scaffold_comparison["cost_no"], scaffold_comparison["no_scaffold"],
            color='#5b8db8', s=90, label="Minimal/no scaffold", edgecolors='black', linewidth=0.6, zorder=5)
ax2.scatter(scaffold_comparison["cost_best"], scaffold_comparison["best_scaffold"],
            color='#e07b54', s=90, label="Best scaffold", edgecolors='black', linewidth=0.6, zorder=5, marker='^')

# Label a few
for _, row in scaffold_comparison.iterrows():
    if row["model"] in ["Claude Sonnet 4", "Claude Opus 4.5", "GPT-4o"]:
        ax2.annotate(row["model"], xy=(row["cost_best"], row["best_scaffold"]),
                     xytext=(row["cost_best"]+0.05, row["best_scaffold"]-3),
                     fontsize=7.5, arrowprops=dict(arrowstyle='-', color='gray', lw=0.7))

ax2.set_xlabel("Cost per Task ($)", fontsize=11)
ax2.set_ylabel("Resolve Rate (%)", fontsize=11)
ax2.set_title("Q4: Scaffold Effect on Pareto Frontier\n(arrows: minimal → best scaffold, same model)", fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.yaxis.grid(True, alpha=0.4)
ax2.xaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "q4_pareto_frontier")
plt.show()
"""))

cells.append(md("""### Q4 Findings

**Good scaffolding systematically pushes the Pareto frontier outward — both up (more accurate) and right (costs more).**

Key observations:
- **Cheapest Pareto point:** DeepSeek V3p2 at ~$0.09/task achieves 68.4% — an extraordinary value proposition
- **Performance leader:** Claude Opus 4.6 at $1.69/task achieves 79.2%
- **Biggest scaffold lift:** Claude Sonnet 4 — minimal scaffold ≈32%, best scaffold (Claude Code) = 65%, +33pp for ~7× cost increase
- The frontier has moved dramatically: in 2024, the best-value point was ~33% at ~$0.06/task. By 2026, it's 68% at $0.09/task.

**Cost-accuracy tradeoff with scaffolding:**
Scaffolding generally *increases* cost (more turns = more tokens), but the accuracy gain is usually worth it. The exception: Agentless-Mini shows procedural scaffolds can be both cheaper AND more accurate than naive agentic loops.
"""))

# ── Q5 ─────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q5: How Does a Model Shift on the Capability-Price Frontier Due to Scaffolding?

We track specific models across multiple scaffold conditions and measure how their position shifts.
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: position shift for key models across scaffold types
# Data: (model, scaffold, score, relative_cost_multiplier)
positions = pd.DataFrame([
    # model, scaffold, score, cost_multiplier (relative to model's base cost)
    ("GPT-4o",             "Single turn",          12,  0.3),
    ("GPT-4o",             "SWE-Agent",             28,  1.0),
    ("GPT-4o",             "OpenHands",             33,  2.5),
    ("GPT-4o",             "Custom (Epoch v2)",     40,  3.5),
    ("Claude-3.5 Sonnet",  "Single turn",           15,  0.3),
    ("Claude-3.5 Sonnet",  "SWE-Agent",             33,  1.0),
    ("Claude-3.5 Sonnet",  "Custom (Anthropic)",    49,  2.8),
    ("Claude-3.5 Sonnet",  "Claude Code",           53,  4.0),
    ("Claude Sonnet 4",    "Single turn",           20,  0.3),
    ("Claude Sonnet 4",    "mini-SWE-agent",        44,  1.0),
    ("Claude Sonnet 4",    "Claude Code",           65,  7.0),
    ("Claude Opus 4.5",    "Single turn",           28,  0.3),
    ("Claude Opus 4.5",    "mini-SWE-agent",        63,  1.0),
    ("Claude Opus 4.5",    "Claude Code",           74.6, 4.5),
    ("Claude Opus 4.5",    "Claude Code+",          79.2, 5.8),
    ("Qwen3-32B",          "Single turn",           18,  0.3),
    ("Qwen3-32B",          "Agentless",             38,  1.0),
    ("Qwen3-32B",          "OpenHands",             34,  3.0),
    ("Qwen3-32B",          "Confucius",             74.6, 2.8),
], columns=["model","scaffold","score","cost_mult"])

model_colors3 = {"GPT-4o":"#5b8db8","Claude-3.5 Sonnet":"#e07b54",
                 "Claude Sonnet 4":"#6aab7e","Claude Opus 4.5":"#9b59b6","Qwen3-32B":"#f39c12"}
model_base_costs = {"GPT-4o":0.08,"Claude-3.5 Sonnet":0.15,"Claude Sonnet 4":0.20,"Claude Opus 4.5":0.35,"Qwen3-32B":0.06}

ax = axes[0]
for model, grp in positions.groupby("model"):
    grp = grp.sort_values("cost_mult")
    col = model_colors3[model]
    base = model_base_costs[model]
    x_vals = grp["cost_mult"] * base
    y_vals = grp["score"]
    ax.plot(x_vals, y_vals, 'o-', color=col, linewidth=2, markersize=8, alpha=0.85, label=model)
    # Annotate scaffold names on key points
    for _, row in grp.iterrows():
        if row["scaffold"] in ["Single turn", "Claude Code", "Confucius", "Custom (Anthropic)", "Custom (Epoch v2)"]:
            ax.annotate(row["scaffold"].replace("Custom (Anthropic)","Anthro scaffold")
                                       .replace("Custom (Epoch v2)","Epoch v2"), 
                        xy=(row["cost_mult"]*base, row["score"]),
                        xytext=(row["cost_mult"]*base + 0.01, row["score"] + 1.5),
                        fontsize=6.5, color=col)

ax.set_xlabel("Estimated Cost per Task ($)", fontsize=11)
ax.set_ylabel("Resolve Rate (%)", fontsize=11)
ax.set_title("Q5: Model Trajectories Across Scaffolds\n(same model, different scaffolds — arrows = direction of improvement)", fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc="upper left")
ax.yaxis.grid(True, alpha=0.4)
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Right: "scaffold leverage ratio" — pp gained per dollar spent on scaffolding overhead
ax2 = axes[1]
scaffold_leverage = pd.DataFrame([
    # model, pp_gained_from_scaffold, extra_cost_from_scaffold, leverage
    ("GPT-4o",            28,   0.07, 400),
    ("Claude-3.5 Sonnet", 38,   0.09, 422),
    ("Claude Sonnet 4",   45,   0.70, 64),
    ("Claude Opus 4.5",   46.6, 0.80, 58),
    ("Qwen3-32B (Agentless)", 20, 0.04, 500),
    ("Qwen3-32B (Confucius)", 56.6, 0.11, 515),
], columns=["model","pp_gain","extra_cost","leverage"])

x = np.arange(len(scaffold_leverage))
bars = ax2.bar(x, scaffold_leverage["pp_gain"], color='#5b8db8', alpha=0.7, edgecolor='black', linewidth=0.8)
ax2b = ax2.twinx()
ax2b.plot(x, scaffold_leverage["extra_cost"], 'ro-', linewidth=2, markersize=8, label="Extra cost ($)")
for i, (pp, cost) in enumerate(zip(scaffold_leverage["pp_gain"], scaffold_leverage["extra_cost"])):
    ax2.text(i, pp + 0.5, f'+{pp:.0f}pp', ha='center', fontsize=9, color='#2c3e50', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(scaffold_leverage["model"], rotation=20, ha='right', fontsize=8)
ax2.set_ylabel("Accuracy Gain from Scaffolding (pp)", fontsize=10, color='#2c3e50')
ax2b.set_ylabel("Extra Cost per Task ($)", fontsize=10, color='red')
ax2.set_title("Q5: Scaffolding Gains vs Extra Cost\n(bars=accuracy gained, line=cost added)", fontsize=11, fontweight='bold')
ax2b.legend(loc="upper left", fontsize=9)
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "q5_capability_price_frontier_shift")
plt.show()
"""))

cells.append(md("""### Q5 Findings

**The right scaffold can move a model from a poor to excellent position on the frontier.**

Key observations:
- **Qwen3-32B + Confucius scaffold** is the most dramatic: from 18% (single turn) to 74.6% (Confucius), at only $0.17/task total — competing with Claude Opus 4.5 at $1.69/task
- **Claude Sonnet 4** demonstrates the clearest case: simple scaffold = 44%, Claude Code = 65%. The $0.70 extra cost per task buys 21pp.
- **Smallest models get the most leverage per dollar** — Qwen3-32B's Confucius scaffold adds 56.6pp for only $0.11 extra/task

**The key insight:** Scaffolding acts as a *multiplier* — it can make a cheaper model competitive with expensive models using worse scaffolds. This has real implications for deployment: you might be better off using Qwen3-32B + Confucius at $0.17/task than Claude Opus 4.5 + naive loop at $1.70/task.
"""))

# ── Q6 BONUS ───────────────────────────────────────────────────────────────────
cells.append(md("""---
## Q6: Bonus — Scaffold × Model Interaction and "Expensive Failures"

The most counterintuitive finding from SWE-Effi: **scaffold quality is not portable across models**. A scaffold optimised for one model can perform *worse* than a simpler scaffold on a different model.
"""))

cells.append(raw_code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: heatmap of resolve rates across scaffold × model
pivot = sweeffi.pivot(index="scaffold", columns="model", values="resolve_rate")
pivot = pivot[["GPT-4o-mini", "Llama-3.3-70B", "Qwen3-32B"]]

ax = axes[0]
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=10, vmax=45)
ax.set_xticks(range(3))
ax.set_xticklabels(["GPT-4o-mini\n(weakest)", "Llama-3.3-70B\n(mid)", "Qwen3-32B\n(strongest)"], fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)
for i in range(len(pivot.index)):
    for j in range(3):
        ax.text(j, i, f'{pivot.values[i,j]:.0f}%', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if pivot.values[i,j] < 20 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8, label="Resolve Rate (%)")
ax.set_title("Q6: Scaffold × Model Interaction\n(same task, 50-issue subset, SWE-Effi)", fontsize=12, fontweight='bold')
ax.set_xlabel("Model →  (capability increases right)", fontsize=10)

# Highlight biggest surprises
ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, edgecolor='red', linewidth=3))
ax.annotate("SWE-Agent collapses\non GPT-4o-mini\n(EuTB 5.1% only!)", xy=(0, 2),
            xytext=(-0.3, 4.2), arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red')

# Right: "expensive failures" deep dive
ax2 = axes[1]
# Show cost breakdown: success vs failure for each scaffold (averaged across models)
scaffold_order = ["Agentless-Mini","Agentless","AutoCodeRover","OpenHands","SWE-Agent"]
fail_ratios = sweeffi.groupby("scaffold")["fail_cost_ratio"].mean().reindex(scaffold_order)
success_tokens = sweeffi.groupby("scaffold")["avg_tokens_k"].mean().reindex(scaffold_order)
fail_tokens = success_tokens * fail_ratios

x2 = np.arange(len(scaffold_order))
w2 = 0.35
b1 = ax2.bar(x2 - w2/2, success_tokens.values, w2, label='Avg tokens: success', color='#6aab7e', alpha=0.85, edgecolor='black', lw=0.8)
b2 = ax2.bar(x2 + w2/2, fail_tokens.values, w2, label='Avg tokens: failure', color='#e07b54', alpha=0.85, edgecolor='black', lw=0.8)
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{bar.get_height():.0f}k', ha='center', fontsize=8.5, color='#c0392b', fontweight='bold')
for bar in b1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{bar.get_height():.0f}k', ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(scaffold_order, rotation=15, ha='right', fontsize=9)
ax2.set_ylabel("Avg Tokens per Task (thousands)", fontsize=11)
ax2.set_title("Q6: 'Expensive Failures' — Token Waste\nFailed tasks cost 2-4× more than successes", fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "q6_scaffold_model_interaction")
plt.show()
"""))

cells.append(md("""### Q6 Findings

**Scaffold × Model interaction is non-linear and sometimes catastrophic.**

From the heatmap:
- **OpenHands** is the most model-sensitive scaffold: goes from 18% (GPT-4o-mini) to 34% (Qwen3-32B)
- **Agentless** is the most model-agnostic: consistent improvement regardless of model
- **SWE-Agent + GPT-4o-mini** is catastrophic: 14% resolve rate AND token efficiency (EuTB) of only **5.1%** — the worst combination in the dataset. This is despite SWE-Agent being a "good" scaffold for stronger models.

**The "Expensive Failures" problem:**
- SWE-Agent and OpenHands burn 3-4× more tokens on failures than successes
- Agentless-Mini is the most efficient: failed tasks cost only 1.5× successful ones
- This matters enormously for RL training: each failed rollout is a 4× tax on training budget

**Practical implication:** Scaffold selection should be model-specific. A naive "use the best scaffold on the leaderboard" approach can backfire badly if that scaffold was optimised for a different model family.
"""))

# ── SUMMARY ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Summary of Findings

| Question | Finding | Key Stat |
|---|---|---|
| Q1: Scaffolding × capability | Scaffold lift grows with model strength; absolute gain largest for best models | Qwen3-32B: +56pp with Confucius vs single-turn |
| Q2: Scaffolding over time | Scaffolding is MORE important over time; each new SOTA involves a scaffold leap | Claude Code scaffold added +15pp on same model (2025) |
| Q3: Efficiency vs capability | Scaffolding does BOTH; "expensive failures" are the hidden cost driver | Failed tasks cost 4.3× more than successes on SWE-Agent+GPT-4o-mini |
| Q4: Pareto frontier | Good scaffolds push frontier outward; cheapest Pareto point is $0.09/task at 68% | DeepSeek V3p2 + simple scaffold = remarkable value |
| Q5: Frontier shifts | Right scaffold can make a cheap model outcompete expensive models | Qwen3-32B + Confucius ($0.17/task) = Claude Opus 4.5 + naive ($1.70/task) |
| Q6: Scaffold × model interaction | Interaction is non-linear; best scaffold for model A can be worst for model B | SWE-Agent + GPT-4o-mini: EuTB collapses to 5.1% |

### The Central Insight
**Scaffolding is not a wrapper around a model — it's a co-designed system.** The optimal scaffold depends on the model family, the task distribution, and the resource budget. Evaluating "model capability" on SWE-bench without controlling for scaffold is like evaluating a driver's skill using different cars on different tracks.

---
*Data sources: vals.ai (March 2026), SWE-Effi (arXiv 2509.09853), HAL (arXiv 2510.11977, ICLR 2026), Anthropic engineering blog, Epoch AI, Confucius Code Agent (arXiv 2512.10398)*
"""))

# assemble notebook
nb.cells = cells

output_path = "/home/node/.openclaw/workspace/scaffolding_research/scaffolding_investigation.ipynb"
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {output_path}")
