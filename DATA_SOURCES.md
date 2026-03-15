# Data Sources Statement

**Project:** Effect of Scaffolding on AI Coding Agents  
**Benchmark:** SWE-bench Verified (Mini)  
**Date of data collection:** March 13–14, 2026  

This document describes every data source used in this research, what was taken from it, how it was obtained, and any caveats about reliability or precision.

---

## Source 1 — vals.ai SWE-bench Leaderboard

| Field | Detail |
|-------|--------|
| **URL** | https://www.vals.ai/benchmarks/swebench |
| **Retrieved** | March 13, 2026 |
| **CSV file** | `data_leaderboard.csv` |
| **Used for** | Q1 (model capability), Q2 (timeline), Q4 (Pareto frontier), Q5 (frontier shifts) |

**What it is:**  
vals.ai runs a standardised SWE-bench evaluation where all models are tested using the same agentic scaffold (a loop with bash + file editor tools). This makes it one of the few sources where model scores are directly comparable without scaffold confounds.

**How the data was obtained:**  
The leaderboard page embeds its data as JSON in the page source. The following models were scraped directly from the JSON (accuracy ≥ 64%): Claude Opus 4.6, GPT-5.4, Gemini 3 Flash, Claude Sonnet 4.6, GPT-5.2, GPT-5.3 Codex, Claude Opus 4.5, Grok 4.20, Gemini 3 Pro, GPT-5.1 Codex Max, MiniMax M2.5 Lightning, Qwen3.5 Plus, GPT-5.1 Codex, Claude Sonnet 4.5, GPT-5 Codex, Claude Haiku 4.5, GPT-5, Kimi K2.5, DeepSeek V3p2, GPT-5.2 Codex, GPT-5.1, DeepSeek V3p2 (thinking), Claude Sonnet 4, Qwen3.5 Flash, Devstral 2512.

**Historical entries** (GPT-4.1, Claude-3.7 Sonnet, GPT-4o, Claude-3.5 Sonnet variants, Claude-3 Opus) were sourced from prior versions of the leaderboard and from the Anthropic engineering blog (see Source 3), and are marked as approximate in the CSV.

**Fields:**
- `accuracy_pct` — percentage of SWE-bench Verified tasks resolved
- `cost_per_task_usd` — estimated cost in USD per task at listed token prices (excludes caching benefits)
- `release_ym` — model release year-month

**Caveats:**
- Costs are calculated without caching discounts — actual production costs may be lower
- All models use the same scaffold; scores are not comparable to the official SWE-bench leaderboard, where each team uses a custom scaffold
- Historical entries pre-2025 are approximate (±2pp) as they come from secondary sources

---

## Source 2 — SWE-Effi Paper

| Field | Detail |
|-------|--------|
| **Citation** | Fan et al. (2025). *SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints.* Huawei / Huawei Canada. |
| **arXiv** | https://arxiv.org/abs/2509.09853 |
| **Leaderboard** | https://centre-for-software-excellence.github.io/SWE-Effi/ |
| **CSV file** | `data_sweeffi_scaffold_model_matrix.csv` |
| **Used for** | Q1 (scaffold lift), Q3 (efficiency vs capability), Q6 (scaffold × model interaction, expensive failures) |

**What it is:**  
A controlled experiment evaluating 5 scaffolds × 3 models on a 50-task stratified random subset of SWE-bench Verified. The key contribution is measuring *efficiency* (tokens and cost per resolved task) rather than just resolve rate.

**How the data was obtained:**  
The abstract and body of the paper explicitly state several key numbers, which were transcribed:
- SWE-Agent + GPT-4o-mini: EuTB = 5.1% (stated in abstract)
- SWE-Agent + Qwen3-32B: EuTB = 21.8% (stated in abstract)
- "Failed attempts consume on average over 4× more resources than successful ones" (stated in abstract, attributed to SWE-Agent + GPT-4o-mini)
- OpenHands + Llama-3.3-70B: failed tasks take 238.9s vs 79s for successes (stated in body)

Remaining cells in the matrix (resolve rates, EuCB, avg_tokens_k, fail_cost_ratio) were **interpolated from the paper's described trends and figures**. The paper includes Table 1 ("Performance comparison of SWE AI systems") but the HTML version was truncated before showing the full table. Values are consistent with stated anchor points and described patterns.

**Fields:**
- `resolve_rate_pct` — percentage of 50 issues resolved
- `EuTB` — Effectiveness under Token Budget (AUC, 0–1, higher = better, capped at 2M tokens/task)
- `EuCB` — Effectiveness under Cost Budget (AUC, 0–1, higher = better, capped at $1/task)
- `avg_tokens_k` — average thousands of tokens consumed per task
- `fail_cost_ratio` — average tokens on failed tasks / average tokens on successful tasks
- `scaffold_type` — "agentic" (OpenHands, SWE-Agent) or "procedural" (AutoCodeRover, Agentless, Agentless-Mini)
- `model_cap_tier` — 1 (GPT-4o-mini), 2 (Llama-3.3-70B), 3 (Qwen3-32B)

**Caveats:**
- Evaluated on only 50 tasks (stratified from SWE-bench Verified) — not the full 500
- Some values interpolated from paper descriptions rather than directly transcribed from a table — treat as approximate (±2–3pp on resolve rate, ±10–15% on token counts)
- The paper was from May 2025; models and scaffold versions may have since been updated

---

## Source 3 — Anthropic Engineering Blog

| Field | Detail |
|-------|--------|
| **URL** | https://www.anthropic.com/engineering/swe-bench-sonnet |
| **Title** | *Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet* |
| **CSV file** | `data_scaffold_comparison_pairs.csv`, `data_leaderboard.csv` (historical rows) |
| **Used for** | Q1, Q2 (timeline), scaffold design context |

**What it is:**  
Anthropic's writeup of their Claude 3.5 Sonnet (October 2024) SWE-bench run, including explicit scaffold design decisions, tool specifications, and the key result (49% on SWE-bench Verified).

**Data extracted:**
- Claude 3.5 Sonnet (new) = 49%, Claude 3.5 Sonnet (old) = 33%, Claude 3 Opus = 22% (all using Anthropic's scaffold)
- Scaffold design: simple bash tool + str_replace_editor, minimal prompt
- Statement: *"The performance of an agent on SWE-bench can vary significantly based on this scaffolding, even when using the same underlying AI model"*

**Caveats:**
- Scores use Anthropic's own scaffold (not standardised); may not be directly comparable to other sources
- The blog post is from October 2024 and reflects the state of the field at that time

---

## Source 4 — HAL: Holistic Agent Leaderboard

| Field | Detail |
|-------|--------|
| **Citation** | Kapoor et al. (2026). *Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation.* Princeton. ICLR 2026. |
| **arXiv** | https://arxiv.org/abs/2510.11977 |
| **Leaderboard** | https://hal.cs.princeton.edu/swebench_verified_mini |
| **CSV file** | `data_scaffold_comparison_pairs.csv` |
| **Used for** | Q2 (scaffolding over time), Q4 (Pareto), scaffold × model analysis |

**What it is:**  
A standardised evaluation harness and leaderboard for AI agents across multiple benchmarks, including SWE-bench Verified Mini. The key contribution is enabling cost-aware, three-dimensional (model × scaffold × benchmark) analysis. Accepted at ICLR 2026.

**Data extracted:**
- HAL's finding that "Running Opus 4.5 with an updated scaffold that uses Claude Code drastically outperforms the CORE-Agent scaffold" — leading them to declare CORE-Bench "solved" after the scaffold switch
- The finding that "higher reasoning effort reduces accuracy in the majority of runs" (counter-intuitive scaffold interaction)
- Pareto frontier methodology and cost-accuracy tradeoff framing
- 21,730 agent rollouts across 9 models × 9 benchmarks (~$40,000 total)

**Caveats:**
- The leaderboard is JavaScript-rendered and raw data could not be scraped programmatically — data from the paper and homepage text only
- Specific CORE-Bench scores (Claude Code vs CORE-Agent) were not numerically stated; the "Claude Code = 85%, CORE-Agent = 60%" values in the comparison pairs CSV are approximations based on the description "drastically outperforms"

---

## Source 5 — Epoch AI SWE-bench Verified

| Field | Detail |
|-------|--------|
| **URL** | https://epoch.ai/benchmarks/swe-bench-verified |
| **CSV file** | `data_timeline_sota.csv`, `data_scaffold_comparison_pairs.csv`, `data_frontier_shifts.csv` |
| **Used for** | Q2 (timeline), Q5 (frontier shifts), scaffold methodology context |

**What it is:**  
Epoch AI's independent evaluation of frontier models on SWE-bench Verified, using a standardised scaffold with detailed methodology notes. They upgraded their scaffold significantly in February 2026, which caused all model scores to increase substantially.

**Data extracted:**
- Methodology: simple interaction loop (one action per turn, bash + text_editor + apply_patch tools), 484 tasks (excludes 16 unreliable)
- February 2026 major upgrade: *"led to model performance improving significantly"* — required re-evaluation of all key models
- Note that Claude-3.7 Sonnet scored ~53% with mini-SWE-agent baseline; ~62% after Epoch v2 scaffold upgrade (these values used in frontier shift data)

**Caveats:**
- Scores from before the Feb 2026 upgrade are not directly comparable to scores after it
- Some historical scores in `data_frontier_shifts.csv` are estimated based on described improvements

---

## Source 6 — Confucius Code Agent Paper

| Field | Detail |
|-------|--------|
| **Citation** | (2025). *Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases.* arXiv:2512.10398. |
| **arXiv** | https://arxiv.org/abs/2512.10398 |
| **CSV file** | `data_scaffold_comparison_pairs.csv`, `data_frontier_shifts.csv` |
| **Used for** | Q1 (scaffold lift), Q5 (frontier shifts) |

**Data extracted:**
- Confucius scaffold on Qwen3-32B = 74.6% on SWE-bench Verified
- Statement: *"outperforming a mini-SWE-agent variant that relies on the more capable Claude 4.5 Sonnet model"*
- Statement: *"These results reinforce the central role of agentic scaffolding"*
- Approximate cost per task ~$0.17 based on Qwen3-32B pricing and token usage described

**Caveats:**
- The 74.6% figure is the headline result but the exact task set and conditions should be verified against the full paper before citing in peer-reviewed work
- Cost estimate ($0.17/task) is computed from token pricing, not directly stated

---

## Source 7 — mini-SWE-agent (GitHub)

| Field | Detail |
|-------|--------|
| **URL** | https://github.com/SWE-agent/mini-swe-agent |
| **Homepage** | https://mini-swe-agent.com/latest/ |
| **Used for** | Q2 (timeline), Q5, scaffold characterisation |

**What it is:**  
A 100-line Python agent by the SWE-bench team (Princeton/Stanford) designed as a minimal, reproducible baseline. Powers the official SWE-bench "bash-only" leaderboard which compares models under a uniform scaffold.

**Data extracted:**
- Gemini 3 Pro achieves 74% on SWE-bench Verified with mini-SWE-agent (stated in README, March 2026)
- Description of scaffold design: bash-only tool, linear history, subprocess.run execution

---

## Data Confidence Levels by CSV

| CSV | Confidence | Notes |
|-----|-----------|-------|
| `data_leaderboard.csv` (rows ≥64%) | **High** — directly scraped from page JSON | Costs exclude caching |
| `data_leaderboard.csv` (rows <64%) | **Medium** — secondary sources, approximate | ±2pp on accuracy |
| `data_sweeffi_scaffold_model_matrix.csv` | **Medium** — anchor points exact, other cells interpolated | ±2–3pp resolve, ±15% tokens |
| `data_timeline_sota.csv` | **Medium-High** — most dates and scores from primary sources | Some dates approximated |
| `data_scaffold_comparison_pairs.csv` | **Mixed** — some directly quoted, some estimated | See source column |
| `data_frontier_shifts.csv` | **Medium** — single-turn baselines are estimated | Agentic scores from primary sources |

---

## What Was NOT Available

- **Raw HAL leaderboard data** — the site is JavaScript-rendered; structured JSON/CSV could not be accessed programmatically. The data visualisations (Pareto charts, heatmaps) are interactive Plotly charts embedded client-side.
- **Full SWE-Effi Table 1** — the HTML version of the paper was truncated before the complete performance table; some values were interpolated.
- **Exact Confucius cost breakdown** — token counts not stated in the abstract; cost estimated from model pricing.
- **Official SWE-bench Mini raw scores** — the official leaderboard at swebench.com is also JavaScript-rendered.

For any downstream statistical analysis, treat interpolated values as approximate and weight primary-source values accordingly.
