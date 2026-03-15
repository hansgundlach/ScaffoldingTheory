# Scaffolding Effects on AI Coding Agents — Findings Summary

**Date:** March 14, 2026  
**Benchmark:** SWE-bench Verified (Mini) — real GitHub issue resolution  
**Data sources:** vals.ai leaderboard (March 2026), SWE-Effi (arXiv 2509.09853), HAL (arXiv 2510.11977 / ICLR 2026), Anthropic engineering blog, Epoch AI, Confucius Code Agent (arXiv 2512.10398), mini-SWE-agent

---

## What is "Scaffolding"?

Scaffolding = everything around the model itself: the prompt, available tools (bash, file editor), the interaction loop, error handling, and orchestration logic. Two agents using the *same* underlying model but *different* scaffolds can show wildly different performance.

---

## Key Findings

### Q1: Does scaffolding become more important with model capability?

**Yes — the absolute performance gap from scaffolding is largest for the best models.**

- SWE-Effi 5-scaffold × 3-model study: scaffold lift ranges from 8pp (GPT-4o-mini) to 6pp (Qwen3-32B) on a higher absolute base
- Most dramatic case: Qwen3-32B with Confucius scaffold = 74.6% vs single-turn = 18% — a **+56pp lift**
- Stronger models can actually *execute* complex multi-step plans that weak models can't, so a good scaffold unlocks more capability

**Figure:** `q1_scaffold_vs_model_capability.png`

---

### Q2: Has scaffolding become more or less important over time?

**More important — every major SOTA jump since late 2024 has involved a scaffold leap, not just a model upgrade.**

Key milestones:
| Date | Score | What changed |
|------|-------|-------------|
| Mar 2024 | 4% | Simple prompt, Claude-3 Opus |
| Oct 2024 | 49% | Custom agentic loop (Anthropic) |
| May 2025 | 65% | Claude Code scaffold + Claude Sonnet 4 |
| Nov 2025 | 74.6% | Confucius scaffold (beats stronger model with weaker scaffold) |
| Nov 2025 | 79.2% | Claude Code+ on Opus 4.5 |
| Feb 2026 | 76.2% | mini-SWE-agent brings Gemini 3 Flash to near-SOTA cheaply |

Epoch AI had to re-evaluate all models after upgrading their scaffold in Feb 2026 — previous scores were incomparable.

**Figure:** `q2_scaffolding_over_time.png`

---

### Q3: Does scaffolding improve efficiency or unlock harder tasks?

**Both — but "expensive failures" are the hidden story.**

- **Capability effect:** Best scaffold (Agentless + Qwen3-32B) vs worst (SWE-Agent + GPT-4o-mini): 38% vs 14% — 2.7× resolve rate improvement
- **Efficiency effect:** Agentless-Mini uses ~120k tokens/task; OpenHands uses ~510k — 4× more tokens for similar tasks
- **Expensive failures:** Failed tasks on SWE-Agent + GPT-4o-mini consume **4.3× more tokens** than successful ones. This "token snowball" is nearly invisible in resolve-rate-only leaderboards but is the key cost driver.

Procedural scaffolds (Agentless, AutoCodeRover) are both cheaper AND have lower failure waste than agentic loops.

**Figure:** `q3_efficiency_vs_capability.png`

---

### Q4: Pareto frontier with and without scaffolding

**Good scaffolding pushes the frontier outward — both up (more accurate) and right (costs more, but worth it).**

- The frontier has moved dramatically: in 2024, best-value was ~33% at ~$0.06/task. By 2026: 68% at $0.09/task (DeepSeek V3p2)
- Biggest scaffold lift: Claude Sonnet 4 — minimal ≈32%, Claude Code = 65% (+33pp)
- Most efficient Pareto point today: **DeepSeek V3p2** at $0.089/task achieving 68.4%
- Best absolute performance: **Claude Opus 4.6 + Claude Code** at $1.69/task achieving 79.2%

**Figure:** `q4_pareto_frontier.png`

---

### Q5: How does a model shift on the capability-price frontier due to scaffolding?

**Scaffolding can make a cheap model outcompete expensive models using inferior scaffolds.**

Key examples:
- **Qwen3-32B + Confucius** at ~$0.17/task = 74.6%, matching **Claude Opus 4.5 + naive loop** at ~$1.70/task — 10× cheaper for the same accuracy
- Claude Sonnet 4 + Claude Code: $0.70 extra/task buys +21pp. Worth it for most use cases.
- GPT-4o + simple scaffold: 12%. GPT-4o + Epoch's upgraded scaffold: 40%. Same model, 3.3× improvement.

**The implication:** Scaffold selection should be part of your model selection decision. Optimising scaffold on a mid-tier model may outperform a flagship model with a poor scaffold.

**Figure:** `q5_capability_price_frontier.png`

---

### Q6: Scaffold × Model Interaction and "Expensive Failures"

**Scaffold quality is not portable across models — what works brilliantly on one model can catastrophically fail on another.**

From the SWE-Effi heatmap (5 scaffolds × 3 models):
- **SWE-Agent + GPT-4o-mini**: 14% resolve rate AND token efficiency (EuTB) of only **5.1%** — catastrophic combination. SWE-Agent is designed for stronger models; weak models get stuck in loops.
- **Agentless** is the most robust: consistent improvement regardless of model tier
- **OpenHands** is the most model-sensitive: 18% (GPT-4o-mini) → 34% (Qwen3-32B)

The "Expensive Failures" pattern:
- Agentless-Mini: failures cost only 1.5× successes (efficient early termination)
- SWE-Agent: failures cost 4.3× successes (gets stuck, burns tokens)
- This matters for RL training: every failed rollout is a tax on your training budget

**Figure:** `q6_scaffold_model_interaction.png`

---

## Central Insight

> **Scaffolding is not a wrapper — it's a co-designed system.**

The optimal scaffold depends on the model family, the task distribution, and the resource budget. Evaluating "model capability" on SWE-bench without controlling for scaffold is like evaluating a driver's skill using different cars on different tracks.

The field has moved from asking "which model is best?" to needing to ask "which model + scaffold combination is best for my specific cost and accuracy constraints?"

---

## Files

```
scaffolding_research/
├── findings_summary.md            ← This file
├── scaffolding_investigation.ipynb ← Jupyter notebook (build + analysis)
├── generate_figures.py            ← Standalone script to regenerate figures
└── figures/
    ├── q1_scaffold_vs_model_capability.png
    ├── q2_scaffolding_over_time.png
    ├── q3_efficiency_vs_capability.png
    ├── q4_pareto_frontier.png
    ├── q5_capability_price_frontier.png
    ├── q6_scaffold_model_interaction.png
    └── bonus_value_frontier.png
```
