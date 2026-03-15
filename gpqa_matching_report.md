# GPQA-Diamond ↔ HAL Benchmark Model Matching Report

This document describes how HAL benchmark model names were matched to
Epoch GPQA-Diamond model IDs for the GPQA vs Agentic performance plots.

## Matching approach

1. HAL uses human-readable names like `Claude Sonnet 4.5 (September 2025)`
2. Epoch GPQA uses machine IDs like `claude-sonnet-4-5-20250929_32K`
3. An explicit mapping table (`HAL_TO_GPQA`) maps HAL names → GPQA base model IDs
4. GPQA models have multiple thinking-budget variants (_16K, _32K, _64K, etc.)
5. For standard HAL entries, we use the **base** (no suffix) GPQA score
6. For HAL entries marked **High**, we use the **best** GPQA score across budgets
7. Some HAL models have no GPQA match (DeepSeek V3, Gemini 2.0 Flash) — these are excluded

## Known issues and limitations

- **Thinking budget mismatch**: HAL 'High' likely uses a specific thinking budget;
  we use best-available GPQA score as a proxy, which may overestimate base capability.
- **GPQA is a static QA benchmark**, not an agentic one — it measures reasoning
  capability without tool use, multi-step planning, or environment interaction.
- **HAL 'High' vs standard**: The same base model appears with different thinking
  budgets in HAL. We map both to the same GPQA base model but use different GPQA
  score variants (base vs best).
- **Missing matches**: DeepSeek V3/V3.1, Gemini 2.0 Flash have no GPQA-Diamond entry.

## Full matching table

| HAL Model Name | GPQA Base ID | GPQA Version Used | GPQA Score | Matched? |
|---|---|---|---|---|
| Claude Haiku 4.5 (October 2025) | claude-haiku-4-5-20251001 | claude-haiku-4-5-20251001 | 60.5% | Yes |
| Claude Haiku 4.5 High (October 2025) | claude-haiku-4-5-20251001 | claude-haiku-4-5-20251001_32K | 71.2% | Yes |
| Claude Opus 4 (May 2025) | claude-opus-4-20250514 | claude-opus-4-20250514 | 69.2% | Yes |
| Claude Opus 4 High (May 2025) | claude-opus-4-20250514 | claude-opus-4-20250514_16K | 76.3% | Yes |
| Claude Opus 4.1 | claude-opus-4-1-20250805 | claude-opus-4-1-20250805 | 73.2% | Yes |
| Claude Opus 4.1 (August 2025) | claude-opus-4-1-20250805 | claude-opus-4-1-20250805 | 73.2% | Yes |
| Claude Opus 4.1 High (August 2025) | claude-opus-4-1-20250805 | claude-opus-4-1-20250805_16K | 77.3% | Yes |
| Claude Opus 4.5 | claude-opus-4-5-20251101 | claude-opus-4-5-20251101 | 80.7% | Yes |
| Claude Opus 4.5 (November 2025) | claude-opus-4-5-20251101 | claude-opus-4-5-20251101 | 80.7% | Yes |
| Claude Opus 4.5 High (November 2025) | claude-opus-4-5-20251101 | claude-opus-4-5-20251101_32K | 86.0% | Yes |
| Claude Sonnet 4 (May 2025) | claude-sonnet-4-20250514 | claude-sonnet-4-20250514 | 66.7% | Yes |
| Claude Sonnet 4 High (May 2025) | claude-sonnet-4-20250514 | claude-sonnet-4-20250514_32K | 78.3% | Yes |
| Claude Sonnet 4.5 (September 2025) | claude-sonnet-4-5-20250929 | claude-sonnet-4-5-20250929 | 73.7% | Yes |
| Claude Sonnet 4.5 High (September 2025) | claude-sonnet-4-5-20250929 | claude-sonnet-4-5-20250929_59K | 82.3% | Yes |
| Claude-3.7 Sonnet (February 2025) | claude-3-7-sonnet-20250219 | claude-3-7-sonnet-20250219 | 66.0% | Yes |
| Claude-3.7 Sonnet High (February 2025) | claude-3-7-sonnet-20250219 | claude-3-7-sonnet-20250219_64K | 78.5% | Yes |
| DeepSeek R1 (January 2025) | DeepSeek-R1 | DeepSeek-R1 | 69.2% | Yes |
| DeepSeek R1 (May 2025) | deepseek-reasoner | deepseek-reasoner | 83.4% | Yes |
| DeepSeek V3 (March 2025) | (no GPQA entry) | — | — | No |
| DeepSeek V3.1 (August 2025) | (no GPQA entry) | — | — | No |
| GPT-4.1 (April 2025) | gpt-4.1-2025-04-14 | gpt-4.1-2025-04-14 | 66.9% | Yes |
| GPT-5 Medium (August 2025) | gpt-5-2025-08-07 | gpt-5-2025-08-07_medium | 85.4% | Yes |
| GPT-OSS-120B (August 2025) | openai/gpt-oss-120b | openai/gpt-oss-120b | — | No |
| GPT-OSS-120B High (August 2025) | openai/gpt-oss-120b | openai/gpt-oss-120b | — | No |
| Gemini 2.0 Flash (February 2025) | (no GPQA entry) | — | — | No |
| Gemini 2.0 Flash High (February 2025) | (no GPQA entry) | — | — | No |
| Gemini 2.5 Pro Preview (March 2025) | gemini-2.5-pro-exp-03-25 | gemini-2.5-pro-exp-03-25 | 83.8% | Yes |
| Gemini 3 Pro Preview High (November 2025) | gemini-3-pro-preview | gemini-3-pro-preview | 92.6% | Yes |
| o3 Medium (April 2025) | o3-2025-04-16 | o3-2025-04-16_high | 81.8% | Yes |
| o4-mini High (April 2025) | o4-mini-2025-04-16 | o4-mini-2025-04-16_high | 79.6% | Yes |
| o4-mini Low (April 2025) | o4-mini-2025-04-16 | o4-mini-2025-04-16_high | 79.6% | Yes |

## Per-benchmark matching statistics


### SWE-bench Mini Verified

- 27 matched points out of 31 total HAL entries
- 2 scaffolds: HAL Generalist Agent, SWE-Agent
- Unmatched HAL models: DeepSeek V3 (March 2025), Gemini 2.0 Flash (February 2025)

### GAIA

- 28 matched points out of 32 total HAL entries
- 2 scaffolds: HAL Generalist Agent, HF Open Deep Research
- Unmatched HAL models: DeepSeek V3 (March 2025), Gemini 2.0 Flash (February 2025)

### CORE-bench Hard

- 40 matched points out of 49 total HAL entries
- 3 scaffolds: CORE-Agent, Claude Code, HAL Generalist Agent
- Unmatched HAL models: DeepSeek V3 (March 2025), DeepSeek V3.1 (August 2025), GPT-OSS-120B (August 2025), GPT-OSS-120B High (August 2025), Gemini 2.0 Flash (February 2025)

### TAU-bench Airline

- 22 matched points out of 26 total HAL entries
- 2 scaffolds: HAL Generalist Agent, TAU-bench Tool Calling
- Unmatched HAL models: DeepSeek V3 (March 2025), Gemini 2.0 Flash (February 2025), Gemini 2.0 Flash High (February 2025)

### USACO

- 11 matched points out of 13 total HAL entries
- 2 scaffolds: HAL Generalist Agent, USACO Episodic + Semantic
- Unmatched HAL models: DeepSeek V3 (March 2025), Gemini 2.0 Flash (February 2025)

### SciAgentBench

- 20 matched points out of 23 total HAL entries
- 2 scaffolds: HAL Generalist Agent, SAB Self-Debug
- Unmatched HAL models: DeepSeek V3 (March 2025), Gemini 2.0 Flash (February 2025)