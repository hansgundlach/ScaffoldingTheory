---
docent_collection_id: 479b7093-5a33-47f1-8d7b-fc9f6f16bb75
docent_source_reading_plan_id: 643b2dc9-440c-4cb1-b3e3-98168dbebc08
title: "GPT-5.1 Codex Failure Mode Analysis on Terminal-Bench"
---

# GPT-5.1 Codex Failure Mode Analysis on Terminal-Bench

This report identifies and classifies the primary reasons GPT-5.1 Codex fails on terminal-bench, a terminal-based coding benchmark. GPT-5.1 Codex scored zero on 193 out of 267 runs (72% failure rate) — significantly worse than GPT-5 Codex, which failed 131 of 251 runs (52%). The analysis summarized each failed transcript, proposed failure mode clusters from a sample, then classified all 193 failures into those categories.

## Overall performance comparison

GPT-5.1 Codex shows a large regression vs GPT-5 Codex across all key metrics. The most striking difference is the timeout rate: nearly half of GPT-5.1 runs hit the time limit.

::dql-table{title="Model performance comparison" query="SELECT model_name, COUNT(model_name) AS total_runs, ROUND(CAST(AVG(reward) AS NUMERIC), 3) AS avg_reward, SUM(CASE WHEN reward = 0 THEN 1 ELSE 0 END) AS zero_reward_runs, SUM(CASE WHEN exception = 'AgentTimeoutError' THEN 1 ELSE 0 END) AS timeout_runs FROM (SELECT COALESCE(metadata_json->'agent'->>'model_name', 'unknown') AS model_name, CAST(metadata_json->'scores'->>'reward' AS DOUBLE PRECISION) AS reward, COALESCE(metadata_json->>'exception', 'none') AS exception FROM agent_runs) AS subq GROUP BY model_name ORDER BY avg_reward DESC"}
::

## Failure mode distribution

All 193 zero-reward GPT-5.1 Codex runs were classified into six failure modes by analyzing each full transcript. The dominant failure mode is **exploratory stall** — the agent spends its entire budget reading files and planning but never produces an artifact.

::dql-table{title="Failure mode distribution (all 193 failed GPT-5.1 runs)" query="SELECT failure_category, COUNT(failure_category) AS run_count, ROUND(CAST(COUNT(failure_category) * 100.0 / 193 AS NUMERIC), 1) AS pct_of_failures FROM (SELECT rr.output->>'failure_category' AS failure_category FROM reading_results rr JOIN reading_result_links rrl ON rrl.result_id = rr.id WHERE rrl.reading_id = '71220aa0-c2c8-4429-809b-3af97ec176f5' AND rr.output IS NOT NULL) AS subq GROUP BY failure_category ORDER BY run_count DESC"}
::

## 1. Exploratory stall without artifact (37% of failures)

The single largest failure mode: the agent spent the entire run inspecting files, grepping source code, reading documentation, and diagnosing — but never created the required deliverable. All 71 runs in this category timed out (AgentTimeoutError), confirming the agent was stuck in an open-ended reconnaissance loop.

::dql-table{title="Exploratory stall: all timeouts" query="SELECT failure_category, exception_type, COUNT(failure_category) AS run_count FROM (SELECT rr.output->>'failure_category' AS failure_category, COALESCE(ar.metadata_json->>'exception', 'none') AS exception_type FROM reading_results rr JOIN reading_result_links rrl ON rrl.result_id = rr.id JOIN agent_runs ar ON CAST(ar.id AS VARCHAR) = rr.arguments_dict->'run'->>'id' WHERE rrl.reading_id = '71220aa0-c2c8-4429-809b-3af97ec176f5' AND rr.output->>'failure_category' = 'exploratory_stall_no_artifact') AS subq GROUP BY failure_category, exception_type ORDER BY run_count DESC"}
::

This is a representative example of the exploratory stall pattern:

::reading-result{id="02315789-7732-4a20-b3f0-23707bce211a" title="Exploratory stall example"}
The agent reads code, greps for patterns, and plans — but runs out of time before producing any output.
::

::callout{color="red" title="Root cause"}
GPT-5.1 Codex lacks a "produce first, refine later" strategy. It over-invests in understanding before acting. A concrete fix: force the agent to produce a minimal artifact within the first 20% of the time budget, then iterate.
::

## 2. Flawed or non-general solution (22% of failures)

The second largest category: the agent produced code or output, but it was fundamentally wrong — overfit to visible examples, used brittle heuristics, violated the required interface, or ignored key semantics. Unlike the exploratory stall, nearly all of these runs completed within the time limit (42 of 43 had no timeout), indicating the agent was confident in a wrong answer.

::dql-table{title="Flawed solution: rarely timeouts" query="SELECT failure_category, exception_type, COUNT(failure_category) AS run_count FROM (SELECT rr.output->>'failure_category' AS failure_category, COALESCE(ar.metadata_json->>'exception', 'none') AS exception_type FROM reading_results rr JOIN reading_result_links rrl ON rrl.result_id = rr.id JOIN agent_runs ar ON CAST(ar.id AS VARCHAR) = rr.arguments_dict->'run'->>'id' WHERE rrl.reading_id = '71220aa0-c2c8-4429-809b-3af97ec176f5' AND rr.output->>'failure_category' = 'flawed_or_non_general_solution') AS subq GROUP BY failure_category, exception_type ORDER BY run_count DESC"}
::

::reading-result{id="08449d35-a6cf-41ae-9f62-6cc6657c1a95" title="Flawed solution example"}
The agent produces code that handles visible test cases but fails on hidden ones due to an overly narrow approach.
::

::callout{color="orange" title="Root cause"}
The agent takes shortcuts that satisfy superficial checks but miss deeper requirements. A fix would be to have the agent reason from the full specification and test edge cases before declaring completion.
::

## 3. Partial workflow not finalized (15% of failures)

The agent began a plausible solution path but failed to complete the end-to-end workflow: it didn't rerun after discovering a fix, left validation failures unresolved, or abandoned mid-task. These runs overwhelmingly timed out (25 of 28), suggesting the agent got bogged down mid-workflow.

::reading-result{id="07812cfc-be2d-4da7-b4e3-bc2dfcf7f0e1" title="Partial workflow example"}
The agent starts well but fails to finalize — leaving a working approach incomplete.
::

::callout{color="orange" title="Root cause"}
The agent lacks end-to-end completion checks. After producing intermediate work, it should verify against the task's stated success condition before stopping or context-switching.
::

## 4. Execution control or protocol failure (12% of failures)

The agent lost control of the terminal or failed to follow the benchmark interaction protocol: runaway foreground processes, interactive editor automation mistakes, or failure to submit the required completion signal. This is split roughly evenly between timeouts (10) and non-timeouts (14).

::reading-result{id="0261b94d-da35-4091-9f84-fdbe4b453d1a" title="Execution control failure example"}
The agent loses control of a foreground process or fails to follow the interaction protocol.
::

::callout{color="orange" title="Root cause"}
Poor process management and protocol awareness. Fixes: run long processes in the background, handle interactive prompts programmatically, and always check protocol requirements before declaring completion.
::

## 5. Unrecovered tooling or external blocker (6% of failures)

The agent encountered environmental blockers — missing compilers, unavailable packages, network failures — and failed to find a workaround. All 11 runs timed out.

::reading-result{id="168a3abb-016c-48a0-be7b-c8f96f994d19" title="Tooling blocker example"}
The agent gets stuck on a missing dependency and cannot find an alternative.
::

## 6. Deployment configuration incomplete (3% of failures)

Five runs (all on `install-windows-3.11` or `qemu-alpine-ssh`) where the agent installed and launched a service but missed a critical configuration detail that the grader checks.

## Timeout vs. non-timeout failures

The failure modes split cleanly by whether the agent timed out, revealing two fundamentally different failure populations:

::dql-table{title="Failure modes by exception type" query="SELECT failure_category, exception_type, COUNT(failure_category) AS run_count FROM (SELECT rr.output->>'failure_category' AS failure_category, COALESCE(ar.metadata_json->>'exception', 'none') AS exception_type FROM reading_results rr JOIN reading_result_links rrl ON rrl.result_id = rr.id JOIN agent_runs ar ON CAST(ar.id AS VARCHAR) = rr.arguments_dict->'run'->>'id' WHERE rrl.reading_id = '71220aa0-c2c8-4429-809b-3af97ec176f5' AND rr.output IS NOT NULL) AS subq GROUP BY failure_category, exception_type ORDER BY run_count DESC"}
::

- **Timeout failures** (128 runs): dominated by exploratory stall (71), partial workflow (25), and tooling blockers (11). The agent runs out of time because it's stuck — either in an analysis loop, an incomplete workflow, or fighting the environment.
- **Non-timeout failures** (65 runs): dominated by flawed solutions (42) and protocol failures (14). The agent finishes but produces wrong output or fails to follow the rules.

## Recommendations

::callout{color="green" title="1. Force early artifact production"}
The single highest-impact intervention. 71 of 193 failures (37%) are pure exploratory stalls. Require the agent to produce a minimal deliverable within the first 20% of its time budget, then iterate. This directly addresses the largest failure mode.
::

::callout{color="green" title="2. Add self-validation before completion"}
43 runs (22%) produced wrong answers confidently. Have the agent test its output against the task specification and edge cases before declaring success. This is especially important because these failures don't timeout — the agent thinks it's done.
::

::callout{color="green" title="3. Implement end-to-end completion checks"}
28 runs (15%) started well but didn't finish. After each significant step, the agent should verify whether the task's success condition is met. If not, it should continue rather than context-switching to investigation.
::

::callout{color="green" title="4. Improve process management and protocol awareness"}
24 runs (12%) failed due to terminal control issues or not following the benchmark protocol. The agent needs better handling of foreground processes, interactive editors, and benchmark-specific submission requirements.
::

## Full classification results

Every individual classification can be inspected below, with citations to the specific transcript evidence:

::reading-results-table{reading_id="71220aa0-c2c8-4429-809b-3af97ec176f5" title="All 193 failure classifications with reasoning"}
Each result includes the failure category, reasoning with transcript citations, and a specific description of what went wrong.
::
