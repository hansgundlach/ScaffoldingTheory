import re
from docent import Docent
from docent.sdk.reading import ParameterContextConfig
from docent.data_models.reading import ContextFilterSection

client = Docent()
collection_id = "479b7093-5a33-47f1-8d7b-fc9f6f16bb75"
client.plan_name = "GPT-5.1 Codex failure mode clustering v6"
client.default_collection_id = collection_id

SUMMARIZE_READING_ID = "68472efc-24a0-4007-be6e-f7e25affdbef"

run_context = ParameterContextConfig(
    agent_run_metadata=ContextFilterSection(mode="include", patterns=["task", "exception", "scores.*"]),
)

# ── Step 1: Summarize each failed GPT-5.1 Codex run (CACHED) ──
failed_runs = client.query(
    collection_id,
    """
    SELECT agent_runs.id AS run,
           metadata_json->>'task' AS task_name,
           COALESCE(metadata_json->>'exception', 'none') AS exception_type,
           metadata_json->'scores'->>'reward' AS reward
    FROM agent_runs
    WHERE metadata_json->'agent'->>'model_name' = 'openai/gpt-5.1-codex'
      AND CAST(metadata_json->'scores'->>'reward' AS DOUBLE PRECISION) = 0
    ORDER BY agent_runs.id
    """,
)
client.show_query_result(failed_runs, name="All failed GPT-5.1 Codex runs (reward=0)")

summarize = client.read(
    prompt_template=[
        failed_runs.run.as_type("agent_run"),
        """
You are analyzing a failed agent run from a terminal-based coding benchmark.
The agent attempted to solve a task but scored 0 (complete failure).

Analyze the transcript and write a 2-3 sentence summary of WHY the agent failed.
Focus on the root cause — what specific mistake, strategy failure, or blocker
caused the agent to not solve the task? Be concrete and specific.

Consider patterns like:
- Did the agent get stuck in a loop (retrying the same failing approach)?
- Did it misunderstand the task requirements?
- Did it fail to install/configure dependencies?
- Did it take a fundamentally wrong approach?
- Did it run out of time because it was too slow or unfocused?
- Did it encounter an environment/tooling issue it couldn't resolve?
- Did it produce output in the wrong format?

Cite specific parts of the transcript that show the failure.
        """,
    ],
    context_config=run_context,
    model="openai/gpt-5.5",
    name="Summarize why each failed run scored zero",
)

# ── Step 2: Fetch cached summaries and propose clusters via text ──
print("Flushing step 1 (summaries - cached)...")
summarize_id = summarize.id

# Fetch all 193 summary texts
all_rows = client.execute_dql(
    collection_id,
    f"""
    SELECT rr.output AS output
    FROM reading_results rr
    JOIN reading_result_links rrl ON rrl.result_id = rr.id
    WHERE rrl.reading_id = '{summarize_id}'
    ORDER BY rr.id
    """,
)
all_dicts = client.dql_result_to_dicts(all_rows)
summaries = []
for row in all_dicts:
    out = row["output"]
    if isinstance(out, dict) and "output" in out:
        out = out["output"]
    if isinstance(out, dict) and "text" in out:
        text = out["text"]
    elif isinstance(out, str):
        text = out
    else:
        text = str(out)
    # Strip inline citations for cleaner clustering input
    text = re.sub(r'\[T\d+B\d+:.*?\]', '', text).strip()
    summaries.append(text)

print(f"Fetched {len(summaries)} summaries")

# Build numbered summary block for the cluster proposal prompt
# Sample 100 for cluster discovery (UUID ordering = random)
sample = summaries[:100]
summary_block = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(sample))

propose_clusters = client.read(
    prompts_list=[
        [
            f"""
You are reviewing failure summaries from GPT-5.1 Codex runs on a terminal-based
coding benchmark (terminal-bench). Each summary describes why an agent run scored
zero (complete failure). This is a sample of {len(sample)} out of {len(summaries)} total failures.

Here are the failure summaries:

{summary_block}

Based on these summaries, propose 6-12 failure mode categories. Each category
should be:
- **Prevalent**: appears in multiple runs (not a one-off)
- **Specific**: a developer reading the category name and description would
  understand what concrete fix or improvement would address it
- **Actionable**: points to a specific agent behavior that could be changed

For each category, provide:
- A short snake_case name
- A clear description of the failure pattern
- An estimate of how many of the summaries above fit this category
- Quote the summary numbers (e.g. [3], [17]) that are most representative

The categories should be mutually exclusive and collectively exhaustive.
If a run has multiple issues, the category should capture the PRIMARY root cause
(the one that, if fixed, would most likely have led to success).
            """,
        ]
    ],
    model="openai/gpt-5.5",
    output_schema={
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "estimated_count": {"type": "integer"},
                        "representative_summaries": {"type": "string"},
                    },
                    "required": ["name", "description", "estimated_count", "representative_summaries"],
                },
            },
        },
        "required": ["categories"],
    },
    collection_id=collection_id,
    name="Propose failure mode clusters from 100 sampled summaries",
)

# Block for cluster results
print("Waiting for cluster proposal (approve in Docent UI)...")
results = propose_clusters.results
assert len(results) > 0
clusters = results[0].output
assert clusters is not None
categories = clusters["categories"]

categories.append({
    "name": "other",
    "description": "Failure does not clearly fit any of the above categories",
    "estimated_count": 0,
    "representative_summaries": "",
})

category_names = [c["name"] for c in categories]
category_descriptions = "\n".join(
    f"  - {c['name']}: {c['description']}" for c in categories
)
print(f"Proposed {len(category_names)} clusters:")
for c in categories:
    print(f"  {c['name']} (~{c['estimated_count']}): {c['description'][:80]}...")

# ── Step 3: Classify every failed run into a failure mode ──
classify = client.read(
    prompt_template=[
        failed_runs.run.as_type("agent_run"),
        f"""
You are classifying a failed GPT-5.1 Codex run from a terminal-based coding benchmark.
The agent scored 0 (complete failure).

Classify the PRIMARY failure mode using one of these categories:
{category_descriptions}

Choose the category that captures the root cause — the issue that, if fixed,
would most likely have led to success. If multiple issues are present, pick
the one that was most decisive.

Provide:
1. Your reasoning (with citations from the transcript)
2. The failure category
3. A specific, concrete description of what went wrong in this particular run
        """,
    ],
    context_config=run_context,
    model="openai/gpt-5.5",
    output_schema={
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "citations": True},
            "failure_category": {"type": "string", "enum": category_names},
            "specific_description": {"type": "string", "citations": True},
        },
        "required": ["reasoning", "failure_category", "specific_description"],
    },
    name="Classify each failed run into a failure mode",
)

# Distribution query
dist = client.query(
    collection_id,
    f"""
    SELECT failure_category, COUNT(failure_category) AS run_count
    FROM (
        SELECT rr.output->>'failure_category' AS failure_category
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        WHERE rrl.reading_id = '{classify}'
    ) AS subq
    GROUP BY failure_category
    ORDER BY run_count DESC
    """,
)
client.show_query_result(dist, name="Failure mode distribution across all failed GPT-5.1 runs")

# Cross-tab with exception type
cross = client.query(
    collection_id,
    f"""
    SELECT failure_category, exception_type, COUNT(failure_category) AS run_count
    FROM (
        SELECT
            rr.output->>'failure_category' AS failure_category,
            COALESCE(ar.metadata_json->>'exception', 'none') AS exception_type
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        JOIN agent_runs ar ON ar.id = (rr.arguments_dict->'run'->>'id')::uuid
        WHERE rrl.reading_id = '{classify}'
    ) AS subq
    GROUP BY failure_category, exception_type
    ORDER BY failure_category, run_count DESC
    """,
)
client.show_query_result(cross, name="Failure modes x Exception type (timeout vs completed)")

print("Waiting for classification step (approve in Docent UI)...")
client.flush()
print("Analysis complete.")
