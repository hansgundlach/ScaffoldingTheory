"""Phase 3: Classify all 193 failed GPT-5.1 Codex runs into failure mode clusters."""
import json
from docent import Docent
from docent.sdk.reading import ParameterContextConfig
from docent.data_models.reading import ContextFilterSection

client = Docent()
collection_id = "479b7093-5a33-47f1-8d7b-fc9f6f16bb75"
client.plan_name = "GPT-5.1 failure classification"
client.default_collection_id = collection_id


def auto_approve_plan(client, collection_id, plan_id):
    """Approve all needs_approval steps in a reading plan."""
    state = client._get_reading_plan_state(collection_id, plan_id)
    aliases_to_approve = []
    for step in state.get("steps", []):
        if isinstance(step, dict) and step.get("derived_status") == "needs_approval":
            aliases_to_approve.append(step["alias"])
    if aliases_to_approve:
        url = f"{client._api_url}/reading/{collection_id}/reading-plan/{plan_id}/approve"
        resp = client._session.post(url, json={"step_aliases": aliases_to_approve})
        resp.raise_for_status()
        result = resp.json()
        print(f"Auto-approved steps {aliases_to_approve}, job_id={result.get('job_id')}")
        return result
    return None


# Load clusters from Phase 2
with open("docent_analyses/2026-05-14_gpt51-codex-failures/clusters.json") as f:
    categories = json.load(f)

categories.append({
    "name": "other",
    "description": "Failure does not clearly fit any of the above categories",
    "estimated_count": 0,
})

category_names = [c["name"] for c in categories]
category_descriptions = "\n".join(
    f"  - {c['name']}: {c['description']}" for c in categories
)

run_context = ParameterContextConfig(
    agent_run_metadata=ContextFilterSection(mode="include", patterns=["task", "exception", "scores.*"]),
)

# All failed GPT-5.1 runs
failed_runs = client.query(
    collection_id,
    """
    SELECT agent_runs.id AS run,
           metadata_json->>'task' AS task_name,
           COALESCE(metadata_json->>'exception', 'none') AS exception_type
    FROM agent_runs
    WHERE metadata_json->'agent'->>'model_name' = 'openai/gpt-5.1-codex'
      AND CAST(metadata_json->'scores'->>'reward' AS DOUBLE PRECISION) = 0
    ORDER BY agent_runs.id
    """,
)
client.show_query_result(failed_runs, name="All 193 failed GPT-5.1 Codex runs")

# Classify each run
classify = client.read(
    prompt_template=[
        failed_runs.run.as_type("agent_run"),
        f"""You are classifying a failed GPT-5.1 Codex run from terminal-bench (a terminal-based coding benchmark).
The agent scored 0 (complete failure).

Classify the PRIMARY failure mode using one of these categories:
{category_descriptions}

Choose the category that captures the root cause — the issue that, if fixed,
would most likely have led to success. If multiple issues are present, pick
the one that was most decisive.

Provide:
1. Your reasoning (with citations from the transcript)
2. The failure category
3. A specific, concrete description of what went wrong in this particular run""",
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
client.show_query_result(dist, name="Failure mode distribution across all 193 failed runs")

# Cross-tab with exception type (timeout vs completed)
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

# Flush and auto-approve
print("Flushing and auto-approving classification step...")
flush_result = client.flush(open_in_browser=False)
plan_id = flush_result["plan_id"]
print(f"Plan ID: {plan_id}")
auto_approve_plan(client, collection_id, plan_id)

# Wait for results (193 LLM calls — will take a few minutes)
print("Waiting for classification of 193 runs to complete...")
classify_results = classify.results
print(f"Classification complete: {len(classify_results)} results")

# Print distribution
print("\n=== FAILURE MODE DISTRIBUTION ===")
dist_rows = client.execute_dql(
    collection_id,
    f"""
    SELECT failure_category, COUNT(failure_category) AS run_count
    FROM (
        SELECT rr.output->>'failure_category' AS failure_category
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        WHERE rrl.reading_id = '{classify.id}'
    ) AS subq
    GROUP BY failure_category
    ORDER BY run_count DESC
    """,
)
for row in client.dql_result_to_dicts(dist_rows):
    print(f"  {row['failure_category']}: {row['run_count']}")

# Cross-tab
print("\n=== FAILURE MODES x EXCEPTION TYPE ===")
cross_rows = client.execute_dql(
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
        WHERE rrl.reading_id = '{classify.id}'
    ) AS subq
    GROUP BY failure_category, exception_type
    ORDER BY failure_category, run_count DESC
    """,
)
for row in client.dql_result_to_dicts(cross_rows):
    print(f"  {row['failure_category']} | {row['exception_type']}: {row['run_count']}")

print("\nDone! View full results at:")
print(f"https://docent.transluce.org/dashboard/{collection_id}/reading-plan/{plan_id}")
