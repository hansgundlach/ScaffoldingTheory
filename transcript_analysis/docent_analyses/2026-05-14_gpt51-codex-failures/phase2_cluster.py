"""Phase 2: Propose failure mode clusters using template readings with auto-approval."""
from docent import Docent

client = Docent()
collection_id = "479b7093-5a33-47f1-8d7b-fc9f6f16bb75"
client.plan_name = "GPT-5.1 failure cluster proposal v3"
client.default_collection_id = collection_id

SUMMARIZE_READING_ID = "68472efc-24a0-4007-be6e-f7e25affdbef"


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


# Use string_agg in DQL to concatenate 50 summaries into one text field
concat_summaries = client.query(
    collection_id,
    f"""
    SELECT string_agg(summary_text, '
---
') AS summaries_text
    FROM (
        SELECT rr.output->'output'->>'text' AS summary_text
        FROM reading_results rr
        JOIN reading_result_links rrl ON rrl.result_id = rr.id
        WHERE rrl.reading_id = '{SUMMARIZE_READING_ID}'
        ORDER BY rr.id
        LIMIT 50
    ) AS sub
    """,
)
client.show_query_result(concat_summaries, name="50 concatenated failure summaries for clustering")

propose_clusters = client.read(
    prompt_template=[
        """You are reviewing failure summaries from GPT-5.1 Codex runs on terminal-bench
(a terminal-based coding benchmark). Each summary describes why an agent run scored
zero (complete failure). This is a sample of 50 out of 193 total failures.
Each summary is separated by --- below.

""",
        concat_summaries.summaries_text.as_type("text"),
        """

Based on these summaries, propose 6-12 failure mode categories. Each category should be:
- Prevalent: appears in multiple runs (not a one-off)
- Specific: a developer reading the category name and description would
  understand what concrete fix or improvement would address it
- Actionable: points to a specific agent behavior that could be changed

For each category, provide:
- A short snake_case name
- A clear description of the failure pattern
- An estimate of how many of the 50 summaries above fit this category

The categories should be mutually exclusive and collectively exhaustive.
If a run has multiple issues, the category should capture the PRIMARY root cause
(the one that, if fixed, would most likely have led to success).""",
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
                    },
                    "required": ["name", "description", "estimated_count"],
                },
            },
        },
        "required": ["categories"],
    },
    name="Propose failure mode clusters from 50 sampled summaries",
)

# Flush and auto-approve
print("Flushing and auto-approving...")
flush_result = client.flush(open_in_browser=False)
plan_id = flush_result["plan_id"]
print(f"Plan ID: {plan_id}")
auto_approve_plan(client, collection_id, plan_id)

# Now wait for results
print("Waiting for cluster proposal to complete...")
results = propose_clusters.results
assert len(results) > 0
clusters = results[0].output
assert clusters is not None
categories = clusters["categories"]

print(f"\n=== Proposed {len(categories)} failure mode clusters ===")
for c in categories:
    print(f"\n  {c['name']} (~{c['estimated_count']} runs)")
    print(f"    {c['description']}")

import json
with open("docent_analyses/2026-05-14_gpt51-codex-failures/clusters.json", "w") as f:
    json.dump(categories, f, indent=2)
print("\nClusters saved to clusters.json")
