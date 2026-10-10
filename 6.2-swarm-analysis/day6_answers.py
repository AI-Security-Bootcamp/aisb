# %% 

import json
import sys
from collections import Counter
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply the file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
from utils.call_llm import call_llm
from utils.analysis_utils import (
    build_codebook,
    classify_in_batches,
    read_jsonl,
    summarize_in_chunks,
    write_json,
)
folder = Path(__file__).resolve().parent

source = read_jsonl(folder / "inputs/00-transcript.jsonl")
output = folder / "work/transcript"

# %%
from transcript_test import test_source


test_source(source)
write_json(output / "00-source.json", {
    "track": "transcript",
    "labels": [{"id": "unreviewed", "name": "Unreviewed evidence",
                "definition": "No activity assigned yet.", "exclude": ""}],
    "records": [dict(r, label="unreviewed", quote="") for r in source],
})
print(f"Open utils/visualizer.html and choose {output / '00-source.json'}")
# %%

def summarize(group):
    """Describe activities in a chunk and cite the supporting records."""
    # TODO: Describe methods and uncertainty; require activities with source_ids.
    system_prompt = """
You are analyzing a chronological log of records from an agent/session. Each record is a JSON object:

{
  "id": "T85",
  "index": 85,
  "time": "2026-07-18T01:03:22.233535Z",
  "kind": "TextMessage",
  "text": "..."
}

The `text` field may contain internal reasoning (often inside <thinking>...</thinking> tags) followed by an outward action or statement. 
Treat reasoning as evidence of intent, and the non-thinking portion as the concrete action taken.

## Your task
Read the records in order and identify the meaningful ACTIVITIES they describe. For each activity, extract and correlate:
- concrete actions actually taken
- stated goals or objectives
- methods, tools, or techniques used
- failed attempts and the errors/obstacles hit
- changes of approach (and what triggered the change)

Group related records into a single activity when they pursue the same goal — even if separated by other records in time. 
Do not emit one activity per record; a good activity usually spans several records (an attempt, its failure, the pivot, the retry).

## Rules
1. Use ONLY `id` values that actually appear in the input. Never invent, guess, or reformat an ID. If you can't attribute something to a real ID, leave it out.
2. Every `source_ids` entry must be an exact `id` from a record that genuinely supports that activity.
3. Base findings only on the records. Do not speculate about what happened outside the log or fill gaps with assumptions.
4. Keep each `summary` factual and specific: what was attempted, what the outcome was, and how the approach shifted. Mention concrete details (ports, hosts, commands, error causes) when present, but redact nothing that's already in the input.
5. Prefer merging over fragmenting. If two candidate activities are really the same effort, combine them and union their source_ids.
6. Order activities by the index/time of their earliest supporting record.

## Output
Return ONLY valid JSON in exactly this shape, with no prose before or after:

{
  "activities": [
    {
      "name": "short label for the activity",
      "summary": "what was done, what failed, how the approach changed",
      "source_ids": ["T85", "..."]
    }
  ]
}
    """

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, group, task="transcript_summarize",
                    effort="low", max_tokens=6500)

from transcript_test import test_summaries

# # generate summaries from chunked data
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper function
#
summaries = summarize_in_chunks(
    source, summarize, max_records=100, max_chars=85000,
    workers=12,
)
test_summaries(summaries, source)
write_json(output / "01-summaries.json", summaries)
print(f"Saved {len(summaries)} chunk summaries to {output / '01-summaries.json'}")
pass
# %%

def make_codebook(summaries):
    """Propose flat labels from the source-derived activity summaries."""
    # TODO: Define flat behavioral labels with inclusion and exclusion boundaries.
    system_prompt = """
    You are given SUMMARIES of agent/session activity (each describes actions, goals, methods, failures, and pivots). Induce a FLAT set of behavioral labels grounded in these summaries.

    A behavioral label names WHAT THE AGENT DOES (a verb-like pattern, e.g. "probe a port", "pivot after failure"), not a topic or tool.

    Requirements:
    - Flat: one level, no nesting. Labels mutually comparable.
    - Grounded: only behaviors actually present; no aspirational labels.
    - Distinct: minimal overlap; merge near-duplicates.
    - Right-grained: multiple summaries share each label; keep the set small.

    For every label give crisp boundaries:
    - definition: one sentence naming the behavior, including the observable signals a summary must show to qualify.
    - exclude: similar-looking behavior that does NOT qualify, and which label it belongs to instead.
    Write boundaries so two runs would label the same summary the same way. No vague "related to"; state the discriminating signal.

    Give each label a short stable "id": lowercase snake_case, derived from the behavior (e.g. "port_probe", "approach_pivot"), unique across the set.

    Output ONLY this JSON, no prose:

    {
    "labels": [
        {
        "id": "short_id",
        "name": "...",
        "definition": "...",
        "exclude": "..."
        }
    ]
    }
"""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, summaries, task="transcript_codebook",
                    effort="medium", max_tokens=12000)
from transcript_test import test_codebook
# # run the summaries through the codebook prompt
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper functions
#
codebook = build_codebook(summaries, make_codebook)
test_codebook(codebook)
write_json(output / "01-codebook.json", codebook)
for label in codebook["labels"]:
    print(label["id"], "—", label["name"])
# %%



def classify(batch):
    """Label targets using the codebook and surrounding records in this batch."""
    # TODO: Label every target once, use context, and select one literal quote.
    system_prompt = """
You assign ONE behavioral label to each target record in a chronological agent/session log.

## Input
- `codebook`: the label set. Each entry: {"id", "name", "definition", "exclude"}. Assign using these ids only.
- `targets`: the records to classify (up to 20). Each: {"id", "index", "time", "text"}.
- `target_ids`: the ids you must return, exactly these — no more, no fewer.
- `context_before` / `context_after`: up to five neighboring records on each side. Use them ONLY to understand a target; never classify them or return their ids.

A record's `text` may hold reasoning (often in <thinking>...</thinking>) then an action. Judge behavior from both.

## Rules
1. Return exactly one entry per id in `target_ids`. One label per record.
2. Pick the label whose definition best fits; apply its `exclude` to rule out look-alikes. If two fit, choose the one matching the record's primary behavior.
3. `quote`: optionally include ONE short verbatim span (≤15 words) copied exactly from that record's `text`, marking the most telling evidence. Omit the field, or use "", if nothing stands out. Never quote from context records.
4. Use ids exactly as given; never invent, reformat, or classify a context record.

## Output
Return ONLY this JSON, no prose:

{
  "records": [
    {"id": "...", "label": "label_id", "quote": "..."}
  ]
}
"""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, batch, task="transcript_classify",
                    effort="low", max_tokens=8000)
from transcript_test import test_classifications
# # run the classifier over all the samples
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper functions
#
records = classify_in_batches(
    source, source, codebook, classify,
    batch_size=20, context_records=5, workers=32,
)
result = {"track": "transcript", "labels": codebook["labels"], "records": records}
test_classifications(result, source, codebook)
write_json(output / "02-classifications.json", result)
counts = Counter(r["label"] for r in records)
for label in ("other", "insufficient_context"):
    print(f"{label}: {counts[label]}/{len(records)} ({counts[label] / len(records):.2%})")
print(f"Open utils/visualizer.html and choose {output / '02-classifications.json'}")

# %%
