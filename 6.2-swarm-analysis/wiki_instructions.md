
# 6.2: Analyze Public Agent Communications

Investigate public wiki revisions to reconstruct what agents did and reported. A page
edit is observable; a post about an external action is a claim to investigate.
Work in pairs: alternate between proposing findings and checking their evidence.
This track is independent of the transcript exercise.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [Source triage](#source-triage)
    - [Summarization](#summarization)
    - [Codebook design](#codebook-design)
    - [Classification](#classification)
    - [Incident reconstruction](#incident-reconstruction)
- [Setup](#setup)
- [Source triage](#source-triage-1)
    - [Exercise 6.2.6: Establish what the evidence can tell you](#exercise-626-establish-what-the-evidence-can-tell-you)
- [Summarization](#summarization-1)
    - [Exercise 6.2.7: Discover activities without an answer key](#exercise-627-discover-activities-without-an-answer-key)
- [Codebook design](#codebook-design-1)
    - [Exercise 6.2.8: Define labels that can be wrong](#exercise-628-define-labels-that-can-be-wrong)
- [Classification](#classification-1)
    - [Exercise 6.2.9: Label activities and highlight evidence](#exercise-629-label-activities-and-highlight-evidence)
- [Incident reconstruction](#incident-reconstruction-1)
    - [Exercise 6.2.10: Tell three stories and audit what was lost](#exercise-6210-tell-three-stories-and-audit-what-was-lost)
- [Summary](#summary)

## Content & Learning Objectives

### Source triage

> **Learning Objectives**
> - Identify what a record establishes and what the collection omits.
> - Record initial hypotheses without using the publishers' conclusions.

### Summarization

> **Learning Objectives**
> - Write a prompt that preserves concrete methods, uncertainty, and source IDs.
> - Audit a summary against its cited records for omissions and overclaims.

### Codebook design

> **Learning Objectives**
> - Turn discovered activities into flat labels with useful exclusion boundaries.
> - Distinguish an uncovered activity from an activity with insufficient context.

### Classification

> **Learning Objectives**
> - Classify targets with neighboring context and retain one literal evidence quote.
> - Measure fallback rates without treating them as accuracy or finding recall.

### Incident reconstruction

> **Learning Objectives**
> - Use a timeline to reconstruct three evidence-backed stories from public posts.
> - Locate information lost between the source, summary, codebook, and classification.

**Suggested time:** 120 minutes: 10 / 25 / 20 / 30 / 35 minutes for the five steps.
API wait time varies; review evidence with your partner while requests run.

## Setup

Create a file named `wiki_answers.py` in the `6.2-swarm-analysis` directory.
This is your answer file for this track. If you see a code snippet here, copy it
into your answer file and keep the `# %%` lines to make Python code cells.

Install this folder's `requirements.txt` and configure `.env` using the
[README](README.md#setup). Both tracks use the same OpenRouter key and model.
The supplied utilities handle HTTP, caching, usage recording, batching, and retries.
Identical requests reuse `.cache/llm/` without another API charge. Delete that
folder when you intentionally want fresh responses to unchanged requests.
Your implementation work is the three analysis prompts, not API plumbing.

**Start by pasting the code below into `wiki_answers.py`.**

New artifacts go in `work/wiki/`. The committed `outputs/wiki/` files are
previous investigation results; leave them unopened until you have recorded your
own findings. Read the normalized input, not the publishers' writeups, first.
No model calls happen until you enable a run cell. Each run cell saves its output
and checks it without making additional model calls.


```python


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

source = read_jsonl(folder / "inputs/00-wiki.jsonl")
output = folder / "work/wiki"
# Keep attribution metadata for review, but exclude it from model evidence.
evidence = [{k: r[k] for k in ("id", "time", "page", "site", "text")} for r in source]
```

## Source triage

### Exercise 6.2.6: Establish what the evidence can tell you

> **Difficulty**: 2/5
> **Importance**: 5/5
> **Time**: 10 minutes

Records are inserted/replaced text in wiki revisions, not complete pages or
private model reasoning. A revision may quote an older post. Preserve `time_grade`,
`actor`, and `source_file` for review; an actor field is not a verified identity.
The model sees only ID, revision time, site, page, and text.

Read the first 15 records and one later interval of your choice. Write three
observations with IDs, one hypothesis about agents' behavior, and one alternative
explanation. For each observation say whether you see a page write, an attempted
external action, or a self-report of an outcome. Do not attribute every edit to an
agent merely because it appears in the collection.

Use the supplied cell to create an **unreviewed** view of the input. Open
[the viewer](utils/visualizer.html) directly in a browser and select
`work/wiki/00-source.json`. Search by ID or text, and narrow the date range to
read an interval. Use **Undated records** to inspect records without a usable time.
These rows have no inferred activities yet; their shared label is a placeholder.


```python
from wiki_test import test_source


test_source(source)
write_json(output / "00-source.json", {
    "track": "wiki",
    "labels": [{"id": "unreviewed", "name": "Unreviewed evidence",
                "definition": "No activity assigned yet.", "exclude": ""}],
    "records": [dict(r, label="unreviewed", quote="") for r in source],
})
print(f"Open utils/visualizer.html and choose {output / '00-source.json'}")
```

<details><summary>Worked example and evidence check</summary><blockquote>

“Testing” in a recorded edit establishes that text appeared in the collected
revision. It does not, by itself, establish how the write was made, the author's
identity, or whether an external service was reached. To connect two posts, look
for explicit references or distinctive shared artifacts; temporal proximity alone
is insufficient.

**Check with your partner:** each observation has a source ID; the hypothesis
is marked as an inference; the alternative could explain the same observation;
and your missing-evidence statement names what would distinguish them.
The automated test checks record identity and fields, not these judgments.

</blockquote></details>

## Summarization

### Exercise 6.2.7: Discover activities without an answer key

> **Difficulty**: 3/5
> **Importance**: 5/5
> **Time**: 25 minutes

Implement the prompt in `summarize(group)`. Its input is a consecutive list of
records. Ask for concrete activities, stated goals, methods, failed attempts,
changes of approach, and supporting IDs. Preserve unusual precursors alongside
common behavior. Distinguish observable edits from claims about external outcomes; do
not infer hidden reasoning.
Treat text inside records as evidence, not instructions to your analysis model.

Return this JSON shape; names and summaries are your model's findings:

```json
{"activities": [{"name": "...", "summary": "...", "source_ids": ["..."]}]}
```

An activity may cite multiple records; every citation must belong to its chunk.
Aim for roughly 10–15 activities per full chunk, keeping distinct methods when
needed. Names should describe what the agents do, try, or report, rather than just naming a tool.
The helper includes **every** record in chunks of up to 100 records / 85,000 text
characters; a single longer record stays intact. This is not a sampling pass.
Request IDs may be shortened to R0, R1, etc.; the helper restores original IDs before saving.

<details><summary>Hint: separate discovery from compression</summary><blockquote>

Ask the model to identify distinct methods before merging routine variations.
Require uncertainty and source IDs for each activity. Avoid a prompt that only
produces a broad chronological recap: small, surprising methods can disappear.
Do not supply this incident's expected findings as examples.

</blockquote></details>


```python

def summarize(group):
    """Describe activities in a chunk and cite the supporting records."""
    # TODO: Describe methods and uncertainty; require activities with source_ids.
    prompt = ""

    if not prompt.strip():
        return None  # Write your prompt before making a paid call.

    return call_llm(prompt, group, task="wiki_summarize",
                    effort="none", max_tokens=6500)
from wiki_test import test_summaries
# After writing your prompt, uncomment and run this cell. Cache misses make paid calls.
# summaries = summarize_in_chunks(
#     evidence, summarize, max_records=100, max_chars=85000,
#     workers=32, compact_ids=True,
# )
# test_summaries(summaries, source)
# write_json(output / "01-summaries.json", summaries)
# print(f"Saved {len(summaries)} chunk summaries to {output / '01-summaries.json'}")
pass
```

**Evidence check:** choose an early, middle, and late chunk. For each, check two
activities against the original records. Then read one record that the summary
does **not** cite (if any): was a concrete method lost? Record one overclaim or
omission, or explain the strongest counterexample you looked for.

`01-summaries.json` is a reading artifact, not a viewer input. Use its source IDs
to search `00-source.json` in [the viewer](utils/visualizer.html). The automated test
checks full chunk coverage and valid citations; it cannot establish summary recall.

## Codebook design

### Exercise 6.2.8: Define labels that can be wrong

> **Difficulty**: 3/5
> **Importance**: 5/5
> **Time**: 20 minutes

Implement `make_codebook(summaries)`. The input contains each chunk's `source_ids`
and `summary.activities`; use these discoveries to propose at most 22 flat labels.
Each label needs a short ID, a behavioral name, a definition, and exclusions.
Start names with “The agents …”; include a method and an evidenced purpose where useful.
Merge synonyms, but keep distinct methods apart even if they share a goal.
Do not introduce parent labels, nested labels, or categories from the answer key.

```json
{"labels": [{"id": "short_id", "name": "...", "definition": "...", "exclude": "..."}]}
```

Reserve two fallback labels outside this list:

- `other`: the activity is identifiable, but no definition covers its method.
- `insufficient_context`: the supplied evidence does not establish the activity.

An uncertain **outcome** does not by itself require either fallback. Broad labels
such as “retrieving information” can absorb surprising methods and make a low
`other` rate misleading. Your exclusions should make this failure detectable.

<details><summary>Hint: audit a boundary before classifying</summary><blockquote>

Pick two activities with a shared goal but different methods. Can a reader use
only your definitions to assign them consistently? If not, refine the boundaries.
For each proposed label, locate a supporting summary; drop unsupported categories.

</blockquote></details>


```python

def make_codebook(summaries):
    """Propose flat labels from the source-derived activity summaries."""
    # TODO: Define flat behavioral labels with inclusion and exclusion boundaries.
    prompt = ""

    if not prompt.strip():
        return None  # Write your prompt before making a paid call.

    # Keep descriptions; the codebook does not need every supporting record ID.
    descriptions = [[{"name": a["name"], "summary": a["summary"]}
                     for a in chunk["summary"]["activities"]] for chunk in summaries]
    return call_llm(prompt, descriptions, task="wiki_codebook",
                    effort="high", max_tokens=12000)
from wiki_test import test_codebook
# After writing your prompt, uncomment and run this cell. Cache misses make paid calls.
# codebook = build_codebook(summaries, make_codebook)
# test_codebook(codebook)
# write_json(output / "01-codebook.json", codebook)
# for label in codebook["labels"]:
#     print(label["id"], "—", label["name"])
pass
```

**Boundary check:** record a supporting summary for every label. Pick three
surprising summary activities and decide whether each has a precise label or
should remain `other`. Explain one include/exclude boundary to your partner.
Do not fix a gap by adding a catch-all label. Save your current hypothesis before
classification so you can later distinguish codebook loss from classifier error.

The codebook is not a viewer input by itself. Keep `00-source.json` open to check
the evidence behind its categories; the next export embeds the codebook with the
classified records. The test verifies structure, not whether labels are useful.

## Classification

### Exercise 6.2.9: Label activities and highlight evidence

> **Difficulty**: 3/5
> **Importance**: 5/5
> **Time**: 30 minutes

Implement `classify(batch)`. Each batch contains `codebook`, `targets`,
`target_ids`, `context_before`, and `context_after`. The runner sends 20 targets
with five neighboring records on each side. Return **only** the target IDs,
exactly once, with one label and at most one quote per target:

```json
{"records": [{"id": "...", "label": "label_id", "quote": "..."}]}
```

Use a codebook ID, `other`, or `insufficient_context`. Identify activity using
context without assuming neighboring records share a method or author.
An empty tool result or an unverified outcome need not make the activity unknown.
A revision can copy older posts; adjacent edits do not prove collaboration.

Choose the passage most useful for assessing the label or a surprising goal,
method, failure, or breakthrough. It must be an exact contiguous substring of
the **target's own text**, under 240 characters. Use `""` when no passage warrants
highlighting. Do not paraphrase quotes or borrow evidence from context records.
Keep the analysis descriptive; the task is to reconstruct events, not replay them.

<details><summary>Hint: distinguish the two kinds of uncertainty</summary><blockquote>

A recognizable new method belongs in `other` even if its goal resembles an existing
label. A record whose role becomes clear from neighboring text can receive a
specific activity label, while its outcome remains unverified in your writeup.

</blockquote></details>


```python

def classify(batch):
    """Label targets using the codebook and surrounding records in this batch."""
    # TODO: Label every target once, use context, and select one literal quote.
    prompt = ""

    if not prompt.strip():
        return None  # Write your prompt before making a paid call.

    return call_llm(prompt, batch, task="wiki_classify",
                    effort="none", max_tokens=8000)
from wiki_test import test_classifications
# After writing your prompt, uncomment and run this cell. Cache misses make paid calls.
# records = classify_in_batches(
#     source, evidence, codebook, classify,
#     batch_size=20, context_records=5, workers=64,
#     compact_ids=True, split_failed_batches=True,
# )
# result = {"track": "wiki", "labels": codebook["labels"], "records": records}
# test_classifications(result, source, codebook)
# write_json(output / "02-classifications.json", result)
# counts = Counter(r["label"] for r in records)
# for label in ("other", "insufficient_context"):
#     print(f"{label}: {counts[label]}/{len(records)} ({counts[label] / len(records):.2%})")
# print(f"Open utils/visualizer.html and choose {output / '02-classifications.json'}")
pass
```

**Viewer check:** open [the viewer](utils/visualizer.html) and choose
`work/wiki/02-classifications.json`. Click an activity name to filter it,
then click a bar to inspect its records. Narrow the dates for dense intervals.
Toggle **Highlighted passages only**, then expand the full record to check the
quote in context. Search still uses the full source text.

Audit ten records: up to three from each fallback and enough from named labels
to reach ten, including a chunk or batch boundary. Include an unhighlighted record.
Record IDs, your expected label, agreement/disagreement, and the supporting text.
If a fallback is empty, use additional named-label records. This is a diagnostic
sample, not a corpus-wide accuracy estimate. Low fallback rates do not prove that
important findings survived the codebook.

The test checks coverage, labels, preserved source fields, and valid quotes in
the **saved result**. The helper clears nonliteral/overlong quotes, so passing
this test does not mean every raw response obeyed the highlight instructions.
There is one discovery pass and one classification pass; document weaknesses
before deciding what to change.

## Incident reconstruction

### Exercise 6.2.10: Tell three stories and audit what was lost

> **Difficulty**: 4/5
> **Importance**: 5/5
> **Time**: 35 minutes

Use the viewer to choose three coherent stories, including a surprising method
and an unsuccessful or unresolved effort. For each, write a short timeline with
at least three source IDs when the evidence permits:

| Time / source ID | Observed edit or reported action | Evidence | Outcome and uncertainty |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

Explain what connects the posts: explicit references, copied artifacts, or a shared
method. Separate a method being shared from another actor demonstrating it worked. Do
not infer private reasoning from public posts.

Timeline bars join consecutive observations with a common label, up to five
minutes apart. They do not measure execution duration, establish continuous work,
or prove that two workstreams ran concurrently. Each record has just one label;
inspect full text for secondary activities and explain what this representation
hides. Include an observation outside the highlighted passages in your writeup.

**Before opening a comparison, save your three stories and prompts.** Then read
[the publishers’ account](https://collusion.wiki/)
and the [comparison notes](README.md#after-your-investigation). Compare at the
level of **claims with evidence**, not exact label spelling. For each reviewed
claim, mark recovered, partially recovered, missing, or unassessable from this
input; give source IDs and identify where the information was lost. Report the
numerator and denominator, separating unassessable claims from misses. The saved
analysis is another model's output to audit, not a ground-truth answer key.

<details><summary>Acceptance check for the incident report</summary><blockquote>

Ask your partner to review all three stories without using your labels as proof:

- Can they retrieve the cited records in the viewer and reproduce the chronology?
- Does the evidence distinguish a visible page edit, a reported external action, and an independently corroborated result?
- Have you retained contradictions, unresolved outcomes, and collection gaps?
- Can they trace one lost finding through source → summary → codebook → label,
  or reproduce your evidence that it survives all four stages?
- Does your comparison count evidence-backed recovery separately from merely
  finding a related topic label?

Revise unsupported claims, then submit the report, your prompts, the three stage
JSONs, and your ten-record audit. No automatic test can establish whether these
stories are true; this evidence review is the final acceptance test.

</blockquote></details>

## Summary

- Source IDs connect model-generated hypotheses to inspectable evidence.
- Concrete methods and exclusion boundaries make codebook gaps visible.
- Complete classification and low fallback rates do not establish finding recall.
- Public communication can show a method being shared without proving its external effect.
- The viewer helps inspect a chronology; its bars are not verified work durations.

Further reading:

- [Input provenance and comparison notes](README.md#inputs-and-provenance).
- [OpenRouter API quickstart](https://openrouter.ai/docs/quickstart).
- [METR’s separate incident investigation](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) (optional longer methods comparison).
