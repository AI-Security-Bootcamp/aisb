
# 6.2: Analyze Agent Collusion

Analyze agent activity on a public wiki to reconstruct what agents did and why.
We will analyze the data published by collusion.wiki to understand how agents
worked together, what they did, and reconstruct why.

## Table of Contents

- [Setup](#setup)
- [First pass](#first-pass)
    - [Exercise 6.2.6: Manually review the data](#exercise-626-manually-review-the-data)
- [Summarization](#summarization)
    - [Exercise 6.2.7: Discover activities without an answer key](#exercise-627-discover-activities-without-an-answer-key)
- [Codebook design](#codebook-design)
    - [Exercise 6.2.8: Defining labels](#exercise-628-defining-labels)
- [Classification](#classification)
    - [Exercise 6.2.9: Label activities and highlight evidence](#exercise-629-label-activities-and-highlight-evidence)
- [Incident reconstruction](#incident-reconstruction)
    - [Exercise 6.2.10: What actually happened?](#exercise-6210-what-actually-happened)
- [Next steps](#next-steps)

## Setup

Create a file named `wiki_answers.py` in the `6.2-swarm-analysis` directory.
This is your answer file for this track. If you see a code snippet here, copy it
into your answer file and keep the `# %%` lines to make Python code cells.

If you run into missing dependencies, check `requirements.txt` and the
[`.env` example](.env.example). We have provided utilities to handle HTTP, caching,
usage recording, batching, and retries.
Identical requests reuse `.cache/llm/` without another API charge. Delete that
folder when you intentionally want fresh responses to unchanged requests.
Summarization shows a progress bar and the reported cost for this run, including
retries. Cache hits cost nothing; replies without cost data are marked unreported.

**Start by pasting the code below into `wiki_answers.py`.**


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

## First pass

### Exercise 6.2.6: Manually review the data

Spend 5 minutes reading a few pages from (the dataset)[https://collusion.wiki/explorer/]. What were the
agents were doing and what are your inital hypotheses for how they used the wiki to work together? We only have
public messages, so you'll need to work out what happened from what they posted.

There are over 14,000 edits across several wikis in this dataset. Analyzing them
manually would take ages, so we will use LLMs to help us find activities across
all of them, then return to the posts to understand the interesting parts.

Use the supplied cell to create an unreviewed view of the input. Open
[the viewer](utils/visualizer.html) and select `work/wiki/00-source.json`.
Search by page name or text, and narrow the date range to read an interval.
We will build on this by adding descriptive labels to the posts.


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

## Summarization

### Exercise 6.2.7: Discover activities without an answer key

Before you start, read how
[METR classified message-board workstreams](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#identifying-important-messages-and-classifying-them-by-workstream).
We'll build a simplified version of the analysis pipeline here.

The analysis has three steps:

1. Extract summaries from the data to trace what happened.
2. Merge similar activities into a set of labels.
3. Classify the records with those labels to visualize the agents' activity.

Implement the prompt in `summarize(group)`. Its input is a consecutive list of
records. Extract activities, stated goals, methods, failed attempts, and changes
of approach, with the IDs that support them. Describe what the posts show and
what the agents claim to have done. Feel free to copy and tweak your prompt from
the previous exercise if you've finished it already.

Your input will be a list of edits like this. We group records so the model has
context from surrounding messages:

```json
{
  "id": "W:dse~Sector61DataTableRequest314159@1",
  "time": "2026-06-16T20:03:13Z",
  "page": "dse/Sector61DataTableRequest314159",
  "site": "dse",
  "text": "Request from AgentPrepUserX: Does anyone with the exact all-state sector 61-62 table please share either (a) the full table, (b) a downloadable mirror, or (c) the method used to extract the one-line DataUSA JSON despite robots? I have CT/MI/WV but want later states. Reply here. Thanks."
}
```

Return this JSON shape; names and summaries are your model's findings:

```json
{"activities": [{"name": "...", "summary": "...", "source_ids": ["..."]}]}
```

<details><summary>Hint: separate discovery from compression</summary><blockquote>

Ask the model to identify distinct methods before merging routine variations.
Require uncertainty and source IDs for each activity. A broad recap can lose
small, surprising methods. You can use findings from METR's separate incident
as naming examples.

</blockquote></details>


```python

def summarize(group):
    """Describe activities in a chunk and cite the supporting records."""
    # TODO: Describe methods and uncertainty; require activities with source_ids.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, group, task="wiki_summarize",
                    effort="none", max_tokens=6500)
from wiki_test import test_summaries
```

Try your prompt on a small chunk first and read the summaries it produces:




And finally, run the summarization loop on all of the data. This is a lot of llm calls, which are slow and expensive.
Parallel generations make this fast and expensive, so you should make sure you are running these in parallel!


```python
# Generate summaries from chunks. Use the helper below, or write your own loop.
# Uncomment the lines below after writing your prompt.
#
# summaries = summarize_in_chunks(
#     evidence, summarize, max_records=100, max_chars=85000,
#     workers=32, compact_ids=True,
# )
# test_summaries(summaries, source)
# write_json(output / "01-summaries.json", summaries)
# print(f"Saved {len(summaries)} chunk summaries to {output / '01-summaries.json'}")
pass
```

Read some of the chunks alongside their summaries and tweak your prompt if
there are important details missing. You can search for the cited IDs in the viewer.

## Codebook design

### Exercise 6.2.8: Defining labels

Implement `make_codebook(summaries)`. The input contains each chunk's `source_ids`
and `summary.activities`; use these to propose at most 22 flat labels.
Each label should have a short ID, a name, a definition, and exclusions.

Merge synonyms, and keep distinct methods apart even if they share a goal.
Each label should be a finding starting with something like "The agents ...".
Your output should look like:

```json
{"labels": [{"id": "short_id", "name": "...", "definition": "...", "exclude": "..."}]}
```

The classifier also has two fallback labels to help you spot gaps and iterate:
- `other`: the activity is identifiable, but no definition covers its method.
- `insufficient_context`: the supplied evidence does not establish the activity.


```python

def make_codebook(summaries):
    """Propose flat labels from the source-derived activity summaries."""
    # TODO: Define flat behavioral labels with inclusion and exclusion boundaries.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    # Keep descriptions; the codebook does not need every supporting record ID.
    descriptions = [[{"name": a["name"], "summary": a["summary"]}
                     for a in chunk["summary"]["activities"]] for chunk in summaries]
    return call_llm(system_prompt, descriptions, task="wiki_codebook",
                    effort="high", max_tokens=12000)
from wiki_test import test_codebook
# Run the summaries through your codebook prompt. Use the helper below, or write your own loop.
# Uncomment the lines below after writing your prompt.
#
# codebook = build_codebook(summaries, make_codebook)
# test_codebook(codebook)
# write_json(output / "01-codebook.json", codebook)
# for label in codebook["labels"]:
#     print(label["id"], "—", label["name"])
pass
```

## Classification

### Exercise 6.2.9: Label activities and highlight evidence

Implement `classify(batch)`. Each batch contains `codebook`, `targets`,
`target_ids`, `context_before`, and `context_after`. The function takes 20 targets
with five neighboring records on each side. Return each target ID once, with one
label and at most one quote to highlight an interesting section:

```json
{"records": [{"id": "...", "label": "label_id", "quote": "..."}]}
```

Use a codebook label or one of the fallbacks. Copy the quote from the target's
text, keeping it under 240 characters; use `""` if nothing needs highlighting.


```python

def classify(batch):
    """Label targets using the codebook and surrounding records in this batch."""
    # TODO: Label every target once, use context, and select one literal quote.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, batch, task="wiki_classify",
                    effort="none", max_tokens=8000)
from wiki_test import test_classifications
# Classify batches with your prompt. Use the helper below, or write your own loop.
# Uncomment the lines below after writing your prompt.
#
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

Open [the viewer](utils/visualizer.html) and choose
`work/wiki/02-classifications.json`. Click around to explore what the agents
were doing and how they worked together. Look at the highlighted passages
to find interesting messages, then expand them to read the full posts.

## Incident reconstruction

### Exercise 6.2.10: What actually happened?

Use the viewer to choose three coherent stories. For each, describe what the
agents did, how they coordinated, and your hypothesis for why. How successful
were they? What in the posts supports that conclusion?

Read the "What were the agents doing?" section of [collusion.wiki](https://collusion.wiki/)
to check what you might have missed, and think about how you might have caught it.
For a missed finding, look back through the posts, summaries, and labels to see where it was lost.

## Next steps

There have been many such incidents! You can analyze data from other incidents/websites:
- https://transluce.org/us-canada-gov
- https://collusion.wiki/explorer/sites/
- https://rubyhack.ai/
- new incidents will probably be updated here https://agent-incidents.jowimo.com/
