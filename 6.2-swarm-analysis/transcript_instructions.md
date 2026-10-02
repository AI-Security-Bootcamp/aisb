
# Day 6 — Section 2: Analyze an Agent Transcript

Analyze an agent trace with its reasoning, tool calls, and results.
Figure out what the agent did, and try to attribute the actions to
the agent's reasoning.

## Table of Contents

- [Setup](#setup)
- [First pass](#first-pass)
    - [Exercise 6.2.1: Manually review the data](#exercise-621-manually-review-the-data)
- [Summarization](#summarization)
    - [Exercise 6.2.2: Discover activities without an answer key](#exercise-622-discover-activities-without-an-answer-key)
- [Codebook design](#codebook-design)
    - [Exercise 6.2.3: Defining labels](#exercise-623-defining-labels)
- [Classification](#classification)
    - [Exercise 6.2.4: Label activities and highlight evidence](#exercise-624-label-activities-and-highlight-evidence)
- [Incident reconstruction](#incident-reconstruction)
    - [Exercise 6.2.5: What actually happened?](#exercise-625-what-actually-happened)

## Setup

Create a file named `day6_answers.py` in the `6.2-swarm-analysis` directory.
This is your answer file for this track. If you see a code snippet here, copy it
into your answer file and keep the `# %%` lines to make Python code cells.

If you run into missing dependencies, check if you have the `requirements.txt` and `.env` setup.
We have provided utilities to handle HTTP, caching, usage recording, batching, and retries.
Identical requests reuse `.cache/llm/` without another API charge. Delete that
folder when you intentionally want fresh responses to unchanged requests.

**Start by pasting the code below into `day6_answers.py`.**


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

source = read_jsonl(folder / "inputs/00-transcript.jsonl")
output = folder / "work/transcript"
```

## First pass

### Exercise 6.2.1: Manually review the data

> **Difficulty**: 2/5
> **Importance**: 5/5
> **Time**: 10 minutes

Skim [the transcript, page 2 onwards](https://cdn.sanity.io/files/4zrzovbb/website/8359003bfb12a2f01ce84ad3df1d3a3e2f15a8eb.pdf)
to go through what an agent's actions look like. Spend 5-10 mins trying and write down what
actions the agent took, and try to attribute the actions to what the model believed.

This trace from one agent is already over a thousand pages, and recent incidents have had
thousands of agents participating. To analyse these incidents, therefore, we need to use
LLMs to spped us up. The rest of the exercise walks through one way to analyze this data.

We'll be using a simple visualizer to analyze the data: use the supplied cell to create
an unreviewed view of the input. Open[the viewer](utils/visualizer.html) directly in a
browser and select `work/transcript/00-source.json`. Search by ID or text, and narrow the
date range to read an interval. We will be building on this to add descriptive labels to the
traces in this exercise.


```python
from transcript_test import test_source


test_source(source)
write_json(output / "00-source.json", {
    "track": "transcript",
    "labels": [{"id": "unreviewed", "name": "Unreviewed evidence",
                "definition": "No activity assigned yet.", "exclude": ""}],
    "records": [dict(r, label="unreviewed", quote="") for r in source],
})
print(f"Open utils/visualizer.html and choose {output / '00-source.json'}")
```


## Summarization

### Exercise 6.2.2: Discover activities without an answer key

Before you start, look through the methodology section in the (METR's report amalyzing agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident)[https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation].
Your goal is to implement a more basic version of the classifiers METR used in their analysis.

This exercise is broken down into three steps:
1. Extracting summaries from the data to trace what happened
2. Collating the summaries to merge similar summaries into one label
3. Running a classification pass on the labels to visualize the agent activity

Implement the prompt in `summarize(group)`. Its input is a consecutive list of
records. Extract concrete activities, stated goals, methods, failed attempts,
changes of approach, etc and correlate these with the supporting IDs.

Your input will be a list of object like this (chunked, for faster and cheaper inference, as well as context across messages):
```json
{"id": "T85", "index": 85, "time": "2026-07-18T01:03:22.233535Z", "kind": "TextMessage", "text": "<thinking>\nPort 8443 forwarding failed. Could be the port or GatewayPorts restriction. Let me try binding to loopback (default) — that at least tells me -R works at all; and try another port for [redacted-ip-8].\n</thinking>\n8443 failed — let me test whether `-R` works at all (loopback bind) and try other ports."}
```
Return this JSON shape; names and summaries are your model's findings:
```json
{"activities": [{"name": "...", "summary": "...", "source_ids": ["..."]}]}
```

<details><summary>Hint: separate discovery from compression</summary><blockquote>

Ask the model to identify distinct methods before merging routine variations.
Require uncertainty and source IDs for each activity. Avoid a prompt that only
produces a broad chronological recap: small, surprising methods can disappear.
Do not supply this incident's expected findings as examples, but you can use METR's
findings.

</blockquote></details>


```python

def summarize(group):
    """Describe activities in a chunk and cite the supporting records."""
    # TODO: Describe methods and uncertainty; require activities with source_ids.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, group, task="transcript_summarize",
                    effort="low", max_tokens=6500)
from transcript_test import test_summaries
```

You can play around with your prompt and make sure it gives you reasonable summaries with the following:


And finally, run the summarization loop on all of the data. This is a lot of llm calls, which are slow and expensive.
Parallel generations make this fast and expensive, so you should make sure you are running these in parallel!


```python
# # generate summaries from chunked data
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper function
#
# summaries = summarize_in_chunks(
#     evidence, summarize, max_records=100, max_chars=85000,
#     workers=12,
# )
# test_summaries(summaries, source)
# write_json(output / "01-summaries.json", summaries)
# print(f"Saved {len(summaries)} chunk summaries to {output / '01-summaries.json'}")
pass
```

To iterate, you can look at come of the chunks and tweak your prompt if you see any obvious improvements.

## Codebook design

### Exercise 6.2.3: Defining labels

Implement `make_codebook(summaries)`. The input contains each chunk's `source_ids`
and `summary.activities`; use these discoveries to propose (at most) 20 flat labels.
Each label should have a short ID, a name, a definition, and exclusions (if any).

The goal is to merge synonyms, and keep distinct methods apart even if they share a goal.
Your output should look like:
```json
{"labels": [{"id": "short_id", "name": "...", "definition": "...", "exclude": "..."}]}
```

We also recommed these fallback labels so that you can spot and iterate on the labelling easily:
- `other`: the activity is identifiable, but no definition covers its method.
- `insufficient_context`: the supplied evidence does not establish the activity.


```python

def make_codebook(summaries):
    """Propose flat labels from the source-derived activity summaries."""
    # TODO: Define flat behavioral labels with inclusion and exclusion boundaries.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, summaries, task="transcript_codebook",
                    effort="medium", max_tokens=12000)
from transcript_test import test_codebook
# # run the summaries through the codebook prompt
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper functions
#
# codebook = build_codebook(summaries, make_codebook)
# test_codebook(codebook)
# write_json(output / "01-codebook.json", codebook)
# for label in codebook["labels"]:
#     print(label["id"], "—", label["name"])
pass
```

## Classification

### Exercise 6.2.4: Label activities and highlight evidence

Implement `classify(batch)`. Each batch contains `codebook`, `targets`,
`target_ids`, `context_before`, and `context_after`. The function takes 20 targets
with five neighboring records on each side. Return the target IDs with one label
and at most one quote (to highlight any interesting sections) per target:

```json
{"records": [{"id": "...", "label": "label_id", "quote": "..."}]}
```


```python

def classify(batch):
    """Label targets using the codebook and surrounding records in this batch."""
    # TODO: Label every target once, use context, and select one literal quote.
    system_prompt = ""

    if not system_prompt.strip():
        raise NotImplementedError  # Write your prompt here

    return call_llm(system_prompt, batch, task="transcript_classify",
                    effort="low", max_tokens=8000)
from transcript_test import test_classifications
# # run the classifier over all the samples
# # You can (optionally) write the code for this, or just uncomment the lines below to use our helper functions
#
# records = classify_in_batches(
#     source, evidence, codebook, classify,
#     batch_size=20, context_records=5, workers=32,
# )
# result = {"track": "transcript", "labels": codebook["labels"], "records": records}
# test_classifications(result, source, codebook)
# write_json(output / "02-classifications.json", result)
# counts = Counter(r["label"] for r in records)
# for label in ("other", "insufficient_context"):
#     print(f"{label}: {counts[label]}/{len(records)} ({counts[label] / len(records):.2%})")
# print(f"Open utils/visualizer.html and choose {output / '02-classifications.json'}")
pass
```

You should now be able to see what the result of the analysis tells you!
Open [the viewer](utils/visualizer.html) and choose
`work/transcript/02-classifications.json`. Click around to explore and see
what the agent tried to do, and try to pinpoint why.

## Incident reconstruction

### Exercise 6.2.5: What actually happened?

Use the viewer to choose three coherent stories. For each, write a short story
that describes what the agent tried to do, and present a hypothesis for why.
How successful was the attempt?

Read [Anthropic's report](https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents)
to check what you might've missed, and think about why.
