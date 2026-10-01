# 6.2 — Swarm analysis

Two independent incident-analysis exercises use different evidence: an agent's
expressed reasoning and tool trace, and public communications attributed to
multiple agents. In each, participants write three analysis prompts, audit the
results, and reconstruct three stories with a supplied timeline viewer.

| Track | Participant instructions | Suggested time | Answer file |
| --- | --- | --- | --- |
| Transcript | [Exercises 6.2.1–6.2.5](transcript_instructions.md) | 120 minutes | `day6_answers.py` |
| Wiki communications | [Exercises 6.2.6–6.2.10](wiki_instructions.md) | 120 minutes | `wiki_answers.py` |

The tracks can be completed separately and do not depend on each other's results.
Times are authoring estimates pending a participant pilot; API wait time varies.
Each track allocates 10 minutes to triage, 25 to summarization, 20 to codebook
design, 30 to classification, and 35 to reconstruction and comparison.

## Curriculum contract


| Area | Prerequisites coming in | Intended learning outcomes           |
| --- | --- |--------------------------------------|
| Engineering | - |                                      |
| ML | - |                                      |
| Security | - | Incident Response for Agent Activity |
| Theory | - | -                                    |

## Background

- (Chase's Agent Incident Response Playbook)[https://docs.google.com/document/d/1qjdoAezkKrh4rZQlfZlpmzMFknomJhLCTR3UJ-Pg2BQ/edit?usp=sharing]
- (METR's Brief independent investigation of agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident)[https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation] 

## Setup

From the repository root:

```bash
python3 -m pip install -r 6.2-swarm-analysis/requirements.txt
cp 6.2-swarm-analysis/.env.example 6.2-swarm-analysis/.env
```

Fill in `OPENROUTER_API_KEY`. Both tracks use the same `OPENROUTER_MODEL`, currently
configured as `deepseek/deepseek-v4.1-flash` through DeepInfra FP8. Defaults load
from this folder's `.env.example`, then environment variables, the repository-root
`.env`, and finally this folder's `.env`; later values take precedence, including
blanks. If the repository `.env` already contains your key, skip the copy step.

Follow the generated instructions in your answer file. The three prompt bodies
are the participant implementation; the run cells are supplied and initially
commented out. Complete a prompt, then uncomment its run cell. Each stage checks
and saves its output before you continue. Empty scaffolds make no paid calls.
The separate source-view cell works without an API key.

The model helper records tokens and provider-reported cost in
`outputs/<track>/usage.jsonl`, including replies rejected by validation. These
scripts do not enforce a dollar cap; set your service-side limit for the lab.
Check usage before rerunning a stage. If a completed JSON is already available,
load it with `json.loads(path.read_text())` to continue without repeating its calls.
Identical requests automatically reuse parsed answers from `.cache/llm/`, even
across script restarts. The cache key includes the endpoint, model, prompt, data,
reasoning effort, token limit, and provider settings. API keys are not stored.
Cache hits make no API call and add no usage row. Failed calls and responses
rejected by the supplied validators are not retained, so retries can request a
fresh answer. Delete `.cache/llm/` to intentionally rerun unchanged requests.
The cache is ignored by Git. It resumes individual requests, not whole stages;
each stage still rebuilds its output from those responses. Concurrent identical
requests share one call within a process.

## Files and outputs

```text
inputs/
  00-transcript.jsonl
  00-wiki.jsonl
transcript_solution.py         # author source: prompts, tasks, tests, reference
wiki_solution.py
transcript_instructions.md     # generated participant views
wiki_instructions.md
transcript_test.py             # generated artifact checks
wiki_test.py
utils/
  call_llm.py                  # credentials, cached API calls, and usage recording
  analysis_utils.py            # batching, retries, ID/quote validation
  visualizer.html
.env.example
requirements.txt
work/                         # your outputs (ignored by Git)
  transcript/                 # wiki/ has the same four files
    00-source.json            # unreviewed source, ready for the viewer
    01-summaries.json
    01-codebook.json
    02-classifications.json   # labels + preserved source + quotes, for the viewer
outputs/                      # committed results from earlier investigations
  transcript/
  wiki/
```

All records are summarized once in consecutive chunks, then classified once as
targets. Context overlaps intentionally. There is one codebook and one flat label
per record; no automatic refinement pass. Helpers preserve input order, retry
malformed replies, and restore shortened request IDs. Quotes that are not literal
or are too long are cleared. Wiki batches can split after three malformed replies.
A single source record longer than the chunk character target stays intact.

| Stage | Transcript | Wiki |
| --- | --- | --- |
| Summaries | Low reasoning, 12 workers | Reasoning off, 32 workers |
| Codebook | Medium reasoning | High reasoning |
| Classification | Low reasoning, 32 workers | Reasoning off, 64 workers |

A run cell replaces that stage's file in `work/<track>/`; a failed stage may leave
an older file in place. Check that the artifacts belong to the run you are auditing.
Tests receive the in-memory result before it is saved. Network or authentication
failures stop the run.

## Use the viewer

Open [utils/visualizer.html](utils/visualizer.html) directly in a browser. Choose
`work/<track>/00-source.json` during triage and summary/codebook review; search for
the IDs in your summaries. Choose `work/<track>/02-classifications.json` after
classification to inspect activities and highlighted evidence. Summaries and
standalone codebooks are not viewer inputs.

Click an activity name to filter it, or a bar to select an interval. Search uses
full source text, IDs, pages, actors, and activity names. Narrow the UTC dates for
dense intervals; select **Undated records** for records without a usable timestamp.
**Highlighted passages only** shows the quote; expand the record to see its context.
The viewer runs locally and makes no model calls.

Bars group consecutive same-label observations at gaps of up to five minutes.
They do not establish continuous execution, concurrent work, verified success,
or causal transmission between posts. Timestamps without a timezone are plotted
as UTC; the original timestamp remains visible. Source HTML is displayed as text.

## Tests and evidence checks

Each run cell invokes generated tests on its actual artifacts. Tests check input
IDs, full chunk/target coverage, citation scope, flat label definitions, source
preservation, and literal highlights. They make no model calls and do not require
particular category names. Because prompt outputs are nondeterministic, these
checks inspect the produced artifacts instead of calling a prompt function again.

Each exercise also has an evidence acceptance check: initial observations,
summary fidelity, label boundaries, a ten-record classification audit, and a
three-story incident report. These assess meanings and outcome claims that the
structural assertions cannot establish. No fake model responses are used.

To check the **committed investigation artifacts**, without making model calls:

```bash
python3 -m pytest -q 6.2-swarm-analysis/transcript_test.py 6.2-swarm-analysis/wiki_test.py
```

The no-argument pytest tests read `outputs/`, not your `work/` results. The run-cell
test calls explicitly receive your own results. You can call those same functions
on JSONs loaded from `work/` to recheck them without rerunning the models.

## After your investigation

Freeze your prompts, three stories, and ten-record audit before opening these:

- [Wiki investigation comparison](../test/research/wiki/FULL_REASONING_BENCHMARK.md)
  and [the publishers' account](https://collusion.wiki/).
- [Transcript investigation comparison](../test/research/transcript/simple_outputs/README.md)
  and [the published transcript release](https://github.com/anthropics/mythos-5-incident-transcript/tree/62858fcf2725fe7b38872d538e973f38846ea744).

Compare claims using source IDs, not label spelling. Mark each reviewed claim as
recovered, partial, missing, or unassessable; identify whether its evidence was
absent from the input or lost in summarization, codebook generation, classification,
or your own review. Report both counts and denominators. These comparison notes
include earlier model judgments, not independently verified ground truth.

The committed outputs are historical: transcript used Terra → Sol → Terra, while
wiki used DeepSeek V4.1 Flash. They are not a new run of the shared-model exercise.
Recorded costs were $7.74 and $2.48 respectively; these are historical costs, not
estimates or guarantees for participants' prompts. The new exercise still needs
a live participant pilot to check cost, timing, and finding recovery.

## Inputs and provenance


Both inputs are exact copies of the investigation's normalized `00-*.jsonl`
files, prepared by [prepare.py](../test/research/prepare.py). `source_file` fields
retain their historical paths for provenance; running these analyses does not
read those paths. All necessary analysis inputs are in this folder.

| Input | Contents | SHA-256 |
| --- | --- | --- |
| `00-transcript.jsonl` | 2,061 assistant records from the published Mythos 5 transcript; reasoning, tool calls, and tool results flattened to text. Images and redactions are not reconstructed. | `553e4e806680455970ed9f2a272400bced56955b957cfec661675aad69b4de6e` |
| `00-wiki.jsonl` | 14,456 nonempty inserted/replaced text records across seven wikis, in timestamp/ID order. Includes publisher-derived diffs and timestamp provenance. | `fbae22855d75a25a224470fb3d06897f3d055a58b686193f67d3a207d814ff63` |

The transcript source was downloaded from the
[pinned Anthropic release](https://github.com/anthropics/mythos-5-incident-transcript/tree/62858fcf2725fe7b38872d538e973f38846ea744).
The publisher's [release notice](../test/additional-public-data/anthropic-readme.md)
documents redactions and asks that this benchmark not be included in training
corpora. Retain that restriction when redistributing the data.

Wiki records come from the [collusion.wiki download](https://collusion.wiki/explorer/download),
including the additional wiki selection. They do not cover every indexed site,
complete request logs, or all deletion events. Actor fields are publisher metadata,
not verified identities; the model receives only ID, time, site, page, and text.
See [collection notes](../test/research/wiki/COLLECTION_AND_METHODS.md) for the
selection process, recovery checks, and missing evidence.

## Authoring

Edit only `*_solution.py` for exercise content and tests. Regenerate the participant
views and tests after changes:

```bash
./build-instructions.sh 6.2-swarm-analysis/transcript_solution.py 6.2-swarm-analysis/wiki_solution.py
```

Running a solution file executes the reference analysis with paid calls and writes
to `work/<track>/`. To use a separate executable instructor reference, build with
`--reference`. Keep the three analysis stages visible; supplied mechanics belong
in `utils/`. The existing `6.2-adversarial-language/` folder still needs a separate
curriculum-numbering decision.
