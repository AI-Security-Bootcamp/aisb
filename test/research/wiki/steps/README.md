# Wiki analysis: one pass with DeepSeek V4.1 Flash

All three model stages use `deepseek/deepseek-v4.1-flash` through OpenRouter,
pinned to DeepInfra FP8. Summarization and classification disable reasoning;
codebook generation now uses high reasoning effort. These are investigation scripts,
separate from the Anthropic transcript pipeline.

| Script | Inputs in `../simple_outputs/` | Outputs in `../simple_outputs/` |
|---|---|---|
| [01-summarize.py](01-summarize.py) | `00-wiki.jsonl` | `01-summaries.json`, `01-codebook.json` |
| [02-classify.py](02-classify.py) | `00-wiki.jsonl`, `01-codebook.json` | `02-records.jsonl` |

The input is a byte-for-byte copy of [units.jsonl](../units.jsonl): 14,456 nonempty
revision changes across seven wikis. See [collection scope](../COLLECTION_AND_METHODS.md)
for preprocessing and omitted sources. Requests include only ID, timestamp,
page, site, and text; previous classifications and publisher conclusions are
not included as evidence. Naming examples from a separate METR investigation
illustrate the desired wording; they are not evidence or expected findings.

Requests use short record IDs such as `R12` to avoid copying long wiki page and
revision names. Scripts map them back to original source IDs before writing any
analytical output. This does not change the source text, ordering, or coverage.

Step 01 reads every record, using chunks of at most 100 records or 85,000 text
characters, with 32 concurrent requests. Each summary describes activities and
lists their supporting record IDs. It then combines all activity names and
descriptions into at most 22 flat labels; source-ID lists remain in the summary
file but are unnecessary for this consolidation call.

## Naming revision and saved results

Summary activity names and final label names now use **"The agents ..."**:
an action, attempt, or discovery, with a concrete method and evidenced purpose.
Four examples from METR illustrate this level of detail without supplying the
wiki findings we want to discover. See [example sources](../METR_METHODS.md).
The codebook keeps this framing when merging activities and preserves distinct
methods that serve the same goal. Qualifications such as "try" and "report"
prevent names from implying unverified success.

This is a focused naming revision to the baseline prompts. The model, batching,
single-pass design, classifier, and output formats are unchanged. The revision
has now run on all 14,456 records; the outputs and viewer show this run. See the
[new comparison](../NAMING_BENCHMARK.md): Other 0.42%, insufficient context 0.46%,
and three strict headline matches, matching the original baseline. All 22 label
names use the requested framing. The run cost $3.102129, including retries.

The earlier METR-example experiment was reverted after it reduced distinct baseline
label recovery from three behaviors to one. The
[original results](../previous_outputs/before_metr_examples/),
[historical comparison](../BENCHMARK.md), and
[experimental artifacts](../previous_outputs/metr_examples/) remain available.

Labels are flat, with definitions and exclusions; unfamiliar methods belong in
Other. The original full-run scripts were not snapshotted, so the baseline
prompts were reconstructed from the retained pilot and codebook template before
adding this naming revision.
The earlier rollback restored the analytical outputs exactly; it did not recover
byte-identical original scripts. The current run used committed prompt revision
`750152e` without changes during execution. Cumulative costs retain all three
full runs and the subsequent codebook-only reasoning experiment.

## Classification and execution

The saved full run above predates the reasoning change. To test only codebook
generation against the same saved summaries, run:

```bash
python3 test/research/wiki/try_reasoning_codebook.py
```

This uses the same codebook function and unchanged prompt, model, temperature,
22-label limit, and 12,000-token output budget. Reasoning counts toward that
budget and is excluded from returned text; the final JSON format is unchanged.
The candidate is saved to `../reasoning_codebook/01-codebook.json`, while usage
appends to the cumulative `../simple_outputs/costs.json`. The active codebook,
classifications, and viewer are preserved until the candidate is assessed.
The [completed experiment](../reasoning_codebook/README.md) used 7,233 reasoning
tokens and cost $0.035984. It retained a new bypass category while preserving
the existing PRNG, heartbeat, and acceleration categories. Deletion awareness
is still missing. No classifications have been run with this candidate.

Step 02 classifies batches of 20 targets with five context records on each side,
using 64 concurrent requests. Each target receives one label and at most
one quote. Labels can be a codebook ID, `other`, or `insufficient_context`.
Adjacent records need not belong to the same actor or activity. Posts about
external success remain self-reports unless independently corroborated.

Every target ID must appear exactly once in a valid response. Quotes are removed
if they are not literal substrings of their target, or are 240 characters or
longer. Their IDs are retained with that request's usage in `costs.json`.
Malformed model responses get at most three attempts; this is not another
discovery or classification pass. A truncated batch, or one that still fails
response validation after its retries, is split into smaller target groups using
the same model and codebook. Failed chunks/batches stop the pipeline while
preserving successful outputs. The viewer refuses incomplete classification
output. Summary length is a guideline; extra activities are retained rather
than discarded merely to meet a count.

Run from the workspace root using Python's standard library:

```bash
python3 test/research/wiki/steps/01-summarize.py
python3 test/research/wiki/steps/02-classify.py
python3 test/research/wiki/build_viewer.py
```

Set `OPENROUTER_API_KEY` in the environment, or enter it at the hidden prompt.
No key is stored in the scripts or outputs. Every run makes fresh paid calls and
overwrites that stage's analytical outputs; there is no response cache, shared
helper module, command-line parser, or simulated model response. `costs.json`
accumulates the actual usage returned by the API, including paid retries.
Requests interrupted before the API returns usage can incur unreported charges.

Open [classification.html](../classification.html) after building it. Labels
appear as separate timeline rows; click a bar for the source text or enable
**Highlighted passages only**. Bars group consecutive observations, separated
at gaps longer than five minutes; they do not establish continuous agent work.
The viewer makes no model calls. Its source text is inert, including posted HTML.

The model-selection pilot and preliminary cost projection are documented
[here](../../openrouter/README.md).
