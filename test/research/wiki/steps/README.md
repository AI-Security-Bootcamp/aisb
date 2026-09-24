# Wiki analysis: one pass with DeepSeek V4.1 Flash

All three model stages use `deepseek/deepseek-v4.1-flash` through OpenRouter,
pinned to DeepInfra FP8 with reasoning disabled. These are investigation scripts,
separate from the Anthropic transcript pipeline.

| Script | Inputs in `../simple_outputs/` | Outputs in `../simple_outputs/` |
|---|---|---|
| [01-summarize.py](01-summarize.py) | `00-wiki.jsonl` | `01-summaries.json`, `01-codebook.json` |
| [02-classify.py](02-classify.py) | `00-wiki.jsonl`, `01-codebook.json` | `02-records.jsonl` |

The input is a byte-for-byte copy of [units.jsonl](../units.jsonl): 14,456 nonempty
revision changes across seven wikis. See [collection scope](../COLLECTION_AND_METHODS.md)
for preprocessing and omitted sources. Requests include only ID, timestamp,
page, site, and text; previous classifications and publisher conclusions are
not included as evidence. Prompts contain no incident-specific naming examples.

Requests use short record IDs such as `R12` to avoid copying long wiki page and
revision names. Scripts map them back to original source IDs before writing any
analytical output. This does not change the source text, ordering, or coverage.

Step 01 reads every record, using chunks of at most 100 records or 85,000 text
characters, with 32 concurrent requests. Each summary describes activities and
lists their supporting record IDs. It then combines all activity names and
descriptions into at most 22 flat labels; source-ID lists remain in the summary
file but are unnecessary for this consolidation call.

## Restored baseline

The METR-example experiment was reverted after it reduced distinct baseline
label recovery from three behaviors to one. The active summaries, codebook, and
classifications are byte-identical to the
[previous results](../previous_outputs/before_metr_examples/). The viewer uses
these restored results. The [comparison](../BENCHMARK.md) and
[experimental artifacts](../previous_outputs/metr_examples/) remain available.

Prompts again ask for descriptive activities without example labels or a forced
sentence style. Labels are flat, with definitions and exclusions; unfamiliar
methods belong in Other. The original full-run scripts were not snapshotted, so
the restored prompts use the retained pre-experiment pilot and codebook template.
This is an exact restoration of the analytical outputs, not a claim that the
scripts are byte-identical to the originals. No model calls were made for the
rollback. Cumulative cost records retain both runs.

## Classification and execution

Step 02 classifies batches of 20 targets with five context records on each side,
using 64 concurrent requests. Each target receives one label and at most
one quote. Labels can be a codebook ID, `other`, or `insufficient_context`.
Adjacent records need not belong to the same actor or activity. Posts about
external success remain self-reports unless independently corroborated.

Every target ID must appear exactly once in a valid response. Quotes are removed
if they are not literal substrings of their target, or are 240 characters or
longer. Their IDs are retained with that request's usage in `costs.json`.
Malformed model responses get at most three attempts; this is not another
discovery or classification pass. A truncated classification batch is split into
smaller target groups using the same model and codebook. Failed chunks/batches stop the pipeline while
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
