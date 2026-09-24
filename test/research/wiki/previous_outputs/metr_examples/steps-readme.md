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
not included as evidence. Naming examples come from a separate incident studied
by METR, not from the wiki findings we want to discover.

Requests use short record IDs such as `R12` to avoid copying long wiki page and
revision names. Scripts map them back to original source IDs before writing any
analytical output. This does not change the source text, ordering, or coverage.

Step 01 reads every record, using chunks of at most 100 records or 85,000 text
characters, with 32 concurrent requests. Each summary describes activities and
lists their supporting record IDs. It then combines all activity names and
descriptions into at most 22 flat labels; source-ID lists remain in the summary
file but are unnecessary for this consolidation call.

## Behavioral labels

The prompts ask for sentences of the form **"The agents [action or method]
[purpose, when supported]."** The summarizer includes five examples from METR's
OpenAI/Hugging Face investigation. They illustrate how to name a coherent activity;
they are not a fixed taxonomy or assertions about any input chunk. The model must
discover stories from the records and supply source references for each activity.
The wiki's target labels and hints about those specific cases have been removed
from both summarization and codebook prompts.

The codebook preserves this style, groups routine variants of the same activity,
and retains rare consequential methods. Classification prioritizes an explicitly
evidenced method over surrounding routine task discussion. Definitions and
exclusions still determine membership; unfamiliar methods belong in Other.
Success, attribution, and purpose must remain qualified when the records only
contain claims, plans, or suspicions.

The [methodology analysis](../METR_METHODS.md) gives sources for the examples and
explains which ideas fit our data. We retain the flat, single-label interface.
Summaries now explicitly distinguish authors' beliefs, local milestones, and
end-to-end outcomes. Removing target examples avoids directly supplying the answer
in the prompt; it does not establish that the model has never encountered the
published incident or that our data selection is unbiased.

The revised prompts have now been run on all 14,456 records. See the
[benchmark comparison](../BENCHMARK.md): the summaries recover six headline
behaviors, but only one survives as a distinct final label. The viewer shows this
new run; [previous outputs](../previous_outputs/before_metr_examples/) are retained.
Input/output formats and models are unchanged. Execution repairs added codebook
validation feedback and an empty-highlight instruction for repetitive filler.

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
