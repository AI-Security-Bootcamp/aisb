# Full wiki run — METR example prompts

Completed 24 September 2026 with `deepseek/deepseek-v4.1-flash` through
OpenRouter / DeepInfra FP8. One flat codebook and one classification pass;
no participant exercise or Other-refinement pass was added.

| Artifact | Contents |
|---|---|
| [00-wiki.jsonl](00-wiki.jsonl) | All 14,456 source revision changes |
| [01-summaries.json](01-summaries.json) | 212 chunks containing 2,718 activities and supporting source IDs |
| [01-codebook.json](01-codebook.json) | 22 flat behavioral labels |
| [02-records.jsonl](02-records.jsonl) | Every unchanged source record, one label, and at most one exact quote |
| [costs.json](costs.json) | Cumulative API usage, including earlier runs and failed attempts |

Open the [updated viewer](../classification.html). Click a timeline bar for source
text or toggle **Highlighted passages only**. There are 13,355 retained quotes;
every quote is a literal substring shorter than 240 characters.

**Results:** Other 289/14,456 (2.00%); insufficient context 188/14,456 (1.30%).
The summaries recover six of ten headline baseline behaviors, but only PRNG
prediction survives as a distinct final label. Important methods are lost during
codebook consolidation or absorbed into general coordination categories.

Read the [benchmark report](../BENCHMARK.md) and
[evidence-linked results](../benchmark_results.json) for the previous-run
comparison and the separate 38-claim summary audit. These are qualitative
assistant reviews, not full-corpus classification accuracy estimates.

New spend for this rerun was **$3.274702**, including retries and repairs. The
cumulative `costs.json` total also includes the earlier $4.108295 run; do not use
that combined total as this run's cost. The previous outputs are preserved
[here](../previous_outputs/before_metr_examples/).

The [scripts README](../steps/README.md) describes inputs, outputs, and execution.
Two failed batches were repaired in smaller groups with an instruction to leave
highlights empty for repetitive filler; all 14,456 records are now complete.
