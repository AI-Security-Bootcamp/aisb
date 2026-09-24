# Wiki analysis — action-based naming run

Completed 24 September 2026 using DeepSeek V4.1 Flash through OpenRouter /
DeepInfra FP8. This run uses the focused
[naming revision](../steps/README.md#naming-revision-and-saved-results) from
commit `750152e`: four separate-incident examples and "The agents ..." names.
There is one flat codebook and one classification pass, with reasoning disabled.

A later [codebook-only reasoning experiment](../reasoning_codebook/README.md)
is saved separately. These analytical outputs and the viewer still use the
non-reasoning codebook; the candidate has not been classified.

| Artifact | Contents |
|---|---|
| [00-wiki.jsonl](00-wiki.jsonl) | All 14,456 source revision changes across seven wikis |
| [01-summaries.json](01-summaries.json) | 212 chunks containing 2,937 activities and supporting source IDs |
| [01-codebook.json](01-codebook.json) | 22 flat labels, all named "The agents ..." |
| [02-records.jsonl](02-records.jsonl) | Every unchanged source record, one label, and at most one exact quote |
| [costs.json](costs.json) | Cumulative API usage, including all three runs and failed attempts |

Open the [viewer](../classification.html). Click a timeline bar for source
text or toggle **Highlighted passages only**. There are 13,270 retained quotes;
every quote is a literal substring shorter than 240 characters.

**Results:** Other 61/14,456 (0.42%); insufficient context 67/14,456 (0.46%).
Three of ten headline baseline behaviors have distinct final labels, matching
the original baseline and improving on one in the reverted experiment. The three
script-injection probes now share a category, but that case remains a partial
match. These are qualitative finding-recovery judgments, not classification accuracy.

Low Other does not establish adequate discovery: the summaries describe the
Power BI proxy bypass, but the codebook lacks a dedicated label. Public posts
also do not independently verify external success or author identity. See
[collection scope](../COLLECTION_AND_METHODS.md) for omitted evidence.

This run cost **$3.102129** across 949 returned responses, including retries and
automatic batch splitting. Earlier runs cost $4.108295 and $3.274702, bringing
the full-run subtotal to **$10.485126**. The later codebook reasoning call cost
$0.035984, bringing cumulative usage to **$10.521110**. No prompt changes
were needed during execution. Source fields and ordering are preserved exactly.

Read the [new benchmark report](../NAMING_BENCHMARK.md) and
[evidence-linked results](../naming_benchmark_results.json) for the comparison,
costs, and proposed codebook improvements. The [original baseline](../previous_outputs/before_metr_examples/)
and [reverted experiment](../previous_outputs/metr_examples/) remain available.
The [scripts README](../steps/README.md) explains rerunning.
