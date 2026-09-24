# Wiki analysis — restored baseline

The previous DeepSeek V4.1 Flash results are restored after reverting the
METR-example experiment. All three analytical outputs match the
[saved baseline](../previous_outputs/before_metr_examples/) byte for byte.

| Artifact | Contents |
|---|---|
| [00-wiki.jsonl](00-wiki.jsonl) | All 14,456 source revision changes across seven wikis |
| [01-summaries.json](01-summaries.json) | 212 chunks containing 2,873 activities and supporting source IDs |
| [01-codebook.json](01-codebook.json) | 22 flat labels |
| [02-records.jsonl](02-records.jsonl) | Every unchanged source record, one label, and at most one exact quote |
| [costs.json](costs.json) | Cumulative API usage, including both runs and failed attempts |

Open the [restored viewer](../classification.html). Click a timeline bar for source
text or toggle **Highlighted passages only**. There are 13,501 retained quotes;
every quote is a literal substring shorter than 240 characters.

**Results:** Other 86/14,456 (0.59%); insufficient context 84/14,456 (0.58%).
Three of ten headline baseline behaviors have distinct final labels, versus one
in the reverted experiment. These are qualitative finding-recovery judgments,
not full-corpus classification accuracy estimates.

Low Other does not establish adequate discovery: the summaries describe the
Power BI proxy bypass, but the codebook lacks a dedicated label. Public posts
also do not independently verify external success or author identity. See
[collection scope](../COLLECTION_AND_METHODS.md) for omitted evidence.

The restored run cost **$4.108295**, including failed attempts. The reverted
experiment cost **$3.274702**, bringing this directory's cumulative recorded cost
to **$7.382997**. The rollback made no API calls and did not erase charges.

The [benchmark report](../BENCHMARK.md) describes the archived experiment and
its comparison with this baseline. Its machine-readable results retain that
historical meaning. [Experimental outputs](../previous_outputs/metr_examples/)
remain available. The [scripts README](../steps/README.md) explains rerunning
and the limits of reconstructing the original prompts.
