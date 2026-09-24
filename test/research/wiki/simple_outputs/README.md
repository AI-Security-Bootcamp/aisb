# Wiki analysis — full run with codebook reasoning

Completed 24 September 2026 using DeepSeek V4.1 Flash through OpenRouter /
DeepInfra FP8. Both numbered steps were rerun from the source, with high reasoning
only for codebook generation. The "The agents ..." prompts remain unchanged.
There is one flat codebook and one classification pass, plus technical retries.
The [previous naming run](../previous_outputs/before_full_reasoning/) and the
separate [codebook-only experiment](../reasoning_codebook/README.md) are preserved.

| Artifact | Contents |
|---|---|
| [00-wiki.jsonl](00-wiki.jsonl) | All 14,456 source revision changes across seven wikis |
| [01-summaries.json](01-summaries.json) | 212 chunks containing 2,967 activities and supporting source IDs |
| [01-codebook.json](01-codebook.json) | 22 flat labels, all named "The agents ..." |
| [02-records.jsonl](02-records.jsonl) | Every unchanged source record, one label, and at most one exact quote |
| [costs.json](costs.json) | Cumulative API usage, including all four full runs, the codebook-only trial, and failed attempts |

Open the [viewer](../classification.html). Click a timeline bar for source
text or toggle **Highlighted passages only**. There are 13,462 retained quotes;
every quote is a literal substring shorter than 240 characters.

**Results:** Other 28/14,456 (0.19%); insufficient context 46/14,456 (0.32%).
Four of ten headline findings have distinct final labels, up from three in the
previous naming run. All reviewed Power BI bypass, PRNG, heartbeat, and clock
acceleration anchors receive their respective labels. These are qualitative
finding-recovery judgments, not classification accuracy.

Low Other does not establish adequate discovery: the summaries describe responses
to message deletion, but the codebook still loses that distinction. Public posts
also do not independently verify external success or author identity. See
[collection scope](../COLLECTION_AND_METHODS.md) for omitted evidence.

This run cost **$2.480310** across 951 returned responses, including retries and
repair. Input caching discounted the bill; the same tokens without caching would
be approximately $3.42 at the checked provider rates. Cumulative wiki usage is
**$13.001420**. Source fields and ordering are preserved exactly.

The codebook succeeded after two attempts used its full output budget. One
20-record classification batch was repaired without rerunning successful records;
the classifier now also splits after repeated response-validation failures.

Read the [full-run benchmark](../FULL_REASONING_BENCHMARK.md) and
[evidence-linked results](../full_reasoning_benchmark_results.json) for the comparison,
costs, and proposed codebook improvements. The [original baseline](../previous_outputs/before_metr_examples/)
and [reverted experiment](../previous_outputs/metr_examples/) remain available.
The [scripts README](../steps/README.md) explains rerunning.
