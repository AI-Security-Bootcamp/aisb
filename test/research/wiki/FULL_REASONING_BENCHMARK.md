# Full wiki pipeline with codebook reasoning

Completed 24 September 2026. Both numbered steps were rerun from all 14,456
source records, then the standalone viewer was rebuilt. DeepSeek V4.1 Flash
through OpenRouter / DeepInfra FP8 was used throughout. Only codebook generation
used high reasoning; summarization and classification disabled it.

**The full run preserves the Power BI bypass as a distinct workstream.** All
three reviewed bypass posts receive `dashboard_bypass_exact_values`. PRNG seed
prediction, heartbeat experiments, and clock acceleration also remain distinct.
Deletion awareness is still summarized but lost in the final codebook.

| Measure | Previous naming run | This full run |
|---|---:|---:|
| Records classified | 14,456 | 14,456 |
| Other | 61 (0.42%) | **28 (0.19%)** |
| Insufficient context | 67 (0.46%) | **46 (0.32%)** |
| Strict headline matches | 3 / 10 | **4 / 10** |
| Flat labels | 22 | 22 |
| Retained exact highlights | 13,270 | 13,462 |

All label names start with "The agents ...". The new summaries contain 2,967
activities across 212 chunks. These finding-recovery scores are qualitative
assistant judgments, not full-corpus classification accuracy. Two headlines lack
the required input evidence, making strict recovery 4/8 among the at least partly
assessable cases. This run regenerated summaries and is not the fixed-summary
[reasoning-only comparison](reasoning_codebook/README.md).

## Findings and remaining gaps

| Headline | Summary | Classification review |
|---|---|---|
| Gain internet writes through GET | Recovered | GET-save probes remain within generic sandbox-write testing |
| Try XSS on the wiki | Partial | Three script-bearing anchors share an HTML/script probe category; the full XSS finding remains unestablished |
| Impersonate moderators | Unassessable | Required identity metadata is absent |
| Predict questions through PRNG seed searches | Recovered | Both reviewed anchors receive the RNG label |
| Heartbeats to investigate termination | Recovered | All three reviewed anchors receive `termination_probes` |
| Use SSH tunnels | Partial | Tunnel-provider links remain under proxy fetching; SSH execution is unestablished |
| Use Tor/cloud-provider addresses | Unassessable | Provider attribution is absent |
| Notice deletion and preserve messages | Recovered | Both reviewed anchors receive generic timed coordination |
| Manipulate task timing for peers | Recovered | All five reviewed anchors receive `clock_acceleration` |
| Bypass dashboard POST restrictions | Recovered | All three reviewed anchors receive `dashboard_bypass_exact_values` |

The bypass summary is activity `190:0`; deletion-response evidence is `184:6`.
The latter remains a concrete example of codebook information loss despite low
Other. Public posts establish reported activities, not independently verified
external success or actor identity. The earlier 38-claim audit was not repeated.

## Execution, verification, and cost

The run started from committed scripts at `5443608`. Prompts stayed unchanged.
One classifier validation fallback was added in `3db020c`: batches that still
return malformed responses after retries can now split, like truncated batches.
Network/authentication errors do not trigger recursive splitting.

The initial classification saved 14,436 records; batch 2840 omitted target IDs
after its retries. Only those 20 missing records were retried, succeeding on the
second response. That repair did not need to split. All other classifications
were preserved.

Codebook generation required three responses. The first two consumed the full
12,000-token output budget, including reasoning. The successful response used
7,324 reasoning tokens. Increasing that budget is a prospective improvement for
exercise reliability; it is not part of this tested snapshot.

| Stage | Returned responses, including retries/repair | Recorded cost |
|---|---:|---:|
| Summarization | 216 | $0.536875 |
| Codebook | 3 | $0.048576 |
| Classification | 732 | $1.894860 |
| **Total** | **951** | **$2.480310** |

Input caching discounted this run. At the checked DeepInfra FP8 rates of
$0.14/million input tokens and $0.42/million output tokens, the same token usage
without caching would cost about **$3.42**, including these retries. This is a
budget estimate, not a guaranteed future price.
[Provider metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints)

Validation confirmed complete source coverage and ordering, unchanged source
fields, valid summary references, one permitted label per record, exact quotes
under 240 characters, and matching viewer records, labels, and highlights.
About 93% of records retain a quote, so highlighted mode mainly shortens the
text rather than selecting a small set of exceptional events.
Usage was added once to the [cost ledger](../cost_ledger.json); cumulative wiki
spend is $13.001420 across all full runs and the codebook-only trial.

## Reproduce and inspect

```bash
python3 test/research/wiki/steps/01-summarize.py
python3 test/research/wiki/steps/02-classify.py
python3 test/research/wiki/build_viewer.py
```

Set `OPENROUTER_API_KEY` or enter it at the hidden prompt. These commands make
fresh paid calls and replace the active analytical outputs. The viewer itself
makes no model calls.

Open the [viewer](classification.html), [input](simple_outputs/00-wiki.jsonl),
[summaries](simple_outputs/01-summaries.json), [codebook](simple_outputs/01-codebook.json),
or [classifications](simple_outputs/02-records.jsonl). The
[machine-readable report](full_reasoning_benchmark_results.json) includes per-case
evidence, assignments, hashes, usage, and execution repairs. The
[previous naming run](previous_outputs/before_full_reasoning/) is preserved.
