# Wiki run: action-based names on the baseline prompts

Completed 24 September 2026 from prompt commit `750152e`. All stages used
DeepSeek V4.1 Flash through OpenRouter / DeepInfra FP8, with reasoning disabled.
The prompts stayed unchanged throughout the run. No target wiki findings were
supplied to the model; four examples came from a separate METR incident.

These are historical results, preserved in
[before_full_reasoning](previous_outputs/before_full_reasoning/) before the
subsequent full pipeline run with codebook reasoning enabled.

**The requested naming style is present, while strict headline recovery matches
the original baseline.** All 22 labels start with "The agents ...". PRNG seed
prediction, heartbeat experiments, and clock acceleration remain distinct.
The Power BI bypass and deletion-response story are summarized but still lost
when the codebook is built.

| Measure | Original baseline | Reverted METR experiment | This run |
|---|---:|---:|---:|
| Records classified | 14,456 | 14,456 | 14,456 |
| Other | 86 (0.59%) | 289 (2.00%) | **61 (0.42%)** |
| Insufficient context | 84 (0.58%) | 188 (1.30%) | **67 (0.46%)** |
| Distinct full headline matches | 3 / 10 | 1 / 10 | **3 / 10** |
| Flat labels | 22 | 22 | 22 |

These are qualitative finding-recovery judgments, not classification accuracy.
Two headlines lack the required input evidence; among the eight at least partly
assessable headlines, strict recovery is 3/8. One run per prompt does not isolate
the prompt's effect from model variation.

## What survived

| Finding | Summary | Final labels |
|---|---|---|
| Internet writes using GET | Recovered | Generic wiki-edit probe label |
| XSS attempts | Partial | All three script-bearing anchors now share the HTML/script-injection probe label |
| Moderator impersonation | Unassessable | Required identity evidence absent |
| PRNG seed prediction | Recovered | Dedicated label; both anchors correctly assigned |
| Heartbeats to investigate termination | Recovered | Dedicated detached-heartbeat label; some related records remain in counter/coordination labels |
| SSH tunnels | Partial | Tunnel-provider URLs remain under generic proxy chains; SSH is unestablished |
| Tor/AWS/DigitalOcean addresses | Unassessable | Provider attribution absent |
| Notice deletion and preserve messages | Recovered | Absorbed into timed coordination |
| Manipulate task timing for peers | Recovered | Dedicated clock-acceleration label; all five anchors correctly assigned |
| Power BI POST restriction bypass | Recovered | No fitting label; anchors assigned Other or timed coordination |

The script-probe grouping is an improvement within the partial XSS result.
The broader rendering/redirect definition and available evidence do not establish
the full baseline finding, so it is conservatively not counted as a fourth full
match. Neither posted scripts nor any label prove successful execution.

Summary activities `190:0` and `184:5` explicitly preserve the bypass and deletion
response. Their loss is therefore a codebook-consolidation failure. Low Other
does not detect this: a bypass post can still receive a generic coordination
label. Several labels also distinguish data-source names instead of strategies.

## Execution and cost

| Stage | Returned responses, including retries | Cost |
|---|---:|---:|
| Summarization | 214 | $1.111692 |
| Codebook | 1 | $0.032899 |
| Classification | 734 | $1.957538 |
| **Total** | **949** | **$3.102129** |

All 212 chunks produced summaries, with 2,937 activities. The classifier's 723
initial batches completed using its existing retries and automatic splitting.
No prompt repairs or additional discovery/classification pass were needed.
All returned usage was added once to the [cost ledger](../cost_ledger.json).
At completion, cumulative wiki usage totaled $10.485126 across the three runs.

Validation confirmed unchanged source records, complete coverage and ordering,
valid summary references, flat labels, and 13,270 exact highlights under 240
characters. The rebuilt viewer preserves every record, label, and quote.
The earlier 38-claim audit was not repeated for this run.

## Follow-up codebook experiments

The first follow-up reused these summaries and enabled reasoning only for codebook
generation, keeping the prompt and other settings fixed. This isolates a small
change before paying for another full classification. The original call consumed 228,294 input
tokens and cost $0.032899. At the model's listed rates, the same input plus 12,000
output tokens including reasoning would be about $0.037 before caching discounts.
[Model metadata and pricing](https://openrouter.ai/api/v1/models)

Keep the final JSON unchanged and allow enough total output budget for both
reasoning and labels. Reasoning tokens are billed and count toward the output
limit. [OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

If that does not help, test a separate prompt revision: merge routine variants
of the same behavior across data sources, preserve distinct mechanisms, and
check for important methods left without a matching definition before returning
the codebook. The target findings should remain in the held-out review.
The [reasoning-only experiment](reasoning_codebook/README.md) has now run: 7,233
reasoning tokens, $0.035984, and a new category explicitly retaining the Power BI
bypass. It has not been classified. The separate prompt revision proposed above
has not been run.

Open the archived [viewer](previous_outputs/before_full_reasoning/classification.html),
[summaries](previous_outputs/before_full_reasoning/01-summaries.json),
[codebook](previous_outputs/before_full_reasoning/01-codebook.json), or
[classified records](previous_outputs/before_full_reasoning/02-records.jsonl). The
[machine-readable comparison](naming_benchmark_results.json) includes source IDs,
summary evidence, assignments, hashes, validation, and costs. The
[original results](previous_outputs/before_metr_examples/) and
[reverted experiment](previous_outputs/metr_examples/) remain available.
