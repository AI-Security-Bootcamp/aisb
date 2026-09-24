# Full wiki run — DeepSeek V4.1 Flash

These saved results predate the [behavioral-label prompt update](../steps/README.md#behavioral-labels).
The updated scripts now use examples from a separate METR incident; these outputs have not
been regenerated with it.

Completed 24 September 2026. Summarization, codebook generation, and classification
all used **`deepseek/deepseek-v4.1-flash`**, pinned to **DeepInfra FP8 through
OpenRouter**, with reasoning disabled. There is one flat codebook and one retained
classification pass; no Other-refinement pass or participant exercise was added.

Open the [classification viewer](../classification.html). Click a bar to read its
source, a label to filter that activity, or **Highlighted passages only** to view
literal excerpts. [Scripts and instructions](../steps/README.md) explain rerunning
the two numbered steps and rebuilding the viewer.

## Datasets

| File | Contents |
|---|---|
| [00-wiki.jsonl](00-wiki.jsonl) | All 14,456 nonempty revision changes across seven wikis; byte-identical to `../units.jsonl`. |
| [01-summaries.json](01-summaries.json) | All 212 chronological chunks, with named activities, descriptions, and original supporting record IDs. |
| [01-codebook.json](01-codebook.json) | 22 flat labels, each with an ID, name, definition, and exclusion boundary. |
| [02-records.jsonl](02-records.jsonl) | Every original source record, unchanged and in order, plus one label and at most one quote. |
| [costs.json](costs.json) | Returned API usage and costs, including failed attempts and repairs. |

This covers the prepared seven-wiki revision dataset, not every site or evidence
type on collusion.wiki. See [collection scope and provenance](../COLLECTION_AND_METHODS.md).
Model inputs exclude previous labels and publisher conclusions. Codebook input
contains every activity name and description from the summaries; record-ID lists
remain in the summary artifact but are omitted from that consolidation request.

## Results and limitations

| Check | Result |
|---|---:|
| Source records classified exactly once | 14,456 / 14,456 |
| Codebook labels | 22, all flat |
| Other | 86 (0.59%) |
| Insufficient context | 84 (0.58%) |
| Records with a retained quote | 13,501 (93.39%) |
| Nonliteral or overlong quotes discarded in the retained pass | 915 |

Every retained quote is an exact substring of its own source record and shorter
than 240 characters. Invalid quotes were cleared without removing the record or
its label. Quotes on 93% of records mean highlighted mode mainly compresses the
text; it is not a selective list of only surprising records.

**Low Other is not evidence of adequate discovery.** The full summaries describe
the Power BI proxy bypass, but the codebook has no dedicated label for it. In
targeted spot checks, these records were assigned `denominator_methodology`:

- `W:dse~OECDJun26PrecisionScout@13`
- `W:dse~OECDLivePBIProof@1`
- `W:dse~OECDJun26PrecisionScout@14`

That label describes raw-versus-displayed values and precision debates. It absorbs
the surrounding task discussion while failing to distinguish the bypass method.
These are known-case checks, not an unbiased accuracy estimate. This taxonomy
gap remains in the saved results for the next investigation round; no author-led
label correction was applied.

Other targeted checks retained useful distinctions: the seed-search post
`W:dse~DataUSAStateSequenceCollab2027@12` is labelled `rng_prediction`, and
`W:dse~CashierCoordOct06OAI@2` / `@3` are labelled `counter_beacon_signaling` /
`termination_probing`. The overall output contains 27 RNG-prediction records,
302 counter-beacon records, 299 termination-probing records, and 177 clock-wait
acceleration records. These are classifier assignments, not independently
validated incident counts. Posted claims of external success remain self-reports.

## Measured cost and technical repairs

**Total recorded API cost: $4.1082952232**, including all technical retries and
the discarded classification attempt. These charges were also added once to
[the investigation ledger](../../cost_ledger.json).

| Work | Returned API responses | Recorded cost |
|---|---:|---:|
| Summaries, including repairs | 334 | $1.43714 |
| Codebook generation | 1 | $0.02698 |
| Discarded classification attempt | 730 | $1.89110 |
| Retained classification attempt and repairs | 732 | $0.75308 |
| Total | 1,797 | **$4.10830** |

The cheaper retained classification attempt benefited from cached input. Its
$0.75 cost should not be treated as the price of a fresh full classification run.
Costs above are OpenRouter's returned `usage.cost`, not token-price estimates.

The initial summary stage left 14 chunks failing format/reference checks after
retries. Short request IDs were introduced and mapped back to original IDs;
198 already validated summaries were retained. The final three failures came
from a hard 15-activity limit. That optional limit was relaxed to preserve extra
findings, while keeping source-reference checks mandatory.

The first classification attempt failed one batch and exited before writing
successful records. Its full $1.89 charge is included above. The script now saves
successful records before reporting batch failures and gives validation feedback
on retries. The rerun preserved 14,436 classifications; one remaining batch
repeatedly reached the 8,000-token output limit. Smaller target groups completed
those final 20 records. Truncated batches now split automatically. Normal
classification batches contain 20 targets and five surrounding context records
on each side, with 64 concurrent requests.

Verification covered full source preservation and ordering, complete summary and
classification coverage, valid supporting IDs, permitted flat labels, literal
short quotes, unique API response accounting, and zero reported reasoning tokens.
These are structural checks; they do not establish semantic classification
accuracy or the fraction of the original authors' findings recovered.

Browser checks confirmed that all saved labels and quotes survive viewer
conversion, source selection and highlighted mode work, labels filter correctly,
and the page reports no JavaScript errors.
