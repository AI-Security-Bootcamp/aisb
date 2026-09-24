# Codebook-only reasoning experiment

**This candidate recovers the Power BI bypass as a distinct category.** High
reasoning also retains PRNG prediction, detached heartbeats, and clock
acceleration. Deletion awareness still has no matching category.

The run reused all 212 saved summary chunks (2,937 activities), now preserved in
[before_full_reasoning](../previous_outputs/before_full_reasoning/). The prompt,
DeepSeek V4.1 Flash / DeepInfra FP8 model, temperature, 22-label limit, and
12,000-token output budget were unchanged. The only request-setting change was
`reasoning: {"effort": "high", "exclude": true}`. The scripts were committed
as `6933577` before the request.

| Measurement | Reasoning disabled | High reasoning |
|---|---:|---:|
| Input tokens reported by provider | 228,294 | 228,318 |
| Reasoning tokens | 0 | **7,233** |
| Visible output tokens | 2,232 | 2,337 |
| Labels | 22 | 22 |
| Cost of codebook generation | $0.032899 | **$0.035984** |
| Returned responses | 1 | 1 |

The input text is unchanged; the provider reports 24 additional input tokens
with reasoning enabled. Reasoning was confirmed in returned usage, and its text
was excluded from the response. No retry was needed. This costs about **0.3 cents
more** than the earlier codebook call; it does not repeat summarization.

## What changed in the codebook

| Finding | Earlier codebook | Reasoning candidate |
|---|---|---|
| Power BI restriction bypass | Missing | `access_bypass` explicitly covers GET-only proxies, allowlisted hostname resolution, Host headers, and POST queries |
| PRNG seed prediction | Present | Preserved in `seed_bruteforce` |
| Heartbeats to investigate termination | Present | Preserved in `heartbeat_probes` |
| Manipulate task time to help peers | Present | Preserved in `clock_acceleration` |
| Notice deletion and preserve messages | Missing | Still missing |
| Gain internet writes through GET | Broad editing category | GET-save probes remain within `placeholder_pages` |
| Script injection probes | Broad HTML/script category | Preserved in `markup_injection`; full XSS finding remains partial |

The candidate groups routine data-source lists under `data_reference_lists`,
instead of assigning separate labels to several named sources. It also separates
the bypass from ordinary proxy/converter requests through an explicit exclusion.
The bypass definition is supported by summary activity `190:0`; the still-missing
deletion response is explicit in `184:5`.

At the **definition level**, four of the ten target mechanisms now have distinct
matching categories, versus three before. This is not a classification or
end-to-end recovery score. SSH remains unestablished by the selected text, and
moderator impersonation and Tor/cloud-provider attribution still lack the
required evidence. The review is qualitative and based on one candidate;
model variation prevents a firm causal claim about reasoning.

## Files and next use

- [01-codebook.json](01-codebook.json): the candidate, with unchanged flat-label format.
- [comparison.json](comparison.json): evidence-linked review, hashes, settings, and actual usage.
- [Saved summaries](../previous_outputs/before_full_reasoning/01-summaries.json): unchanged input for this experiment.
- [Saved codebook](../previous_outputs/before_full_reasoning/01-codebook.json): the non-reasoning comparison.
- [Runner](../try_reasoning_codebook.py): reruns only codebook generation.

This candidate was **not used for classification**. This experiment left the
active codebook, classifications, and viewer unchanged; there are no Other or
insufficient-context percentages for this candidate. The later full pipeline
run starts again from the source, generating fresh summaries and a fresh codebook.

The $0.035984 charge was appended once to both the cumulative wiki usage and
the [investigation ledger](../../cost_ledger.json). Cumulative recorded wiki
spend at this experiment's completion was $10.521110, including all earlier
full runs and this experiment.
