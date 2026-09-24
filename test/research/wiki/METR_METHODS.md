# METR investigation: implications for the wiki analysis

Reviewed 2026-09-24. This is an investigation note, not participant material.

The original prompt experiment described below has been reverted. Its scripts
and results are [archived](previous_outputs/metr_examples/). The active scripts
now include the narrower naming revision described next; its completed run is
shown in the saved results and viewer. See the [scripts README](steps/README.md).

## Current naming revision

The [active summarizer](steps/01-summarize.py) reuses the first four examples
listed below: flag generation, target substitution, tool-call spoofing, and
research handoffs. These teach short "The agents ..." findings, with concrete
methods, supported purposes, and qualified outcomes. They do not supply the
wiki's target findings. The codebook prompt keeps that naming style and avoids
merging distinct methods solely because they serve a common goal.

This revision only changes the two prompts. Classification, batching, models,
and formats are unchanged. Its [completed benchmark](NAMING_BENCHMARK.md) finds
three strict headline matches, Other 0.42%, and insufficient context 0.46%, for
$3.102129. All 22 names follow the requested style, but the bypass and deletion
response still disappear in consolidation. The previous experiment's scores
below describe that separate, archived run.

The subsequent [codebook-only reasoning experiment](reasoning_codebook/README.md)
reuses those summaries and prompts. It retains the bypass as a distinct
definition, uses 7,233 reasoning tokens, and costs $0.035984. Deletion awareness
still has no matching label. This candidate has not been used for classification;
the saved full-run outputs and viewer remain unchanged.

## Original investigation notes

My recommendation is to keep the simple summarization → codebook → classification
pipeline, but make its unit of discovery an evidenced activity with a method and
purpose. A label should give a reader something to investigate on the timeline.
Its name alone should not certify success or establish an agent's motive.

## What the report actually supports

METR ran twelve behavior-specific transcript sweeps, then followed up high-rated
and sampled findings. [Sweep methodology](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#classifier-sweeps-for-important-behaviors).

Separately, it linked important message reads and writes with context, developed
56 workstream subcategories from investigator notes, classified with Terra, and
merged unreliable distinctions. This was iterative, investigator-informed
reconstruction. [Workstream methodology](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#identifying-important-messages-and-classifying-them-by-workstream).

The report describes false beliefs about scoring, analysis-agent errors,
incomplete records, and spoofed tool outputs. Its impact weighting uses readers'
reactions. [Scoring beliefs](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#coordinated-workstreams-to-fool-the-evaluation-process),
[limitations](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#limitations-due-to-the-scale-and-complexity-of-this-incident),
[weighting](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#appendix-importance-weighted-workstream-activity).

The five naming examples are retained in the
[archived summarizer prompt](previous_outputs/metr_examples/01-summarize.py).
Examples 1, 2, and 5 are grounded in the
[collective-project account](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#coordinated-workstreams-to-fool-the-evaluation-process).
Example 3 uses the
[tool-call investigation](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#agents-engaged-in-a-large-and-successful-collective-project-to-develop-techniques-for-spoofing-tool-calls).
Example 4 uses the
[research handoff](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#july-9th-phaseone10841-passes-on-its-work-to-phaseonebig-which-establishes-several-ambitious-workstreams).
The wording is our adaptation, not a quotation or METR's published taxonomy.

## My assessment for our pipeline

The following are design judgments and proposed checks for our data, not additional
claims about METR's implementation.

**A workstream needs an analytical level between a broad goal and individual
artifacts.** A broad label can absorb almost everything relevant to finishing a
task. An artifact label can split one activity across several tools. Our useful
middle level is an action or method whose progress could form a coherent story.
For each proposed label, ask whether a reader could describe what changed over
time within it. Merge routine implementation variants, but retain a materially
different method even when it serves an existing goal. Keep this as one flat
list; we do not need parent labels to express purpose in the name.

**Preserve the distinction between purpose, belief, and outcome.** A posted
explanation can establish what its author claimed to be trying to do without
establishing that the explanation is sincere or that its assumptions are correct.
Summaries should retain that attribution. Separately ask what was proposed,
attempted, demonstrated, or merely reported. A component-level result and an
end-to-end result need different wording. This fits in the existing summary
string; adding several mandatory fields would complicate the interface without
automatically improving the evidence.

**Context helps interpretation but does not supply missing observation.** Our
classifier sees 20 targets with five neighboring records on either side. That
helps with terse replies, but neighboring revisions can concern unrelated actors
or repeat earlier text. It cannot tell us who read a post, what they privately
thought, or whether an external action happened. An explicit reply or reproduction
claim is evidence of a reported connection; mere proximity is not. Treat missing
thread context as a limitation of the available records, not something to fill in
from knowledge of the incident.

**Importance needs a criterion independent of message count.** For our summaries,
I would prioritize a changed capability, a new method, a decisive failure, or
evidence of adoption. A highly repetitive activity should not crowd out a brief
but consequential one. These are reviewer judgments, not a validated importance
score. Without reader traces we cannot estimate behavioral influence from posting
volume. Timeline density should remain a count of observations, not a measure of
impact, agent effort, or continuous work.

**Coverage and correctness are different questions.** Our saved run classified
every input record, yet its [known-case review](simple_outputs/README.md) found a
consequential method described in summaries and lost during codebook consolidation.
That isolates a failure between steps, rather than demonstrating a need for a
larger classifier. Low Other is compatible with broad labels swallowing unfamiliar
behavior. Conversely, high Other can reflect useful abstention. Neither number
alone establishes that we reconstructed the important stories.

**The codebook is the main compression bottleneck to inspect next.** We send all
activity descriptions into one consolidation request, with a maximum of 22 labels.
Changing sentence style will not necessarily preserve a rare method under that
limit. Review each distinctive summary activity against the resulting definition
and exclusions. If it disappeared, inspect this consolidation before adding
classifier passes. The current change retains the limit and one-pass design;
it does not demonstrate that those choices are sufficient.

## Review after the next run

These checks are proposed, not completed validation of the new prompts:

| Check | Evidence to inspect | What it distinguishes |
|---|---|---|
| Discovery | Original records against `01-summaries.json` | Missing an event versus summarizing it incorrectly |
| Consolidation | Distinctive summary activities against `01-codebook.json` | Losing a method versus intentionally merging equivalent ones |
| Classification | Selected records, their context, and label definitions | A bad assignment versus a missing label |
| Highlighting | The retained quote within the full original record | A relevant passage versus a merely literal substring |
| Timeline | Selected bars against individual revision times | Observations versus inferred duration or sequence |

Review a mixture of unusual cases, ordinary cases, and ambiguous assignments;
looking only at the most striking quotes would conceal routine errors. A small
review can diagnose problems but should not be presented as an accuracy estimate
for the entire dataset.

Keep the published wiki findings outside the model requests. After generating the
results, use them as a separate reference for checking which stories were recovered
and where omissions arose. Distinguish exact recovery, partial recovery, and
unsupported additions. Removing target examples prevents direct answer leakage;
it does not establish that a pretrained model has never seen the publication.

## Changes made

The experimental summarization and codebook prompts used the separate incident's examples
and the distinctions above. Target examples and case-specific hints were removed.
The classification prompt already requires contextual evidence, one flat label,
and at most one literal quote, so it needed no further change.

The initial prompt edit added no extra pass, output field, model call, or viewer
feature. All wiki stages remain on DeepSeek V4.1 Flash. A subsequent full run and
[benchmark review](BENCHMARK.md) now evaluate these prompts: the summaries preserve
several key findings, but codebook consolidation loses important distinctions.
The [experimental results and viewer](previous_outputs/metr_examples/) are
archived, and that rerun cost $3.274702. It was subsequently reverted; the current
results are from the narrower naming revision above. All charges remain recorded.
