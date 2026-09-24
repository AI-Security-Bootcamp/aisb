# Wiki collection, provenance, and investigation methods

Reviewed 22 September 2026. This report concerns externally recorded activity. It does not infer hidden reasoning from posts or treat user names as verified agent identities. Collection used public read interfaces; the DSE preference submission established an anonymous reader session and did not edit a page. No paid model calls were used for these collection checks.

## What is available and what the baseline analyzed

The [published download](https://collusion.wiki/explorer/download) supplies four-wiki revision histories, event records, additional wiki selections, shortener records, and a broader text/link compilation. The [current sites index](https://collusion.wiki/explorer/sites) lists 30 sites spanning wikis, shorteners, paste services, a document store, and a package registry. Its units differ by site; its headline counts are not interchangeable with revision counts or unique actors.

| Local source | Verified contents | Interpretation |
|---|---|---|
| [full-wiki-logs](../../full-wiki-logs/) | 14,591 revisions on 4,579 pages; DSE 13,403, Probier 1,013, Fractal 169, Dorfwiki 6. All five files passed the supplied `SHA256SUMS` check. | A published historical selection, cut at write date 1 May 2026, not a census of all ordinary wiki activity. |
| [events.jsonl](../../full-wiki-logs/events.jsonl) | 14,591 save rows, 5,217 delete rows, four publisher-derived revert rows, 101 publisher-labeled probe rows. | These populations overlap. The physical total of 19,913 rows is not a total of distinct incidents or actors; the manifest explicitly warns against summing them as such. |
| [other-wikis.json](../../additional-public-data/other-wikis.json) | Eight pages: PublicTestWiki 58 selected revisions across four pages, Uncyclopedia 17 on one page, UseMod 15 across three pages. | Selected historical material; counts exclude ordinary revisions omitted by the publisher's selection. |
| [shortener-logs.json](../../additional-public-data/shortener-logs.json) | A publisher-selected set of 499 rmn.re links. | Records of link creation/selection are not proof that an agent successfully opened the destination. |
| [records.jsonl](../../additional-public-data/records.jsonl) and [links.jsonl](../../additional-public-data/links.jsonl) | 13,703 text records and 23,877 link records, with provenance and selection fields. | Multiple origins can attach to one text. A copy in a community repository is a provenance layer, not another independent observation. |
| [site-coverage.csv](../../additional-public-data/site-coverage.csv) and [coverage-gaps.csv](../../additional-public-data/coverage-gaps.csv) | Published inventory includes attempted coverage and unresolved gaps. | A listed site or successful HTTP response does not establish substantive recovered content, agent attribution, or whole-site completeness. |

[prepare.py](../prepare.py) selects nonempty inserted/replaced text from the four-wiki release and the three additional wiki selections. Its resulting [inventory](inventory.json) contains **14,456 units across seven wikis**, with DSE accounting for 13,194. The initial discovery sample has 197 units; the separate classification pilot covered 363 wiki rows. The executed structured classification scan uses **209 wiki jobs** in [scan_jobs.json](../scan_jobs.json), with chunk targets of 85,000 text characters or 100 records, plus nine technical retries for truncated wiki outputs. The inventory's `full_pass_chunks: 387` instead uses preparation defaults of 42,000 characters/70 records and is not the executed count. The baseline therefore does **not** cover all 30 indexed sites, the broader text compilation, shortener records, deletions, or request evidence. Those omissions are collection/input limitations, not findings that the omitted activities did not occur.

The baseline removes publisher semantic categories from model input but still uses publisher-generated diff hunks and reconciled timestamps. This is source-only model discovery with inherited preprocessing, not a wholly independent reconstruction. The first retained revision may also contain earlier material; copying that material into a new revision does not make the saving account its original author.

## Demonstrated collection beyond relying on the export

### DSE archive reads work

Initial ordinary page/history requests returned pages but did not recover the desired historical bodies. Published [ProWiki archive documentation](https://www.prowiki.org/prowiki/wiki.cgi?PageArchive) and the downloaded [source](../sources/prowiki-source.pl) establish the archive operations. After the ordinary anonymous preferences form, `action=archive&cmd=list&id=…` enumerated revisions and `action=archive&cmd=page&version=…&id=…` returned them. The page's linked `edit=1&html=0` variant exposes archived text in a textarea; the collector only reads that response and never submits its contents.

[check_dse_archive.py](../check_dse_archive.py) made a bounded recovery:

- All 40 versions listed for three known cases: `OAIEquityDec30Raw` (15), `Mar30TooltipEvidence` (9), and `OECDJun26PrecisionScout` (16).
- The earliest listed version for each of twelve export pages selected with seed `20260922`.
- **52 recovered revision bodies across 15 pages, all 52 exactly matching their corresponding exported body.**

The evidence is preserved in [recovered revisions](dse_recovered_revisions.json), [page/version coverage](dse_collection_coverage.json), and the [73-request acquisition log](../sources/dse-reader.acquisition.json), including reader setup and preliminary probes. Raw captures retain URLs, retrieval times, and hashes.

This demonstrates that live historical recollection is feasible and validates these exported bodies. It does not establish that all historical pages or versions remain recoverable, that deleted pages can be enumerated, or that the export's publisher selection is complete. Known-case pages were author-guided checks; the twelve random pages were sampled from the export, so they test preservation among already-known pages rather than discovery outside it.

### PublicTestWiki yields a less filtered historical sequence

[collect_testwiki.py](../collect_testwiki.py) queried the read-only [MediaWiki revision API](https://publictestwiki.com/w/api.php?action=query&prop=revisions&titles=Sandbox&rvprop=ids%7Ctimestamp%7Ccontent%7Cuser%7Ccomment&rvslots=main&rvstart=2026-05-01T00%3A00%3A00Z&rvend=2026-05-28T00%3A00%3A00Z&rvdir=newer&rvlimit=100&format=json), followed continuation, and recovered **86 full `Sandbox` revisions** in two responses for the requested 1–28 May interval. The first and last returned revisions are dated 6 May and 27 May. Raw API responses and the [acquisition log](../sources/testwiki-collection.acquisition.json) are retained; [derived additions](testwiki_recovered_deltas.json) preserve revision IDs, parent IDs, dates, and source URLs.

Three additions contain `web2md.site`: revisions [82066](https://publictestwiki.com/w/index.php?oldid=82066), [82068](https://publictestwiki.com/w/index.php?oldid=82068), and [82072](https://publictestwiki.com/w/index.php?oldid=82072), all on 17 May. They establish posted links, not successful downstream fetches. The first returned body is context because its predecessor lies outside the collected range. These 86 revisions on one page cannot be subtracted from the publisher's 58 selected revisions across four pages to obtain a meaningful missing-revision count.

The acquisition preserves mundane tests and ordinary intervening edits. That makes it useful for assessing a proposed progression from tests to link posting without selecting only the apparent endpoint. Different displayed editors and copied link strings require explicit attribution checks before treating the whole sequence as one agent's adaptation.

### Wayback captures recover bodies absent from the export

Both the CDX index and actual archived captures were reachable. Five seeded selected pages marked as absent from the revision export were then opened, and **all five contained actual page content**:

| Page | Capture time, UTC | Directly inspected body |
|---|---|---|
| [AgentClarkNewsTestA](https://web.archive.org/web/20260609113112id_/https://prowiki.org/dse/wiki.cgi?AgentClarkNewsTestA) | 9 June, 11:31:12 | A hello/test marker and a `pure.md/example.com` link. |
| [ZZMyCreateQ54828636](https://web.archive.org/web/20260609123210id_/https://prowiki.org/dse/wiki.cgi?ZZMyCreateQ54828636) | 9 June, 12:32:10 | A save-test string and summary text. |
| [AgentClarkVariantDZ](https://web.archive.org/web/20260606225236id_/https://prowiki.org/dse/wiki.cgi?AgentClarkVariantDZ) | 6 June, 22:52:36 | A short marker. |
| [Agent0CharlestonCitations](https://web.archive.org/web/20260609112852id_/https://prowiki.org/dse/wiki.cgi?Agent0CharlestonCitations) | 9 June, 11:28:52 | A greeting. |
| [AgentYahooFeedUniquePpq](https://web.archive.org/web/20260606230317id_/https://prowiki.org/dse/wiki.cgi?AgentYahooFeedUniquePpq) | 6 June, 23:03:17 | Reference-link text and `example.com`. |

See [selection records](wayback_extra_selection.json), [acquisition records](../sources/wayback_extra_requests.acquisition.json), and `test/research/sources/wayback-extra-{0..4}.{html,txt}`. The Agent0 page was also checked earlier, so these are five distinct recovered pages, not six. Capture times establish that the captured state existed by then; displayed last-change dates are a separate clock and are not complete edit histories. Agent-like titles alone do not authenticate authorship. These small bodies are evidence of coverage gaps, even when their contents are mundane.

### Additional origins outside the current sites index

A local audit of `records.jsonl` found seven additional origin labels not listed as sites in the current index:

| Origin | Origin occurrences |
|---|---:|
| `url.popcat.xyz` | 76 |
| `u.ethz.ch` | 9 |
| `paste.probyte.ee` | 5 |
| `pastebin.tarcseh.me` | 5 |
| `infinitypaste.club` | 3 |
| `html.cafe` | 2 |
| `paste.flashrom.org` | 2 |

These are **103 origin occurrences**, not 103 unique posts or verified agents, and not seven newly discovered sites: they already occur in the broader published compilation. Their distinction from the narrower sites index is an inventory finding. Preserve each original source URL, acquisition date, selection basis, and authorship limitation before using their contents. A live independent recollection of these origins was not demonstrated in this pass.

## Published reconstruction methods

Published collection code was also inspected at pinned repository revision `60c9373e6ebe8374dfb6a91f73d40db38f165493`. The community archive's [wiki crawler](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/60c9373e6ebe8374dfb6a91f73d40db38f165493/scripts/wiki_crawler.py) discovers candidate engines from a bounded link crawl, probes engine-specific RecentChanges interfaces, and records signature matches. Its host-level deduplication can miss multiple installations on one host; a signature match is only a lead. The [archival crawler](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/60c9373e6ebe8374dfb6a91f73d40db38f165493/scripts/archival_crawler.py) is explicitly a sketch: candidate GETs produce response hashes, optional capture files, and a manifest, with failures distinguished from real responses. Neither script establishes exhaustive historical revision recovery. They were read, not executed; this investigation used the narrower known-interface collectors described above. The fetched source hashes were `9012a0bc2f92ca4508f0738cec0bf67305c6fb2e6d6ef0cf0a2cbfde89b812bc` and `3945c825e87247bd9113f48c5ab7fa14ed2c1a81c4c17cf9a805184d614316e8`, respectively. No additional local raw copy was created.

The original [collusion.wiki report](https://collusion.wiki/) combines archived edits with evidence not fully reproduced in the revision download, including web-server requests. Its [export manifest](../../full-wiki-logs/manifest.json) records source provenance, timestamp reconciliation, linked events, and checks. Use those fields to understand the export and conduct a reference audit; keep semantic conclusions such as task-family assignments and probe categories outside discovery prompts. Exported summaries of unavailable request logs cannot independently validate those logs.

The [Swarm reconstruction study](https://arxiv.org/html/2609.12748v1) extracts statements from revision deltas and separates server evidence, agent self-reports, modeled cohorts, and interpretation. It tests regex extraction against 400 stratified sentences labeled by two independent model raters, with disagreements manually adjudicated. It reports high precision but incomplete recall for progress extraction and warns about cumulative-page attribution, copied claims, missing reads, timer interpretation, and unsupported PRNG inferences. Its cohort/population and timing models require explicit assumptions; their estimates are not observed agent counts. Temporal exposure can bound possible influence but cannot establish transmission or causal performance gains. Transfer the evidence classes, delta attribution, negative cases, and manually reviewed validation set. Do not import its numerical estimates or inferred cohorts into our discovery as facts. [Local reviewed text](../sources/swarm-methods.txt)

[MessageBoardAuditBench's data-processing specification](https://github.com/hamzah2304/messageboardauditbench/blob/main/docs/data-processing.md) audits contamination from fields, values, IDs, and aggregates. Its raw variant removes publisher judgments; its augmented variant adds explicitly identified report excerpts. This demonstrates why dropping a field named `label` is insufficient: a `probe` type, an `attacklog` ID, or an administrator designation can still reveal the answer. Its additions and synthetic-name substitutions must remain distinct from original records. For this investigation, keep the raw captures unchanged, construct neutral analysis views, and use its [claim/feasibility artifacts](https://github.com/hamzah2304/messageboardauditbench/tree/main/benchmark) only in the reference audit. [Local processing specification](../sources/auditbench-processing.md)

## Reproducible refinement and evidential limits

1. Pin raw bytes and source hashes before normalization. Record the requested and final URL, request time, status, content type, and selection reason. Classify empty/template/challenge responses separately from recovered content; HTTP 200 is not enough.
2. Enumerate revisions through each engine's documented read interface and follow continuation. For archive aliases, record both host and full query string, then verify actual captures. Retain unsuccessful reads and selection exclusions.
3. Recompute predecessor deltas for a validation subset, preserving full before/after bodies. Treat first-observed states, deletions, replacements, and re-creations distinctly. The currently demonstrated 52 exact body matches validate body preservation, not all publisher diff boundaries or clocks.
4. Include mundane tests, failed saves, sparse sources, and deletion/request evidence in separate analysis strata. Keep source-only exploration separate from author-guided recovery of the known PublicTestWiki and Power BI cases.
5. Preserve server time, post-reported time, archive capture time, acquisition time, and uncertainty separately. Do not turn a page's first/last observation into continuous agent activity.
6. Validate story edges: an earlier post, a later similar post, and a matching username do not establish that the latter author read the former. Explicit acknowledgment is stronger evidence but still a self-report unless joined to an independent read record. Require corroboration before claiming an external exploit or successful policy bypass.
7. Freeze baseline outputs before reference comparison. Diagnose misses as omitted sources, sampling, lost context, extraction, grouping, or evidence validation. Keep improved author-guided recovery separate from independent discovery, and retain unassessable findings in the all-findings denominator.

The demonstrated collection improves historical coverage and makes a broader next pass feasible. It does not recover private harness state, complete successful-read logs, hidden agent identities, or ground-truth task outcomes. Those absences constrain claims about causal coordination and performance even if every retained page is read correctly.
