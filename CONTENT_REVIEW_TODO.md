# AI Security Bootcamp — Content Review & Remediation Plan

This file consolidates a two-pass review of the whole repo (per-exercise pedagogy,
week-level design, toolchain health, and research currency) into **dispatchable tasks**.
Each task is self-contained so a subagent can pick it up with only this file plus the
repo. Work through them by owner tag.

## How to use this file

- **Owner tags**: `[SONNET]` = mechanical / well-specified (find-replace, dead-code removal,
  typo fixes, adding tests to a clear spec). `[OPUS]` = needs judgment, design, or
  cross-file reasoning (restructuring, new content, info-hazard policy, pedagogy).
- Each task has: **Scope** (files), **Context** (why / what the reviewer found),
  a **Checklist** of concrete steps, and **Acceptance criteria**.
- **Global rules for every task that edits a `*_solution.py`:**
  - The `*_solution.py` is the single source of truth. Never edit generated
    `*_instructions.md` / `*_test.py` / `*_reference.py` by hand.
  - After editing, run `./build-instructions.sh <file>` (add `--force` if it silently
    no-ops) and confirm no new warnings. Then confirm the file still parses
    (`python3 -c "import ast; ast.parse(open('<file>').read())"`).
  - Follow the conventions in `CLAUDE.md` (note: emoji section numbers and emoji
    difficulty ratings are **retired** — see Task S1).
  - Do not weaken or delete an exercise to make a test pass; fix the underlying issue.

## Repo layout reminder

Content lives in top-level `X.Y-name/` folders, each with `sectionN_solution.py`:
- Day 1: `1.1-llm-internals`, `1.2-logprobs`, `1.3-instruction-hierarchies`
- Day 2: `2.1-prompt-injection`, `2.2-coding-agents`, `2.3-monitoring`, `2.4-control-protocols`, `2.5-safety-simulator`
- Day 3: `3.1-tokenization`, `3.2-jailbreaking`, `3.3-guardrails`, `3.4-kd-attacks`, `3.5-weight-extraction`
- Day 4: `4.1-model-editing`, `4.2-backdoor`, `4.3-safety-finetuning`
- Day 5: `5.1-adversarial-vision`, `5.2-adversarial-language`, `5.3-prefix-tuning`, `5.4-watermarking`
- Day 6: `day6-infrastructure/` (not yet split into `6.x`; has structural cruft — see O3)
- Day 7: `7.1-atlas-killchain`, `7.2-adversary-matrix`

**Legacy trap:** every day also has a stale pre-refactor `dayN-*` folder (`day1-intro/`,
`day2-agents/`, …, `day7-governance/`) containing a monolithic `dayN_solution.py` + generated
`.md`, and in some cases still-imported shared assets (e.g. `day3-inference/day3_setup.py`,
`setup/` images). These folders are outdated and diverged — **the whole `dayN-*` folder set is
slated for full removal** after shared assets are migrated into the `X.Y` layout (see Task O5).
Do not edit them; do not add new content to them.

## Boss feedback & execution status (2026-07-07)

**WAVE 1 LANDED (Days 1–5 + 7):** items 2 (stale refs), 3 (structure/emoji section numbers),
4 (sys.path→CLAUDE.md + applied), 5-scaffolds, and 1 (tests authored) are DONE for Days 1,2,3,4,5,7.
Verified: no emoji section headers remain; difficulty ratings left as-is; canonical snippet applied;
all files build + parse. **NOT yet run** (no GPU/API/Docker here) — tests are authored + build-verified
only; a human must run them on the remote box. **Still open:** Day 0 and Day 6 (Day 6 needs T-O3 dedup
first); the answer-file *naming* convention is still inconsistent across days (see below); correctness
bugs (item 6) deferred — compiled under T-O8.

Directives on the six cross-cutting issues, and what is being done in the current wave:

1. **Tests collapsed** → WRITE better tests now. **In progress** (T-O1). Note: this sandbox
   can't run them (Day 1-2 need the OpenRouter key, Day 3-6 the GPU box, Day 0 GitHub) — tests
   are authored + build/AST-verified here; each is tagged with the env it needs for a human to
   run pass/fail.
2. **Stale refs** → grep + fix. **In progress** (T-S2, README/CLAUDE.md done centrally).
3. **Structure ownership ad hoc** → fix all. **In progress** (title-block ownership + drop emoji
   section numbers; part of T-S1 + T-O4).
4. **sys.path divergence** → pick ONE, add to CLAUDE.md, use everywhere. **DONE (decision):**
   canonical snippet is now in CLAUDE.md ("Canonical `sys.path` / import boilerplate" —
   walk-up-to-`aisb_utils`, depth-independent). Applying everywhere is part of the current wave
   (T-S4).
5. **Ratings & scaffold discipline** → **LEAVE difficulty/importance ratings AS-IS** (boss will
   fix manually later). So: **T-S5 is DEFERRED**, and the emoji→`N/5` bulk rating conversion is
   **NOT** done now (CLAUDE.md still specifies `N/5` for *new* content). Dropping emoji *section
   numbers* still happens (that's structure, not difficulty). **Fix the scaffolds now** (T-O7,
   plus scaffold parts of the per-day work).
6. **Correctness bugs** → **NOTE ONLY for now**, verify + iterate after this wave lands. So
   **T-O8 and T-S8 are NOT executed yet** — they remain a tracked list to test-for-correctness
   later. (Do add tests that would *catch* these bugs where cheap, but don't change the
   suspected-buggy logic itself yet.)

---

# GROUP A — Convention migration & mechanical cleanup (mostly SONNET)

## T-S1 [SONNET] Retire emoji section numbers and emoji difficulty ratings
**Scope:** every `X.Y-*/sectionN_solution.py` (21 files) + `day6-infrastructure/day6_final_solution.py`. Do NOT touch the legacy `dayN_solution.py` (being deleted).

**Context:** Section headers currently use emoji digits (`## 1️⃣ Section Name`,
`### 1️⃣ ...`). These drift out of order whenever content is moved between days/sections
(which has happened repeatedly) and require error-prone manual reconciliation. Difficulty
and importance are currently emoji circles (`🔴🔴⚪⚪⚪` / `🔵🔵🔵🔵⚪`), which are hard to
author and easy to miscount. CLAUDE.md has been updated to the new conventions:
plain `## Section Name` headers and numeric `Difficulty: N/5` / `Importance: N/5`.

**Checklist:**
- [ ] In prose headers, strip the leading emoji digit + space: `## 1️⃣ Guardrails` → `## Guardrails`; same for any `### N️⃣ ...`. Keep `### Exercise N.M: Title` unchanged (those numbers are meaningful and stay).
- [ ] Convert every `> **Difficulty**: <emoji>` to `> **Difficulty**: N/5` where N = count of filled 🔴 (filled circles). Same for `> **Importance**: N/5` (count filled 🔵).
- [ ] Grep afterwards to confirm no stray `1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣🔴⚪🔵` remain in solution files: `grep -rlP "[\x{1F534}\x{26AA}\x{1F535}\x{0031}\x{FE0F}\x{20E3}]" */section*_solution.py`.
- [ ] Rebuild all touched files; confirm TOCs still generate (the TOC is built from `##` headers — dropping the emoji is fine).

**Acceptance:** No emoji digits in section headers; all ratings are `N/5`; all files build clean.

## T-S2 [SONNET] Fix stale paths & answer-file naming after the X.Y refactor
**Scope:** setup sections of all `X.Y-*/sectionN_solution.py`; top-level `README.md`.

**Context:** The refactor renamed folders to `X.Y-name` but many setup blocks and the
README still point at pre-refactor locations. Confirmed instances:
- Day 5 setup blocks tell participants to create files in `1-adversarial-vision`,
  `2-adversarial-language`, `3-prefix-tuning`, `4-watermarking` — all missing the `5.` prefix.
- Day 1 tells participants to create `day1_answers.py` in `day1-intro/` (legacy folder).
- Day 4 references `day4-training-and-data/` and `day4_requirements.txt`.
- Top-level `README.md` "Completing exercises" says navigate to `day1-intro` — steers
  participants into the OUTDATED legacy material.
- CLAUDE.md convention is `wNdM_answers.py` in the `wNdM` dir, but files variously use
  `dayN_answers.py` and `sectionN_answers.py`.

**Checklist:**
- [ ] In each section's setup block, point the answer-file path at the correct current
      location. Decide one convention and apply uniformly (recommend: one answer file per
      day, named `dayN_answers.py`, created in that day's first `X.Y` folder — OR follow the
      `wNdM` convention in CLAUDE.md; pick one and note it in the file). Flag to a human if unsure which folder participants actually `cd` into.
- [ ] Fix Day 5's `N-adversarial-*` → `5.N-...` references.
- [ ] Update top-level `README.md` "Completing exercises" to reference `X.Y-*` folders,
      not `dayN-*`.
- [ ] Grep for other stale folder refs: `grep -rn "day[1-7]-[a-z]" */section*_solution.py README.md`.

**Acceptance:** Every path a participant is told to use exists and matches the current layout.

## T-S3 [SONNET] Fix confirmed typos & copy-paste bugs
**Scope:** listed files.

**Context:** Discrete, unambiguous string bugs found during review.

**Checklist:**
- [ ] `PREREQUISITES.md:~193` — `conda activate asib` → `conda activate aisb` (env is created as `aisb`; this fails for everyone). Also fix "Seting"/"Seting up" header typos nearby.
- [ ] `day0-setup/day0_solution.py:~305-311` — `test_analyze_user_behavior` queries `"pranavgade20"` but assert messages say "karpathy". Make messages consistent with the actual query - drop karpathy.
- [ ] `2.3-monitoring/section3_solution.py:~144` — `inspect view --log-dir day20-agents/logs` → `day2-agents/logs`.
- [ ] `2.3-monitoring/section3_solution.py:~442` — prose "lower than your 4.1 baseline" → "3.1 baseline".
- [ ] `3.4-kd-attacks/section4_solution.py:~592` — "Combine the CE loss from exercise 3.1 with the KD loss from exercise 3.2" → "4.1" and "4.2".
- [ ] `2.1-prompt-injection/section1_solution.py` — typos "retrieves relevant document", "attacker is LLM user"; and count mismatch: 1.3 (`:~477`) says "two-step attack" but 1.2 is framed as three stages. Reconcile to one count.
- [ ] `2.2-coding-agents/section2_solution.py:~139` — broken/merged sentence "Subtle backdoors or sabotage attempts A scalable ...". Repair.
- [ ] `3.2-jailbreaking/section2_solution.py` — time budget says "~15 minutes" (`:~40`) vs "~20 minutes" (`:~83`); reconcile. Also the intro paragraph is duplicated between the file-top docstring and the `##` block — remove one (see T-S7).
- [ ] `3.5-weight-extraction/section5_solution.py:~268` — "quiet expensive" → "quite expensive".
- [ ] `day6-infrastructure/day6_final_solution.py` — `TO BE DISCOSED` and similar typos if present in the canonical file (the nested duplicate is being deleted, ignore it).

**Acceptance:** Each listed string is corrected; files rebuild clean.

## T-S4 [SONNET] Unify sys.path + report-import boilerplate
**Scope:** all `X.Y-*/sectionN_solution.py`.

**Context:** The `sys.path` insertion block appears in 2-deep, 3-deep, and
`_workspace`-based variants (four forms in Day 3 alone). `report` is imported as both
`from aisb_utils import report` and `from aisb_utils.test_utils import report` within the
same days. Both import forms work (the package re-exports), but inconsistency is noise and
some files carry a dead `sys.path` block importing nothing from the project (4.2, 4.3, 7.2).

**Checklist:**
- [ ] Define ONE canonical snippet: insert the workspace root (and the day-utils dir where a
      section imports a shared `dayN_setup`/`dayN_utils` module) onto `sys.path`, then
      `from aisb_utils import report`. Document the chosen snippet at the top of this task's PR.
- [ ] Apply it verbatim to every section that needs `aisb_utils` or a shared day module.
- [ ] Delete `sys.path` blocks from files that import nothing from the project root
      (verify with a grep for project imports first: 4.2, 4.3, 7.2 are candidates).
- [ ] Standardize on `from aisb_utils import report` everywhere (drop the `.test_utils` form).

**Acceptance:** One import/path idiom across the repo; no dead boilerplate; all files build & import.

## T-S5 [SONNET] Add missing / fix miscalibrated difficulty & importance ratings
**Scope:** listed exercises. (Do this AFTER T-S1 so the format is already numeric.)

**Context:** CLAUDE.md requires both ratings on every exercise. Missing on:
`3.2` Ex 2.1; `4.4` (2.4-control-protocols Ex 4.4 "Compare"); `5.1` Ex 1.4; `5.1` (2.5-safety-simulator Ex 5.1). Miscalibrated: `4.1-model-editing` Ex 1 is rated 1/5 for a
"read a paper + adapt a ROME notebook" task (realistically 2-3/5).

**Checklist:**
- [ ] Add `Difficulty: N/5` + `Importance: N/5` to each exercise listed as missing. For
      run-and-observe/"compare" blocks that aren't really exercises, either add ratings or
      convert the header from `### Exercise ...` to a plain `###` subsection (pick based on
      whether there's a participant task).
- [ ] Re-rate `4.1` Ex 1 to 2/5 or 3/5.

**Acceptance:** Every `### Exercise N.M` has both ratings; no 1/5 on paper-reading tasks.

## T-S6 [SONNET] Remove dead code, unused imports, and leaked authoring comments
**Scope:** listed files.

**Context:** Dead weight that can break the "solution file must run" requirement or leak answers.

**Checklist:**
- [ ] `2.5-safety-simulator/section5_solution.py` — remove the large commented-out MCP block
      (Exercises numbered "7️⃣", no 6) and the heavy imports only it used (`jwt`, `FastMCP`/`mcp`,
      `threading`, `urllib`, `datetime`, `time`). These can crash setup if the packages aren't
      installed. If the MCP content is wanted later, that's a separate new-content task, not this cleanup.
- [ ] `3.5-weight-extraction/section5_solution.py` — remove unused `log_s`/`gaps` computations
      (`:~174-176`) OR wire them into actually deriving `detected_h` (that's the T-O7 version;
      here just remove if leaving hardcoded). Remove the leaked authoring comment
      `# ... without this, SVD takes too much memory` (`:~160`).
- [ ] `1.1-llm-internals/section1_solution.py` — remove the commented-out `trust_remote_code=True` cruft in the Mistral tokenizer load.
- [ ] Grep for other leaked comments / debug prints: `grep -rn "TODO(author)\|XXX\|HACK\|too much memory\|# debug" */section*_solution.py`.

**Acceptance:** Solution files import without needing packages they don't use; no leaked comments; files run.

## T-S7 [SONNET] De-duplicate intro paragraphs (file-top docstring vs `##` block)
**Scope:** `3.2`, `3.3`, `3.4` (confirmed); check all Day 2 & Day 3 sections.

**Context:** Several sections repeat their opening paragraph verbatim — once in the
file-top triple-quoted docstring and again in the `## Section` block right below it.

**Checklist:**
- [ ] For each affected section, keep the day/section title block at the top and remove the
      redundant repeated paragraph from the `##` block (or vice versa), so the same prose
      isn't rendered twice in the instructions.
- [ ] Verify the generated `.md` reads cleanly (no doubled paragraph).

**Acceptance:** No verbatim-duplicated intro paragraphs in generated instructions.

---

# GROUP B — Correctness bugs (SONNET for clear fixes, OPUS where judgment needed)

## T-S8 [SONNET] Fix the Day 1 logprobs API-endpoint mismatch
**Scope:** `1.2-logprobs/section2_solution.py`.

**Context:** Prose (`:~43`), the function docstring, and the linked reference all say the
**completions** API, but the code calls `chat.completions.create`. The link points at the
wrong endpoint. Also the reference list-comprehension guards `if choice.logprobs` *after*
already dereferencing `choice.logprobs.content`, so it raises before the guard if logprobs
is None.

**Checklist:**
- [ ] Make prose, docstring, and the reference link consistent with the code (chat.completions), OR switch the code to the completions endpoint — pick whichever the exercise intends (chat.completions is the safer default). 
- [ ] Reorder the guard so `choice.logprobs` is checked before `.content` is accessed.
- [ ] Confirm the test still passes (note: it makes a live API call — see T-O1 flake note).

**Acceptance:** Endpoint naming/link matches the code; guard is safe.

## T-O8 [OPUS] Correctness bugs requiring judgment
**Scope:** `3.3-guardrails`, `5.1-adversarial-vision`, `5.4-watermarking`, `1.3-instruction-hierarchies`.

**Context / per-file:**
- **`3.3-guardrails/section3_solution.py:~77-89`** — the setup download loop pulls a
  **nonexistent model id `google/gemma-4-E4B-it`** (there is no Gemma 4) plus unused
  Llama-3-8B / DeepSeek / gemma-2 tokenizers, and never downloads the `Qwen/Qwen3-4B` the
  section actually uses (via `day3_setup.py`). As written this hard-fails the setup cell that
  gates the whole guardrails section. Fix: download exactly the models the section uses
  (Qwen3-4B + whatever `day3_setup` and the exercises reference), or delete the block since
  `day3_setup` already loads the model on import. Verify against `day3-inference/day3_setup.py`.
- **`5.1-adversarial-vision/section1_solution.py`** — (a) Ex 1.2b docstring says "untargeted"
  but the implementation is fully targeted (CE toward `target_class_id`, drives to "daisy");
  reconcile prose and code (probably make it genuinely untargeted, or relabel as targeted).
  (b) `torch.clamp(perturbed_inputs, 0, 1)` (`:~378`) assumes [0,1] pixels but the ViT
  processor normalizes to ~[-1,1]; clamping to [0,1] silently discards half the valid range.
  Fix to the true normalized min/max or add a comment justifying it. (c) The network image
  fetch + exact `assert class_idx == 285` is brittle offline / across preprocessing versions —
  make resilient (try/except + fallback, soften the exact-index assert).
- **`5.4-watermarking/section4_solution.py`** — (a) `np.fft.fft2` on an H×W×3 array transforms
  over W-and-channel, not the two spatial axes (`:~297`); use `axes=(0,1)` or convert to
  grayscale. (b) `20*np.log(...)` labeled "dB" uses natural log; dB uses `log10`. (c) The
  `output[0][1]` batch-index assumes classifier-free-guidance batch≥2 (`:~174`); guard or
  handle batch size 1. (d) **Framing**: this is a 2%-band attenuation with no payload and no
  blind detector — it is a toy perturbation, not a recoverable watermark, yet is called
  "watermarking" throughout. Reframe honestly (see also T-O10 for the GCR angle) and add a
  short contrast with real methods (Tree-Ring, Stable Signature, SynthID-Text). NOTE: parts of
  this overlap with new-content Task N6/N8 — do the correctness/labeling fix here; leave any
  new SynthID-Text exercise to the research group.
- **`1.3-instruction-hierarchies/section3_solution.py:~39`** — "Anthropic dropped support for
  [prefill] starting with Claude Opus 4.6" is stated as settled fact. Prefill returns a 400
  across the current Claude family; version-pinned vendor claims age badly. Soften to e.g.
  "newer Claude models reject assistant-turn prefill" and/or cite the API docs. If unsure of
  the current exact behavior, consult the `claude-api` skill/docs rather than guessing.

**Checklist:**
- [ ] Fix each item above; where behavior is model-dependent, prefer robustness + a comment.
- [ ] Re-run each solution file end-to-end (GPU files may need the remote box; if you can't
      run them, reason carefully and note that a human must execute-verify).

**Acceptance:** No known-wrong math/labels; setup cells reference real models; claims are defensible.

---

# GROUP C — Test infrastructure (OPUS; large, needs judgment on what to assert)

## T-O1 [OPUS] Restore test coverage across the course
**Scope:** Day 4 (all 3 sections), Day 5 (all 4 sections), `3.1`, `3.5`; fix broken
extraction in `2.1` and `2.3`.

**Context:** Tests have largely collapsed post-refactor:
- **Days 4 & 5 have ZERO `@report` tests** (all `*_test.py` are 0 bytes) despite most
  exercises being testable. Day 4 uses bare top-level `assert`s; Day 5 uses bare asserts,
  several **tautological** (`loss_history[-1] <= loss_history[0]` is guaranteed by
  construction and tests nothing).
- **`3.1` and `3.5`** have empty test files but testable exercises (3.5: detected dim ≈768;
  weight cosine-similarity after alignment).
- **`2.1` broken extraction**: `test_targeted_attack_succeeds` / `test_attack_is_specific`
  reference `rag_query`, `SMALL_MODEL`, `openrouter_client` that live in un-extracted
  top-level setup → the generated `section1_test.py` NameErrors on collection. Fix by wrapping
  that setup in `if "TEST_FIXTURE":` so it's emitted into the test file.
- **`2.3` broken extraction**: extracted test imports `day2_utils` but its boilerplate only
  adds the section dir to `sys.path`, not `day2-agents` → import fails. Fix the path handling
  so the extracted test can import shared utils (may require T-S4's canonical snippet or a
  `TEST_FIXTURE`-wrapped path insert).

**Checklist:**
- [ ] For each testable exercise lacking a test, add an `@report def test_<name>(solution):`
      with informative assert messages (input/expected/actual) and ≥1 edge case, called at top
      level. Follow the CLAUDE.md test conventions. Prioritize: Day 4 (2.6 trigger-fire rate;
      2.1/2.2 dataset construction; 3.3 fine-tuned refusal-rate < base), Day 5 (target_loss
      finite/positive & correct slice length; gradient-replacement shapes & forbidden-token
      exclusion; adversarial-example success), 3.5 (detected dim ≈768; `compare_weights`
      cosine-sim above threshold).
- [ ] Replace tautological asserts with meaningful ones (e.g., final loss below an absolute
      threshold, or strictly-decreasing-with-early-stop, not `final <= initial`).
- [ ] Convert Day 4's bare top-level asserts into `@report` test functions.
- [ ] Fix `2.1` (TEST_FIXTURE the RAG setup) and `2.3` (path for `day2_utils`) so
      `pytest <section>_test.py` actually collects and runs.
- [ ] For live-model / network tests (2.1, 3.1, 3.3, 1.2, 1.3, 5.x), add a one-line comment
      that they are non-deterministic / require network+GPU, so instructors expect occasional
      flakes. Use `do_sample=False` where a deterministic result is expected.
- [ ] Build each file and run the generated `*_test.py` where the environment allows; note
      which require the GPU box for a human to verify.

**Acceptance:** No empty `*_test.py` for a section that has testable exercises; `2.1`/`2.3`
extracted tests collect without NameError/ImportError; no tautological asserts.

---

# GROUP D — Structural / de-duplication (OPUS; deletion decisions)

## T-O5 [OPUS] Migrate remaining shared assets, then DELETE the legacy `dayN-*` folders entirely
**Scope:** `day1-intro/`, `day2-agents/`, `day3-inference/`, `day4-training-and-data/`,
`day5-training-and-data/`, `day7-governance/`. (Day 6 legacy cruft is handled separately in
T-O3; do not double-handle it here.)

**Goal (per the maintainers):** the `dayN-*` legacy folders should be **removed completely**,
not just have their `dayN_solution.py` deleted. This is a bigger job than a plain delete
because some `dayN-*` folders still hold *shared assets that the `X.Y` sections import at
runtime* — those must be migrated into the `X.Y` layout and all imports/paths updated FIRST,
then the folders deleted.

**Context:** Every day has a pre-refactor monolith (`dayN_solution.py` + generated `.md`/`.py`)
plus supporting files, living alongside the new `X.Y` folders. The monoliths were last touched
~April 2026; the X.Y refactor is the July 2026 HEAD, and they've **diverged, not just split**
(Day 3 legacy = 2,618 lines vs the X.Y sections' 5,322 — content grew in the X.Y set). Known
shared assets still referenced from `dayN-*`:
- `day3-inference/day3_setup.py` — imported by `3.3-guardrails` (model, `generate`, helpers).
- `day3-inference/setup/` (SSH screenshots) and `day3-inference/gpt2-small.png` — referenced by
  Day 3 instruction prose.
- Possible `day2-agents/day2_utils*` imported by `2.3`/`2.4`.
- Any images referenced with relative paths.
Also note `day5-training-and-data/` is misnamed (it holds vision content) and `day0-setup/`
duplicates the top-level README/PREREQUISITES (see other tasks) — but `day0-setup/` is NOT in
scope here (it's the setup folder, not a legacy day monolith).

**This task has two stages. Do NOT delete anything until Stage 1 passes.**

**Stage 1 — Assert clean migration (a dedicated verification agent):**
- [ ] For each `dayN-*` folder, enumerate every file and classify it: (a) superseded content
      now in `X.Y`, (b) a shared asset still imported/referenced by an `X.Y` file, or
      (c) content that exists ONLY here and has no `X.Y` equivalent.
- [ ] For (a): spot-diff a few sections to confirm the `X.Y` version is a superset (not missing
      content). Produce a short per-day coverage assertion ("all of dayN's content is present in
      X.Y sections A/B/C, confirmed by …").
- [ ] For (b): list each shared asset and every `X.Y` file + line that imports/references it.
- [ ] For (c): **STOP and flag to a human** — do not delete anything unique that isn't covered.
- [ ] Deliver a written go/no-go report per day. Only days with a clean "everything is either
      covered or a migratable shared asset" verdict proceed to Stage 2.

**Stage 2 — Migrate shared assets, then delete (only for green-lit days):**
- [ ] Move each shared asset into the `X.Y` layout. Options, pick per asset: co-locate in the
      importing section folder (e.g. `3.3-guardrails/day3_setup.py`), or create a small shared
      dir (e.g. a top-level `common/` or `assets/`) if used by multiple sections. Prefer the
      simplest that keeps imports clean.
- [ ] Update every import and relative path in the `X.Y` files that referenced the old location
      (grep first: `grep -rn "day[1-7]-[a-z]" */section*_solution.py`), and update the
      `sys.path` snippet accordingly (coordinate with T-S4).
- [ ] Rebuild and, where the environment allows, run each affected `X.Y` section + its extracted
      test to confirm imports resolve and images render. GPU-only files: reason carefully and
      mark for human execute-verification.
- [ ] Delete the entire `dayN-*` folder once its assets are migrated and nothing references it.
      Final grep must return zero references to the folder anywhere in the repo (solution files,
      READMEs, CLAUDE.md, this file).
- [ ] Update `CONTENT_REVIEW_TODO.md` and `README.md` if they mention the deleted folders.

**Acceptance:** No `dayN-*` legacy folder remains (except any explicitly human-exempted); every
former shared asset lives in the `X.Y` layout with updated imports; all `X.Y` files build and
their tests import cleanly; a repo-wide grep for `day[1-7]-` folder references returns nothing
stale.

**Dependency note:** run this BEFORE T-S2's path fixes settle, or coordinate — T-S2 fixes
participant-facing paths and this task changes where shared assets live; they touch overlapping
lines. Recommended: do T-O5 Stage 1 (assert) first, then T-S2 + T-O5 Stage 2 together.

## T-O3 [OPUS] Untangle Day 6 structure
**Scope:** `day6-infrastructure/`.

**Context:** Day 6 content is strong and GCR-relevant (RAND SL1–SL5 weight-security levels,
a technically accurate CVE-2025-23266 container-escape walkthrough, and a RowHammer→PTE→root
"GPUBreach" simulator) and `day6_final_solution.py` runs clean end-to-end. But the folder is
a mess:
- A **nested duplicate** `day6-infrastructure/day6-infrastructure/` containing two OLDER
  standalone solutions (`day6_gpu_solution.py` — superseded, has a `<!-- FIXME -->` and
  "W2D4" refs; `day6_gpubreach_solution.py` — later inlined into `day6_final_solution.py`),
  plus duplicate `gpubreach_sim/` and `module1/`.
- **Two stale/mis-scoped READMEs**: the top-level `day6-infrastructure/README.md` is a
  project-wide brainstorm; the ACTUAL Day-6 README ("[Nitzan] Day 6: Infrastructure") is
  buried in the nested dir.
- `day6_final_solution.py` has **duplicated section numbering**: a top-level `1️⃣/2️⃣/3️⃣`
  set and then §3 restarts with its own `1️⃣/2️⃣/3️⃣` (Phase 1/2/3), which breaks the TOC.
- Stale per-folder `aisb_utils/`, `build-instructions.sh`, `requirements.txt` copies (see T-O6).

**Checklist:**
- [ ] Confirm `day6_final_solution.py` is canonical (it is per review); delete the nested
      `day6-infrastructure/day6-infrastructure/` after verifying nothing unique lives only there.
- [ ] Promote the real "[Nitzan] Day 6" README to `day6-infrastructure/README.md`; delete the
      stray brainstorm README.
- [ ] Fix section numbering in `day6_final_solution.py`: nest the GPUBreach phases under §3
      (e.g. `## GPUBreach lab` with `### Exercise 3.x`), so there's a single header hierarchy.
      (Combine with T-S1: drop emoji digits entirely.)
- [ ] Decide the final home: either keep `day6-infrastructure/` as-is or split into `6.x-*`
      folders to match the rest of the course (flag this decision to a human — it affects
      answer-file paths and the README).
- [ ] Rebuild; confirm the TOC is single-hierarchy and the lab still runs (`smoke_test.py` passes).

**Acceptance:** No nested duplicate; one correct README; single header hierarchy; lab runs.

## T-O4 [OPUS] Resolve Day 7 triplication and fix 7.2 structure
**Scope:** `7.1-atlas-killchain/`, `7.2-adversary-matrix/`, `day7-governance/day7_solution.py`.

**Context:** The same Day 7 content exists in three places: `7.1` (title block + Section 1,
but it also carries Section 2's stranded learning objectives), `7.2` (Section 2 + Summary,
but NO title block / `<!-- toc -->` / objectives of its own, and a dead `sys.path` shim), and
`day7-governance/day7_solution.py` (the full combined day — a superset duplicate). Also the
ATLAS technique IDs in 7.1's reference kill-chain table look written from memory (e.g.
Evade-ML-Model mapped to Initial Access; ATT&CK tactic names used where ATLAS differs), and
7.2's Day 1–6 cross-references (the "which day covered which tactic" hint + kill-chain) are at
high risk of being stale post-refactor.

**Checklist:**
- [ ] Pick the canonical layout (recommend: keep `7.1` + `7.2`, delete
      `day7-governance/day7_solution.py` per T-O5). Move Section 2's stranded objectives out of
      `7.1` into `7.2`.
- [ ] Give `7.2` a proper title block + `<!-- toc -->` + its own learning-objectives block;
      remove the dead `sys.path` shim.
- [ ] Verify every `AML.Txxxx` ID and tactic name in `7.1`'s table against the live MITRE
      ATLAS matrix; fix mismatches; add a note that ATLAS tactics ≠ ATT&CK tactics.
- [ ] Verify `7.2`'s Day 1–6 references (day names + exercise numbers) against the current
      `X.Y` content; fix stale labels.
- [ ] Rebuild both; confirm each generates a well-formed standalone instructions page.

**Acceptance:** One Day-7 source; both sections well-formed; ATLAS IDs and cross-refs verified.

## T-O6 [OPUS] De-duplicate `aisb_utils` and harden the build system
**Scope:** `aisb_utils/` (canonical, top-level), `day0-setup/aisb_utils/`,
`day6-infrastructure/aisb_utils/`, `build-instructions.sh`, `aisb_utils/build_instructions.py`.

**Context:** There are stale COPIES of `aisb_utils/` in `day0-setup/` and
`day6-infrastructure/`. Both **lack `TEST_FIXTURE` support** (they're ~2.5KB smaller and miss
`CollectTestFixtures` + the fixture branches) and day6's also lacks `env.py`. `TEST_FIXTURE`
is used in real content (`3.3`, `3.4`, `day3-inference`), so building any such file via those
folders' local `build-instructions.sh` would silently mis-render `if "TEST_FIXTURE":` as
literal code. Additionally the build has silent-failure modes: (a) it **no-ops on unchanged
files** with zero output (mtime skip — looks like breakage); (b) it **swallows `SyntaxError`
and exits 0**, so a malformed solution "succeeds" without rebuilding; (c) the venv-bootstrap
fallback calls bare `python` which isn't on PATH here (`python3` is).

**Checklist:**
- [ ] Delete the per-folder `aisb_utils/` copies + local `build-instructions.sh` +
      `requirements.txt` in `day0-setup/` and `day6-infrastructure/`; standardize on the single
      top-level `aisb_utils/` and `build-instructions.sh`. Verify nothing imports the local copies.
- [ ] Build system: print an explicit "up to date, skipping (use --force)" line instead of a
      silent no-op.
- [ ] Build system: on `SyntaxError`/`UnicodeDecodeError`, print the error AND exit non-zero
      (so `--watch` and CI surface it).
- [ ] Build wrapper: use `python3` (not `python`) for venv bootstrap.
- [ ] Sanity-check: build a `TEST_FIXTURE`-using file (e.g. `3.3`) and confirm the fixture is
      correctly unwrapped in instructions/test/reference.

**Acceptance:** One `aisb_utils`; TEST_FIXTURE renders correctly everywhere; build fails loudly
on syntax errors and reports skips; bootstrap works on a clean machine.

## T-O7 [OPUS] Fix non-runnable scaffolds and answer-leaking scaffolds
**Scope:** `3.5-weight-extraction`, `4.3-safety-finetuning`, `5.4-watermarking`.

**Context:** CLAUDE.md requires scaffolds to be runnable and to not hand over the answer.
Violations:
- **`3.5` non-runnable**: shared vars (`n_queries`, `max_prompt_length`, `vocab_size`,
  `W_extracted`) are defined only inside the `if "SOLUTION":` branch, so the participant-facing
  `else: pass` scaffold leaves them undefined and downstream top-level code NameErrors. Also
  `detected_h = 768` is hardcoded instead of derived from the gap analysis the code sets up —
  make the exercise actually compute it. (Convert Ex 5.1/5.2 into functions with returns; ties
  into T-O1's tests.)
- **`4.3` over-hand-holding**: the prose gives the full literal LoRA config + `TrainingArguments`
  (`:~210-231`), nearly handing over a 4/5 exercise; and `per_device_train_batch_size=4 or 1 if
  you don't have enough GPU memory` is written as literal pseudo-code that won't run if pasted.
  Convert config to hints/partial, and rewrite the batch-size line as guidance not a value.
- **`5.4` leak**: Ex 4.1's TODO literally contains the answer `generator=...` line; Ex 4.4 has
  no scaffold at all (full solution shown). Give real scaffolds or demote 4.4 from "Exercise".

**Checklist:**
- [ ] `3.5`: move shared vars out of the SOLUTION-only branch; make Ex 5.1/5.2 functions that
      return the detected dim / extracted weights; derive `detected_h` from the singular-value
      gaps rather than hardcoding.
- [ ] `4.3`: reduce config leakage to hints (or lower the rating to match hand-holding); fix the
      pseudo-code batch-size line.
- [ ] `5.4`: add proper `if "SOLUTION"/else` scaffolds; demote non-exercises to plain subsections.
- [ ] Confirm each participant-facing scaffold runs without NameError before edits.

**Acceptance:** Every `else` scaffold runs; no scaffold contains the reference answer verbatim.

---

# GROUP E — Info-hazard / dual-use (OPUS; highest-priority GCR-specific gap)

## T-O2 [OPUS] Add dual-use / info-hazard framing and safety hygiene
**Scope:** top-level `README.md`; intros of Days 3, 4, 5; `4.3-safety-finetuning` (cleanup);
spot edits in `3.2`, `3.3`, `5.2`.

**Context:** A full-corpus grep for `info-hazard | dual-use | responsible disclosure |
harmless proxy` returns **zero hits**. For a course whose thesis is misuse-uplift and
proliferation as catastrophic risk, there is no framing on why harmful capabilities are being
elicited, how to handle outputs/artifacts, or responsible-disclosure norms. Concretely:
- **`4.3` abliteration is the sharp edge**: trains a LoRA on harmful responses on real
  Llama-2-7B-Chat, prints the model answering "how to build a bomb", and leaves participants
  holding a **safety-stripped adapter on the shared GPU box with no cleanup and no framing** —
  exactly the proliferation vector Day 7's kill chain names as the catastrophic payoff.
- `5.2` GCG optimizes toward a literal bomb-making target string; `3.3` uses a dataset that
  includes "instructions for synthesising a neurotoxin" as the success condition; `3.2` sends
  participants to live production jailbreak arenas with no terms-of-use note.
- Credit: `3.4` (distillation) uses "France" as a harmless proxy and says so; `4.2` uses
  "James Bond"→positive. The course knows how to do this — it's just inconsistent and unstated.

**Checklist:**
- [ ] Write a short **dual-use policy** box for the top-level README: why the course elicits
      real attack techniques, the principle of harmless proxies, that generated harmful text and
      any safety-stripped/backdoored artifacts must NOT be shared or exfiltrated from the lab
      environment, and responsible-disclosure norms. Keep it crisp (a professional audience).
- [ ] Add a one-paragraph dual-use callout at the top of Days 3, 4, 5 referencing the policy.
- [ ] **`4.3` mandatory cleanup step**: at the end of the exercise, add code + prose that
      deletes the saved abliterated adapter and clears it from the shared box, and a note that
      this artifact is the literal proliferation risk the course studies.
- [ ] Where an exercise elicits genuinely dangerous content (`3.3`, `5.2`), add a one-line note
      that the target strings are proxies and outputs should not be retained/shared.
- [ ] Consider whether `4.3` should use a less capable / less sensitive base model or a more
      clearly proxied "refusal" behavior; flag the tradeoff to a human (pedagogical value of a
      real model vs. artifact risk).

**Acceptance:** A stated dual-use policy exists and is referenced from every attack-heavy day;
`4.3` ends by deleting its safety-stripped artifact; harmful-content exercises carry a proxy note.

---

# GROUP F — Week-level design (OPUS; pedagogy)

## T-O9 [OPUS] Rebalance the week and fix cross-day friction
**Scope:** Day 1 (add depth) & Day 2 (trim/de-risk); Day 3 setup; Day 0; shared generation util.

**Context:** Difficulty arc is `easy → spike → heavy → heavy → even → very-heavy → cooldown`.
Day 1 is thin (pairs finish early); Day 2 is an overstuffed cliff (5 sections, Docker/
ControlArena-gated, ~7-8h, no API-only fallback); the API→GPU seam lands mid-week at Day 3
behind an SSH-setup wall (a pair blocked there loses the rest of the week); Day 4's HF-token
prerequisite is an unmet `<!-- FIXME -->`; the generation loop is reimplemented 3× (3.1 Ex1.1,
3.3 Ex3.0, again in 4/5) and Day 3 uses two conflicting cache dirs (`/tmp/cache-tokenizer` vs
`/workspace/model-cache`) causing multi-GB re-downloads.

**Checklist:**
- [ ] Rebalance Day 1↔Day 2: promote Day 1's optional prefill/instruction-hierarchy content to
      full status, and/or make Day 2's `2.5` clearly optional and trim `2.4`. Add "read ahead
      while loops run" notes to compute-bound exercises.
- [ ] Provide an API-only fallback path (or clear "if Docker/ControlArena is down, do X") for the
      Day 2 control sections so a broken framework doesn't cascade-fail sections 3-5.
- [ ] Move the Day 3 Remote-SSH setup into a **Day 0 dry-run** so pairs validate SSH before Day 3.
- [ ] Resolve the Day 4 HF-token FIXME on Day 0: document the token, add gated Llama-2 access to
      setup, update `.env.example`.
- [ ] Unify the two Day 3 cache dirs to one value so models download once.
- [ ] Extract ONE canonical `generate()`/model-loader into a shared util imported by Days 3-5;
      keep `3.3` Ex 3.0 as an explicit optional "build it yourself" that then uses the shared one.
- [ ] Standardize answer-file + directory naming (ties into T-S2).

**Acceptance:** No day is >6h by its own estimates; a single SSH/token failure doesn't nuke the
week; generation code exists once; models download once.

## T-O10 [OPUS] Make the GCR through-line explicit on Days 3-6
**Scope:** day intros/summaries of Days 3, 4, 5, 6 (esp. `3.5`, `4.x`, all of Day 5).

**Context:** Days 2 and 7 carry the catastrophic-risk thesis well, but Days 3-5 read as
generic ML-security/appsec (adversarial ViT on cat photos, diffusion "watermarking") with the
loss-of-control / misuse-uplift / proliferation framing left implicit until Day 7 collects it.
Weight extraction (3.5) is framed as IP theft, not proliferation → the exact Day-7 kill-chain
step. There's no consistent per-day "why this matters for catastrophic risk" element.

**Checklist:**
- [ ] Add a short, consistent "Why this matters for catastrophic risk" callout to each day
      (3-6), tying the day's techniques to one of: loss of control, misuse uplift, proliferation.
- [ ] Reframe `3.5` (weight extraction) and `4.3` (abliteration) explicitly as the
      proliferation steps Day 7's kill chain depends on (forward-reference Day 7).
- [ ] Give Day 5 a day-level GCR framing (adversarial robustness of frontier multimodal systems /
      jailbreak-as-uplift), so it doesn't read as ML-security 101.

**Acceptance:** Every day states its catastrophic-risk relevance; Day 7's kill chain is
foreshadowed by the days that build its steps.

---

# GROUP G — New content / research additions (OPUS; each is a design task)

Each task below is a candidate new exercise or reading, grounded in verified 2024-2026
research. All citations were checked against primary sources during review. **Before building,
confirm the item isn't already partially present, and prefer a small hands-on lab over a
reading where the audience benefits.** Scope each to fit the existing day's time budget (see
T-O9) — these are proposals; a human should greenlight which to build.

## T-N1 [OPUS] Day 3 capstone: the adaptive-attack arms race
**Why:** Day 3 presents guardrails (keyword filter, LLM-judge, output classifier, **linear
activation probes**) as if robust. The frontier lesson for a security audience is that the
*composed* stack falls to adaptive attacks. This is the single highest-value research add.
**Build:** a capstone where participants attack the probe-based monitor they built earlier.
**Citations:** Circuit Breakers / RepE (arXiv:2406.04313, Gray Swan — already the Day 3 arena)
+ its break (2407.15902); Obfuscated Activations (2412.09565, directly defeats the linear
probe); STACK (2506.24068, breaks full input→model→output pipelines); Simple Adaptive Attacks
(2404.02151, the "always evaluate adaptively" principle). Also refresh the existing
Constitutional Classifiers cite with the 2025 red-team result (2501.18837).
**Checklist:**
- [ ] Reading: circuit breakers + break; obfuscated activations; simple adaptive attacks.
- [ ] Lab: obfuscation-style attack against the Day-3 probe monitor; measure ASR before/after.
- [ ] Add a caveat to the existing probe/classifier exercises that they are adaptively bypassable.

## T-N2 [OPUS] Day 4/5: the open-weight durability pillar (unlearning ↔ recovery)
**Why:** Day 4 teaches safety *removal* (abliteration, LoRA) but never the safeguards being
attacked — the central GCR question of whether open weights can be safely released.
**Citations:** Refusal Is Mediated by a Single Direction (2406.11717 — **the paper abliteration
is built on; verify Day 4 cites it, likely a gap**); Qi et al. fine-tuning breaks safety in
~10 examples (2310.03693); WMDP + RMU unlearning (2403.03218); unlearning fragility /
relearning (2409.18025, 2502.05374); TAR tamper-resistance (2408.00761) + durability critique
(2412.07097).
**Checklist:**
- [ ] Add the Arditi et al. citation to the `4.3` abliteration section.
- [ ] New section (Day 4 or 5): unlearn with RMU on a small model (WMDP-proxy drop vs MMLU
      retention), then recover via relearning/quantization/jailbreak — a visceral "unlearning
      isn't robust" lab. Coordinate with T-O2 (info-hazard) since WMDP touches hazardous knowledge — use the benchmark's proxy framing, don't elicit real hazardous content.

## T-N3 [OPUS] New module: dangerous-capability evaluation (bridges Day 2 → Day 7)
**Why:** The course jumps from *control* (Day 2) to *governance* (Day 7) with no unit on how
you actually MEASURE dangerous capability — the trigger for every safety framework. Biggest
thematic gap.
**Citations:** DeepMind Evaluating Frontier Models for Dangerous Capabilities (2403.13793,
open Apache-2.0 suite, Inspect-compatible); Cybench (2408.08926, a CTF harness — ideal for
this audience); CyberSecEval 3 (2408.01605); WMDP (2403.03218). Plus the adversary's view:
sandbagging (2406.07358, 2412.01784), evaluation awareness (Apollo 2025), SAD situational-
awareness benchmark (2407.04694).
**Checklist:**
- [ ] Reading: DC-eval methodology + one cyber-uplift benchmark.
- [ ] Lab: run a DC eval (self-proliferation or a Cybench task) via Inspect (reuses Day 2 muscle).
- [ ] Lab/reading: build a password-locked (sandbagging) model on a toy task and try to elicit
      the hidden capability — "evals can underestimate capability".

## T-N4 [OPUS] Day 2: frontier misalignment / scheming as measured phenomena
**Why:** Misalignment currently appears only implicitly as AI Control. The empirical scheming
literature is the core GCR story.
**Citations (headline first):** Natural Emergent Misalignment from Reward Hacking (Anthropic,
2511.18397, Nov 2025 — production RL reward-hacking spontaneously generalizes to sabotage/
alignment-faking/monitor-disruption; includes "inoculation prompting" mitigation). Also: Apollo
in-context scheming (2412.04984); OpenAI×Apollo anti-scheming + situational-awareness confound
(2509.15541); Alignment Faking (2412.14093); Agentic Misalignment / insider-threat framing
(2510.05179); Emergent Misalignment narrow-finetune→broad (2502.17424) + persona features
(2506.19823, SAE mechanism).
**Checklist:**
- [ ] Reading + discussion on-ramp to Day 2 (before the control labs).
- [ ] Lab (headline): reproduce a toy reward-hack→misgeneralization pipeline; test inoculation
      prompting as a mitigation. This is the strongest new lab candidate in the whole course.

## T-N5 [OPUS] Day 2: CoT monitorability ↔ faithfulness ↔ steganography
**Why:** Day 2 does CoT monitoring but treats it as reliable. It isn't, and it's directly
attackable.
**Citations:** CoT Monitorability position paper (2507.11473, 41-author cross-lab); Anthropic
"Reasoning Models Don't Always Say What They Think" (2505.05410, ~25% verbalization); steganographic
CoT under process supervision (2506.01926 — penalizing reasoning terms trains models to ENCODE
reasoning); untrusted-monitor collusion (2402.07510).
**Checklist:**
- [ ] Lab: hint-insertion faithfulness test — measure how often a model verbalizes the cue that
      drove its answer.
- [ ] Reading: monitorability paper + steganographic CoT, as the adversarial counterpoint to the
      Day-2 monitoring content.

## T-N6 [OPUS] Day 6: defensive infrastructure (fix the attack-only asymmetry) + governance backbone
**Why:** Day 6 teaches weight-security ATTACKS thoroughly (GPUBreach, RowHammer, CVE) but has
almost NO defensive technical content.
**Citations:** Confidential computing / TEEs for weights (H100/H200→Blackwell; arXiv:2409.03992,
2502.11347) — RAND calls this consensus-required for SL4/SL5; SL5 Standard (IST, 2605.08449,
2026 — refresh the Nov-2025 draft already in the repo); verifiable inference / model-substitution
auditing (2504.04715); RowHammer-as-weight-integrity (DeepHammer USENIX'20; 1-bit-flip Trojan
USENIX'25). For Day 7 governance backbone: flexHEG hardware-enabled guarantees (2506.15093/15100),
compute-monitoring verification (Shavit "Catch a Chinchilla" 2303.11341; newer empirical work).
**Checklist:**
- [ ] Day 6 defensive lab: parse/validate an NVIDIA attestation report; reason about which
      GPUBreach steps a TEE stops (pairs against the existing sim).
- [ ] Refresh the SL5 discussion to cite the formal 2026 standard.
- [ ] Reframe the RowHammer sim to also cover weight-integrity sabotage.
- [ ] Day 7: add flexHEG + compute-monitoring as the hardware/verification backbone for the
      governance discussion.

## T-N7 [OPUS] CBRN / cyber uplift (Day 7 or the DC-eval module) — the real GCR pathways
**Why:** The archetypal mass-casualty (bio) and most-operational (cyber) catastrophic pathways,
with real 2025 evidence.
**Citations:** Bio-uplift human trials — Anthropic Opus 4 system card (2.53× uplift → first real
ASL-3 trigger), Virology Capabilities Test (2504.16137, o3 beats 94% of expert virologists),
RAND RR-A2977-2 (2024 NULL result — the methodology-matters contrast). Autonomous cyber-offense —
DARPA AIxCC finals (18 real vulns found), Google Big Sleep (pre-empted a live SQLite 0-day),
XBOW (#1 on HackerOne US). Forecasting — METR time horizons + RE-Bench (2411.15114).
**Checklist:**
- [ ] Reading + discussion: the bio-uplift trio (critique the three methodologies; design a
      better uplift trial). Handle as reading/discussion, NOT a hands-on elicitation (info-hazard).
- [ ] Reading/lab: autonomous cyber-offense case studies; optionally run an open AIxCC CRS or an
      agent against a seeded-vuln target and measure find/patch rate (fits this audience well).

## T-N8 [OPUS] Day 3/5: expand the jailbreak taxonomy + add measurement rigor
**Why:** Day 3's taxonomy is single-turn only; Day 5 teaches white-box GCG but not the
automated black-box family; and there's no rigorous way to SCORE any attack lab.
**Citations:** Multi-turn Crescendo (2404.01833); many-shot jailbreaking (Anthropic 2024,
long-context power law); Best-of-N (2412.03556, cross-modal, compute-buys-jailbreaks);
PAIR (2310.08419) / TAP (2312.02119) / AutoDAN-Turbo (2410.05295) automated red-teaming;
StrongREJECT (2402.10260) + HarmBench (2402.04249) evaluation; Jailbreak Tax (2504.10694 —
successful jailbreaks often yield degraded output; a GCR uplift-calibration lesson).
**Checklist:**
- [ ] Day 3: add a multi-turn axis (Crescendo) and long-context axis (many-shot) to the taxonomy;
      small labs (both are easy to code).
- [ ] Day 5: add PAIR (tractable) as the automated black-box complement to GCG.
- [ ] Add StrongREJECT as the scoring harness for ALL attack labs; discuss the jailbreak-tax /
      "does ASR overstate real uplift" question.

## T-N9 [OPUS] Day 4: poisoning at scale
**Why:** Complements Day 4's fine-tuning backdoor with the pretraining threat model and the
defining 2025 result.
**Citations:** Near-constant poison count (Anthropic×UK AISI, 2510.07192 — ~250 docs backdoor a
model regardless of size; note the caveat it's a DoS/gibberish backdoor at that scale); Sleeper
Agents (2401.05566, backdoors survive safety training); Persistent Pre-Training Poisoning
(2410.13722).
**Checklist:**
- [ ] Reading: the poisoning trio.
- [ ] Optional lab: recreate a mini backdoor on a tiny model with a fixed poison count; show it
      doesn't scale with dataset size. Contrast pretraining vs fine-tuning injection with the
      existing `4.2` backdoor exercise.

## T-N10 [OPUS] (Optional) Interpretability-for-safety capstone beyond probes/ROME
**Why:** Interpretability currently stops at linear probes (Day 3) and ROME (Day 4).
**Citations:** Auditing LMs for Hidden Objectives (2503.10965, the blinded "auditing game");
Scaling Monosemanticity (transformer-circuits.pub, 2024, SAEs/dictionary learning).
**Checklist:**
- [ ] Optional Day 3/4 capstone framed as a safety-auditing game using SAE features to uncover a
      hidden objective. Lower priority than N1-N9.

---

# Suggested execution order

1. **First, mechanical & safety (parallelizable, low-risk):** T-S1, T-S2, T-S3, T-S4, T-S5,
   T-S6, T-S7, T-S8 (Sonnet). These clear noise so later reviewers see real issues.
2. **Then structural decisions (Opus, some are prerequisites):** T-O5 Stage 1 (assert legacy
   `dayN-*` coverage) early — it gates the folder removal; then T-O6 (dedup utils/build) which
   unblocks everything; T-O3 (Day 6), T-O4 (Day 7). Run T-O5 Stage 2 (migrate assets + delete
   folders) together with T-S2 since they touch overlapping path/import lines.
3. **Correctness & tests (Opus):** T-O8, T-O7, T-O1.
4. **The GCR-critical policy (Opus):** T-O2 — do this early; it's cheap and important.
5. **Pedagogy (Opus):** T-O9, T-O10.
6. **New content (Opus, greenlight first):** T-N1 … T-N10, prioritizing N1, N3, N4, N6.

## Verification snippets

```bash
# find remaining emoji conventions after T-S1
grep -rlP "[\x{1F534}\x{26AA}\x{1F535}]|[1-7]\x{FE0F}\x{20E3}" */section*_solution.py

# rebuild one file and check it parses
./build-instructions.sh <path>/sectionN_solution.py --force
python3 -c "import ast; ast.parse(open('<path>/sectionN_solution.py').read()); print('ok')"

# run an extracted test (where env allows)
python3 -m pytest <path>/sectionN_test.py -q
```
