---
name: exercise-walkthrough
description: Simulate a first-time student working through a bootcamp exercise file step-by-step — reading progressively (without peeking ahead), pasting code into an answers file, executing cells in a live Jupyter kernel, attempting TODOs, and producing a structured review of bugs, ambiguities, and missing context.
disable-model-invocation: true
---

# Exercise Walkthrough Review

Simulate a first-time participant working through a day's exercise file and capture a detailed review. The value of this skill is that an author can't easily see the problems a learner hits — ambiguous wording, missing prerequisites, code that doesn't run, tests that fail for surprising reasons. To surface those, you have to approach the material the way a student would: in order, with only the information they'd have at each point.

## Why progressive reading is the whole point

If you read the full `*_instructions.md` upfront, the review is worthless — you already know what's coming, so you can't tell when something is ambiguous or underspecified. A real student reads top-to-bottom, one exercise at a time, and can't see the next section until they get there.

Follow these constraints strictly:

- **Never open `*_solution.py` under any circumstances during the walkthrough.** The solution file is the *authored source* — it contains the reference implementation for every TODO, hidden inside `if "SOLUTION":` blocks that the build system strips before producing the student-facing `*_instructions.md`. Students never see it; neither should you. Even a quick glance ("just to orient myself", "just to check where the setup imports come from") contaminates every subsequent attempt, because you'll unconsciously steer toward the reference answer instead of producing what a student would actually write. If you catch yourself about to open it, stop and re-read this paragraph. The instructions file, the kernel, and your own reasoning are the only inputs.
- Do not open `*_test.py` upfront either. The tests get *executed* during the walkthrough (via imports from the code blocks in the instructions), but reading the asserts ahead of time leaks the expected behavior. You may open the file if the user asks you to fix an exercise explicitly.
- Do not open `<details>` / `<summary>` bodies until after you've genuinely attempted the task and written your own answer. Those blocks typically contain the answer or a spoiler hint — looking at them first defeats the exercise.
- Read the instructions file itself in chunks using `Read` with `offset` and `limit`. A reasonable unit is "one exercise and its surrounding prose" — roughly one `### Exercise D.S.E` block at a time.
- If you realize you've accidentally read ahead (e.g., a Read returned more than you intended, or you reflexively opened the solution file), acknowledge it in the review so the user can discount that section.

Between exercises — and more often, whenever something notable happens inside an exercise — stop and write the note to `WALKTHROUGH_REVIEW.md` before moving on. Batching notes at the end loses detail and risks losing them entirely to compaction. See Workflow §3.

### Build an explicit reading plan before you read any content

You need two things to read progressively without spoiling yourself: (1) where each section ends, so you don't over-read into the next exercise, and (2) where each `<details>` body sits, so you don't slurp an answer block into the same `Read` that gets you the prose above it. Resolve both in one upfront planning pass, then walk the plan. Two greps and some arithmetic — cheap.

**Step 1: read the top of the file.** First ~30 lines — title, intro, `## Table of Contents`, `## Content & Learning Objectives`. Students see this when they open the file; you can too.

**Step 2: grep for section headers.** `pattern: "^#{2,4} "`, `output_mode: "content"`, `-n: true`, `path` = the instructions file. Returns header text + line numbers only; no bodies. Each section runs from its header line to one line before the next header.

**Step 3: grep for details tags.** `pattern: "</?(summary|details)>"`, `output_mode: "content"`, `-n: true`, same `path`. Returns line numbers for every `<summary>`, `</summary>`, and `</details>` tag. The `<summary>` text is visible to students (that's what they click) so seeing it is safe; the body between `</summary>` and `</details>` is what you must not read until you've attempted the task.

**Step 4: assemble the plan.** Write it in this exact format, in your scratch notes:

```
Reading plan for 1.1-llm-internals/section1_instructions.md:

1. Setup                                         lines 35–79
   folds: none

2. Exercise 1.1.1: Conversation Serialization    lines 86–209
   folds: none

3. Exercise 1.1.2 (Optional): Comparing Templates lines 211–226
   folds: none

4. Exercise 1.1.3: Injecting Control Tokens      lines 228–268
   folds:
     - summary @ 252 "Answer"                            body 253–259
     - summary @ 262 "Vocabulary: Base vs. Instruct"     body 263–267

5. Logprobs — What the Model Actually Computes   lines 272–...
   folds: ...
```

Each entry: section name, inclusive line range (header → one before next header), and a list of folds. Each fold: the summary line, its visible text (from the grep), and the body range (from `</summary>` + 1 up to `</details>`).

**Step 5: walk the plan in order.** For each section:

- **`folds: none`** — one `Read offset=<start> limit=<length>` call. Process the section normally.
- **Section with folds** — read in pieces, interleaved with attempts. Worked example for Exercise 1.1.3 above:
  1. `Read offset=228, limit=25` → lines 228–252: prose through the `</summary>` of the first fold. Summary text ("Answer") is visible; body is not.
  2. Attempt the in-prose question. Write your answer into `dayN_answers.py` as a `"""..."""` block.
  3. `Read offset=253, limit=7` → first fold body (253–259). Compare; note the gap in the review.
  4. `Read offset=260, limit=3` → prose up through the next `</summary>` (260–262).
  5. `Read offset=263, limit=5` → second fold body (the vocab box — no task, just background reading; read straight through since there's nothing to attempt).
  6. Done with the section; move to the next.

A summary with no task wording (just `Solution` / `Reference solution`) means the task is in the surrounding prose — still attempt before revealing the body.

**If the file is unusually long,** spawn an `Explore` subagent and give it this exact plan template to fill in. That isolates the reconnaissance from the main context entirely.

## Setup: persistent kernel

A student runs cells in an IDE or notebook where state persists between cells. Mirror this with a persistent Jupyter kernel via the bundled `scripts/kexec.py`, which handles kernel lifecycle and code execution.

**Pin the connection file to the day directory.** Without this, two agents running the skill at once would attach to the same default kernel and clobber each other's state. Use `dayN-*/.kexec-kernel.json` as the connection file and pass it via `-c` on **every** kexec invocation — including `--start`, `--stop`, `--status`, and code-exec calls. The path is derivable from the day folder you're already working in, so there's nothing to remember mid-walkthrough.

Use the venv's interpreter directly — `python` may not be on `PATH` and `source .venv/…/activate` differs by OS. The path is `.venv/bin/python` on macOS / Linux and `.venv/Scripts/python.exe` on Windows; Glob at the start to find which one exists, then use its full path everywhere below in place of `python`.

A normal code-exec call looks like:

```bash
python .claude/skills/exercise-walkthrough/scripts/kexec.py -c 1.1-llm-internals/.kexec-kernel.json <<'EOF'
# the cell's code goes here
EOF
```

State (variables, imports, open file handles) persists across calls that share the same `-c` path — exactly like in a notebook.

Lifecycle commands (always with the same `-c`):

```bash
python .claude/skills/exercise-walkthrough/scripts/kexec.py -c 1.1-llm-internals/.kexec-kernel.json --status   # is a kernel running?
python .claude/skills/exercise-walkthrough/scripts/kexec.py -c 1.1-llm-internals/.kexec-kernel.json --start    # start one explicitly (idempotent)
python .claude/skills/exercise-walkthrough/scripts/kexec.py -c 1.1-llm-internals/.kexec-kernel.json --stop     # shut it down
```

**Start each walkthrough with a fresh kernel.** Run `--stop` then `--start` (or just `--stop` and rely on auto-start) so you're not inheriting variables from an earlier session. A student opens their IDE with no state — you should too.

Verify with a trivial cell (`print("ok")`) before the first real cell so you catch environment problems immediately.

### Mirror VSCode Interactive Window semantics — set `__file__` up front

A bare Jupyter kernel does **not** define `__file__`. VSCode's Interactive Window (which is what most bootcamp students use for `# %%` cells) **does** — it sets `__file__` to the source script's path. If you don't mirror this, any setup block that does `Path(__file__).parent` for `sys.path` manipulation will crash here but work fine for students — a false positive in your review.

Before running any cells from the answer file, seed the kernel with `__file__`:

```bash
python .claude/skills/exercise-walkthrough/scripts/kexec.py -c 1.1-llm-internals/.kexec-kernel.json <<EOF
__file__ = r"<absolute path to dayN_answers.py>"
EOF
```

Do this once right after `--start`, before the Setup cell. If you hit a `NameError: name '__file__' is not defined` during the walkthrough, that's the symptom — set `__file__` and re-run; do **not** log it as a student-facing bug unless you've verified it also fails in VSCode's Interactive Window.

## Workflow per walkthrough

### 1. Locate files and create scratch state

When the user says **"review day N"** (or "walk through day N", "do day N", etc.), they mean the student-facing instructions file — `dayN_instructions.md`, which lives under `dayN*/` (the exact folder name varies - glob for `day<N>*/day<N>_instructions.md`). The instructions file is the **only** input; the sibling `dayN_solution.py` is the authored source and is off-limits (see the top of this skill).

- Instructions to read: `dayN-*/dayN_instructions.md`
- Answer file to write: `dayN-*/dayN_answers.py` — **overwrite** if it exists, start clean, as if the student just created it.
- Review output: `dayN-*/WALKTHROUGH_REVIEW.md` — create empty, append as you go.

**`dayN_answers.py` should look like what a real student would hand in** — not a review scratchpad. Keep it to:

- `# %%` cells copied verbatim from the instructions (imports, setup, driver code).
- Your implementations of the TODO functions.
- Written answers to discussion/prose questions, embedded in the same cell as `"""..."""` blocks. For questions where the instructions hide an answer in `<details>`, the student would write their own answer before expanding — so should you. Format like:
  ```python
  # %%
  """
  Q: How can logprobs be misused by an attacker?

  My answer: <what you think, written before peeking>
  """
  ```
  Keep your attempt even if it turned out wrong — the whole point of the walkthrough is to capture what a student actually produces.

Keep review meta-commentary (what confused you, what was ambiguous, suggested fixes) **out** of `dayN_answers.py`. That belongs in `WALKTHROUGH_REVIEW.md`. The answers file should be plausibly something a participant would turn in.

### 2. Walk through the file

Work from the reading plan you built in "Step 4: assemble the plan" above. That plan — with its line ranges and fold map — is your map; don't freelance Reads outside it, or you'll over-read into the next section or slurp a `<details>` body.

For each section in the plan, in order:

- Read the next chunk using the `offset`/`limit` from the plan (header line → one before the next header). For sections with folds, read in pieces as shown in the "Step 5: walk the plan" worked example: prose up to `</summary>`, attempt, then fold body.
- Copy each code block into `dayN_answers.py` as a `# %%` cell (preserving the cell boundary).
- For TODO exercises, implement using **only what's been shown so far** in the instructions. No peeking at later sections, the solution file, or `<details>` blocks.
- For prose questions: write your attempted answer into the answers file as a `"""..."""` block *first*, then expand the `<details>` reference answer to compare. Note the gap in the review (not in the answers file).
- Execute the cell via the kernel. Observe output.
- If it errors: behave like a student. Re-read the prose in the current section, look at the error, try one or two reasonable fixes. If you still can't resolve it in a few minutes of student-time, mark it as a blocker and move on.
- For test functions that run: let them run, note pass/fail, note the failure message quality ("does it tell the student what's wrong?").

### 3. Capture notes continuously

**Write to `dayN-*/WALKTHROUGH_REVIEW.md` as you go — don't hold notes in your head or in chat.**

Use these tags:

- **✅ Works** — ran fine, output matched expectations from the prose. (You don't need one of these for every cell — just call out when something is notably well done.)
- **❌ Bug** — code error, typo, wrong import, failing test with a legitimate cause, incorrect expected output in prose.
- **⚠️ Ambiguous** — the instructions didn't clearly specify something you had to decide. Say what choice you made and what alternative you considered.
- **📚 Missing context** — the exercise assumes knowledge the student likely doesn't have at this point. Name the concept and where it could be introduced.
- **💡 Suggestion** — not a bug; a way the exercise could land better.

For each note include:

- **Where**: exercise number + approximate line range in the instructions file (using the clickable `[file.md:42-51](path#L42-L51)` format).
- **What happened**: concrete observation — error message, the decision you had to make, how long you were stuck.
- **Fix**: suggested change, described against the *solution.py* since that's the editable source (the `.md` is generated by the build script). Describe the change in prose — quote the instructions-file line you'd expect to see differently, and say what `if "SOLUTION":` block or prose paragraph in the solution file should be edited. **Do not open `*_solution.py` to locate it precisely** — the author knows their own file, and a vague-but-clearly-scoped suggestion ("in the Setup section, add a note about X") is fine.

### 4. Final summary

At the end, prepend (or write at top) a brief overall summary:

- Rough time spent as a student
- Overall experience (one paragraph)
- Top 2–3 issues to address first

## What a good note looks like

Bad:

> The setup is confusing.

Good:

> ⚠️ Ambiguous — Setup, [section1_instructions.md:41-78](1.1-llm-internals/section1_instructions.md#L41-L78)
>
> The setup block imports `from section1_test import test_serialization` inside Exercise 1.1.1's code block ([section1_instructions.md:152](1.1-llm-internals/section1_instructions.md#L152)). At the moment a student pastes setup into `section1_answers.py`, `section1_test.py` doesn't exist — it's a build artifact. A student who hasn't run `./build-instructions.sh` yet will hit `ModuleNotFoundError` and have no hint why.
>
> Fix in `section1_solution.py`: add one line to the Setup prose, e.g. "Make sure `./build-instructions.sh` has been run first so that `section1_test.py` exists." Alternatively, lazy-import inside the test call.

Notes that specific are actionable. Notes that vague aren't.

### 5. Post review
If the user asks you to fix issues after you're done with the review, address them in the solution file, do not edit instructions file directly.

## Scope and practical notes

- **Optional exercises**: attempt them. A real student sees them too. If the user explicitly says to skip, skip.
- **API calls**: if an exercise needs `OPENROUTER_API_KEY` or similar and it isn't set, note the blocker, skip the cell, continue the walkthrough on cells that don't need it. Do not fabricate API output.
- **Don't edit the solution file** unless the user asks. The output of this skill is the review. The author acts on it.
- **Kernel cleanup**: leave the kernel running at the end unless explicitly asked for cleanup; the user may want to re-run cells or continue.

## Anti-patterns

- Reading the full instructions file at the start "just to understand the scope" — this invalidates the review.
- Pattern-matching from memory of similar bootcamp material — if you recognize the exercise, still go through it in real time; the point is to observe the *experience*, not to produce a correct solution fast.
- Fixing problems silently in your head instead of writing them down — if you had to figure something out, a student will have to figure it out, and that's worth noting.
- Treating `<details>` answers as ground truth without attempting first — the whole review signal comes from the gap between your attempt and the reference answer.
