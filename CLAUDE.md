# CLAUDE.md

This project is for creating content for AI Security Bootcamp, a 1-week program designed to teach participants about AI-relevant cybersecurity. Each day is preceded by required background reading (with longer overlapping resources clearly marked optional), starts with 1 hour of lecture, followed by 6 hours of hands-on labs. (The bootcamp is loosely inspired by other bootcamps, MLAB2 and ARENA, which were focused on teaching ML skills and concepts relevant for AI alignment.) The participants work in pairs to complete the labs, and have instructors available to answer questions and debug issues.

The participants will be experienced cybersecurity experts who need upskilling in the AI-relevant topics.

## Participant baseline and curriculum contracts

Use two different prerequisite levels when writing content:

1. **Assumed foundations.** Participants are experienced cybersecurity
   professionals and competent programmers. Content may assume security
   fundamentals such as threat modeling, trust boundaries, authentication and
   authorization, least privilege, injection, vulnerability analysis, detection,
   and defense in depth, along with working Python, shell, Git, and general API
   skills. Do not reteach or create exercises for these foundations; apply them
   directly to AI systems.
2. **Expected prior exposure, but not reliable fluency.** Participants should
   have encountered neural networks and gradient descent; introductory PyTorch
   tensors, autograd, optimizers, and training loops; tokenization and chat
   templates; the high-level LLM training and post-training pipeline; and LLM
   application concepts such as RAG, MCP, and coding agents in the published
   prework. Do not devote exercises to reviewing this background. When a section
   depends on it, provide a concise reference infobox, diagram, or precise
   refresher link so participants can recover the concept without losing the
   exercise's main thread.

Do not assume prior fluency with specialized libraries such as Hugging Face
Datasets/Trainer, PEFT, vLLM, ControlArena, diffusion-model hooks, or
representation-probe tooling unless an earlier section has taught it. If one of
these is a genuine engineering learning objective, introduce it explicitly and
scaffold it in proportion to its importance; otherwise provide the necessary
interface or reference.

Every instructional folder has a `README.md` that records its curriculum
contract:

- prerequisites coming in, separated into engineering, ML, security, and theory;
- intended learning outcomes in those same categories;
- required background or references; and
- TODOs describing where the current exercise differs from the intended state.

Prerequisites must be self-assessable. Write observable competencies such as
“given a confusion matrix, calculate precision and recall” or “write the
`zero_grad` → forward → loss → `backward` → `step` sequence,” not labels such as
“general software experience,” “basic ML,” or “familiarity with APIs.” A
participant should be able to check a prerequisite with one or two concrete
questions or tasks. If a category has no section-specific prerequisite, write
`-` rather than inventing a vague baseline.

Do not repeat assumed professional-programmer mechanics in section tables:
Python collections and control flow, JSON parsing, ordinary HTTP/API calls,
shell and Git use, simple rate calculations, and running or parameterizing a
function are baseline rather than section-specific prerequisites. Use `-` when
these are the only engineering requirements. Keep concrete requirements for
specialized work such as PyTorch/autograd, NumPy linear algebra, asynchronous
control APIs, model/tokenizer tooling, or infrastructure internals.

Use a **follow → show → lead** progression for each key concept:

1. **Follow (README):** link the concise external background and worked examples
   participants should review before the section, and state the observable
   learning outcomes. Background is required unless it is a long document that
   substantially duplicates the exercise; mark only those longer extensions as
   optional.
2. **Show (exercise):** specify the outcome and constraints, then ask participants
   to apply the technique without prescribing every implementation step. Provide
   scaffolding in proportion to whether the engineering work is itself a learning
   objective.
3. **Lead:** only add an independent extension when a content owner has designed
   and reviewed a specific activity with a clear deliverable, learning objective,
   and time budget. Do not append generic extension lists as boilerplate.

Keep this contract current when changing an exercise. It is acceptable—and often
desirable—to ask participants to retrieve a training-loop or API pattern from
official documentation and adapt it. Make that engineering objective explicit,
give a precise reference, and ensure the implementation work exposes the
security mechanism rather than merely repeating unrelated framework boilerplate.

The content should focus on in-depth topics, especially as they relate to catastrophic risks from AI, rather than shallow overviews.

## Code style
Code created here will be used for educational purposes. Make sure all the code is clean, readable, following best practices, and well explained with comments.
Prefer simplicity.

## Tool calling instructions
- Prefer your core tools for working with files (Read, Write, Edit, Glob, Grep, ...) over bash to avoid unnecessary tool call user approvals. Tell this to your Explore subagents too.
- Prefer reading library source files directly from site packages location over using Python introspection. Site packages may be in .venv, or /usr/local/lib/python3.12/site-packages if inside a Dev Container.

## Writing *_solution.py files

Solution files are the **single source of truth**. The build system (`./build-instructions.sh`) parses each `*_solution.py` and produces:

- **`*_instructions.md`** — Markdown document for participants. Top-level triple-quoted strings become prose; all other code becomes fenced Python blocks. Solution code is replaced with scaffolds.
- **`*_test.py`** — extracted `test_*` functions with necessary imports, ready to run with pytest.
- **`*_reference.py`** (with `./build-instructions.sh --reference` flag) — Python file containing only the solution blocks, for instructor use.

Everything a content creator writes goes in the `*_solution.py` file. 

The full signature of the build script is
```bash
./build-instructions.sh [--watch] [--force] [--reference] [files ...]
```


### File structure

A solution file is a sequence of **top-level statements** parsed in order:
- **Top-level triple-quoted strings** → rendered as Markdown prose in the instructions.
- **Everything else** (imports, functions, variable assignments, etc.) → rendered inside fenced ```python``` code blocks.

Adjacent code statements are merged into one code block. A new triple-quoted string starts a new Markdown section. Use `# %%` cell markers to visually separate sections (they are ignored by the build).

### Solution markers

Use `if "SOLUTION":` blocks to separate the reference answer from what participants see:

```python
def solve(data):
    if "SOLUTION":
        return [x * 2 for x in data]
    else:
        # TODO: transform each element
        pass
```

Participants see the `else` branch (or `"TODO: YOUR CODE HERE"` if there is no `else`). The reference keeps only the `if` body.

For functions where the **entire body** is the solution, use a bare string constant instead of an `if` block. Everything after `"SOLUTION"` is stripped (replaced with `pass`):

```python
def solve(data):
    "SOLUTION"
    return [x * 2 for x in data]
```

### Other block markers

- `if "SKIP":` — removed from **both** instructions and reference. Use for author-only debugging/testing code.
- `if "REFERENCE_ONLY":` — removed from instructions but **kept** in the reference. Use for code participants shouldn't see but the reference implementation needs.
- `if "TEST_FIXTURE":` — shared setup (sample data, API clients, constants) that belongs in all three outputs. The `if` body is **unwrapped to top level** in instructions, reference, and test files. If an `else:` branch is provided, the `else` body is shown in instructions (e.g. with a placeholder key) while the `if` body is used in the reference and test files (e.g. with the real value).

### Test functions

Define `test_*` functions at the top level. The build system automatically extracts them into `*_test.py` with the necessary imports. They never appear in the instructions.

```python
def test_solve():
    assert solve([1, 2, 3]) == [2, 4, 6]
```

### Markdown features inside triple-quoted strings

- **Table of contents**: Place `<!-- toc -->` where you want a generated TOC (built from `##`+ headers).
- **Hints**: Use `<details>`/`<summary>` blocks — they are automatically wrapped in `<blockquote>` for better rendering.
- **FIXME comments**: `<!-- FIXME ... -->` in Markdown or `# FIXME ...` in Python trigger build warnings. Use as authoring reminders.

## Content structure conventions

Each section's solution file follows a consistent layout. Day-level files that intentionally
combine several sections use `# Day D — Day Title` and `## Section D.S: Section Title`
instead; Day 0 setup exercises use `Exercise 0.E` because Day 0 has no numbered sections.

### Overall structure

1. **Title block** — first `# %%` cell with a triple-quoted string containing:
   - `# Day D — Section S: Section Title`
   - Introductory paragraph (1-3 sentences framing the day)
   - `<!-- toc -->` placeholder for auto-generated table of contents
   - `## Content & Learning Objectives` section with numbered subsections and learning objectives in blockquotes
2. **Setup section** — imports, shared constants, API clients, utility functions
3. **Content sections** — each introduced with `## Section Name` (plain text, no leading emoji digit — see below), containing explanatory prose and exercises
4. **Summary section** — key takeaways (bulleted list) and further reading (linked list)
5. Use `# %%` cell markers between logical sections for readability

### Section and exercise headers

- Top-level content sections: `## Section Name` (or `## Section D.S: Section Name` in a combined day-level file)
- Exercises: `### Exercise D.S.E: Title` (D = day, S = section, E = exercise within section)
- Sub-exercises or optional variants: `### Exercise D.S.Ea (Optional): Title`

**Do not put emoji digits (`1️⃣`, `2️⃣`, …) in section headers.** They were used previously but drift out of order whenever sections are added, removed, or moved between days (which happens often), and require manual reconciliation that is easy to get wrong. Use plain `## Section Name`. The exercise numbers (`D.S.E`) still encode ordering; keep those accurate, but section *headers* carry no number.

### Setup section
Include these instructions when a section has code for participants to execute
(replacing D with the day number and S with the section number). Do not add an empty setup to a
prose-only section; instead state its discussion or written deliverable directly.

```markdown
Create a file named `dayD_answers.py` in the `D.S-section-name` directory. This will be your answer file for this section.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your dayD_answers.py file.**
```

### Canonical `sys.path` / import boilerplate

Every solution file that imports `aisb_utils` (or any shared module) must make the
**workspace root** importable using the **exact snippet below** — do not hand-roll
`parent.parent` chains (they broke when files were nested at varying depths). This snippet
walks up to whichever ancestor directory contains `aisb_utils`, so it works regardless of how
deeply the file is nested:

```python
import sys
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply this file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
```

Import `report` via `from aisb_utils import report` (not `from aisb_utils.test_utils import
report`). Shared modules that live alongside the content (e.g. a day's `*_setup.py`) should be
imported relative to the workspace root; keep them inside the `X.Y-*` content folders, not in
legacy `dayN-*` folders.

### Learning objectives

Place learning objectives at the top of each numbered section inside the Content & Learning Objectives block:

```
### Section Name
Short description of the section.

> **Learning Objectives**
> - First objective
> - Second objective
```

### Difficulty and importance ratings

Place immediately after the exercise header line, as a plain `N/5` number (1-5 scale):

```
### Exercise 1.1.1: Title

> **Difficulty**: 2/5
> **Importance**: 4/5
```

**Use the numeric `N/5` form, not emoji circles.** The old `🔴🔴⚪⚪⚪` / `🔵🔵🔵🔵⚪` style is being retired: it is hard to author, easy to miscount, and inconsistent across files. Write `Difficulty: 2/5` and `Importance: 4/5`. Every exercise (code or prose) should carry both ratings.

## Exercise design patterns

### Code exercises

The typical pattern for a coding exercise:

1. **Prose** (triple-quoted string) explaining the task and any background
2. **Function definition** with a descriptive docstring
3. **`if "SOLUTION":` block** with the reference implementation
4. **`else:` block** with scaffold code for participants (TODO comments, placeholder values like `"YOUR ... HERE"`, step-by-step hints in comments)
5. **Top-level code** calling the function and printing results so participants see output
6. **Test function** (`def test_...():`) validating correctness

Coding exercises typically only make sense if correctness is testable by asserts in a test_ function. Prose exercises are more suitable for open ended questions.

Example skeleton:
```python
"""
### Exercise 1.1.1: Do the Thing

Explanation of what participants should do.
"""


def do_the_thing(data: list[int]) -> int:
    """Compute something from data.

    Returns the result.
    """
    if "SOLUTION":
        return sum(x * 2 for x in data)
    else:
        # TODO: Compute the result.
        # 1. Iterate over data
        # 2. Transform each element
        # 3. Aggregate
        pass


result = do_the_thing([1, 2, 3])
print(f"Result: {result}")


@report
def test_do_the_thing(solution: Callable[[list[int]], int]):
    result = solution([1, 2, 3])
    assert result == 12, f"Expected 12 for [1, 2, 3], got {result}"
    result = solution([])
    assert result == 0, f"Expected 0 for empty list, got {result}"
    print("  All tests passed!")


test_do_the_thing(do_the_thing)
```

### Prose / discussion exercises / open questions

For exercises that don't involve writing code, you can use building blocks such as these

* Ask **Question: ...** and provide answer in `<details><summary>Answer</summary>...</details>`
* Ask multiple questions in separate blocks: `<details><summary><b>Question 1:</b> ...</summary> answer...</details>`
* Give a task as `<details><summary><b>Task:</b> ...</summary> reference answer...</details>`
* Give a task as `**Task: [description]** <details><summary>Reference solution</summary> ...</details>`
* Provide reference solution in `<details><summary>Reference solution</summary>...</details>`
* Optionally provide progressive hints in `<details>` block


### Scaffold code conventions

- Use `# TODO:` comments describing what participants should implement
- Break complex tasks into numbered steps in comments
- Provide placeholder values: `"YOUR ... HERE"`, `pass`, `...`
- When useful, include partial code or type annotations as hints
- Keep the scaffold runnable (it should not crash before the participant changes it — use `pass` or return a dummy value)

## Hints and collapsible sections

Use `<details>`/`<summary>` for all collapsible content inside Markdown strings. The build system auto-wraps them in `<blockquote>` for better rendering.

Common patterns:

- **Progressive hints**: `<summary>Hint 1</summary>`, `<summary>Hint 2</summary>`, etc. — each reveals more, from a nudge to near-complete solution
- **Vocabulary boxes**: `<summary>Vocabulary: Topic Name</summary>` — define terms relevant to the section
- **Background info**: `<summary>Background: Topic</summary>` — optional deeper context
- **Reference solutions**: `<summary>Solution</summary>` or `<summary>Reference solution</summary>` — for prose exercises
- **Answers to questions**: `<summary>Answer</summary>` — for inline discussion questions

## Test function conventions

- Import `from aisb_utils import report` and decorate test functions with `@report` so results are visible to participants
- Test functions are automatically extracted to `*_test.py` by the build system
- Tests should verify correctness without revealing the full solution approach
- Assert messages should be informative — tell the participant what went wrong (input, expected, actual) without giving away the implementation. E.g. `f"Expected 64 bytes for empty input, got {len(result)}"`
- Include both basic cases and edge cases
- Print a short success message: `print("  All tests passed!")` or similar
- Test functions take the solution function as a parameter (so participants can pass their own implementation):

```python
@report
def test_md5_padding(solution: Callable[[bytes], bytes]):
    assert len(solution(b"")) == 64
    assert len(solution(b"a" * 56)) == 128
```

- Tests must be called explicitly at the top level of the solution file, passing the solution function as an argument. This is what participants will copy into their answer file:

```python
test_md5_padding(md5_padding)
```

## Content creator checklist

Before submitting a solution file, verify:

**Structure**
- [ ] File starts with a title block (`# Day D — Section S: Title`, or `# Day D — Title` for a combined day-level file) containing intro, `<!-- toc -->`, and learning objectives
- [ ] Exercises use `### Exercise D.S.E: Title` format
- [ ] Exercises have difficulty and importance ratings
- [ ] File ends with a Summary section (key takeaways + further reading)
- [ ] `# %%` cell markers separate logical sections

**Exercises**
- [ ] Every code exercise has a clear docstring explaining the task
- [ ] `if "SOLUTION":` / `else:` blocks are present for every function participants implement
- [ ] Scaffold code is runnable (won't crash before participant edits it)
- [ ] Top-level code calls the function and prints output so participants see results
- [ ] Test function(s) exist where relevant (decorated with `@report`)

**Build**
- [ ] `./build-instructions.sh <file>` runs without errors
- [ ] The `*_solution.py` file is executalbe without errors
- [ ] No unintended `<!-- FIXME -->` comments remain (they trigger build warnings)
- [ ] Generated `*_instructions.md` renders correctly (check in a Markdown previewer)

**Content quality**
- [ ] Prose is concise and well structured; paragraphs are short
- [ ] Code is clean, readable, and well commented
- [ ] External resources are linked where relevant
- [ ] Images are stored in a `resources/` subdirectory and referenced with relative paths

## Day README Format

Each instructional folder has a `README.md` with:

- a section-level suggested time and a direct link to `*_instructions.md`;
- a table of observable incoming prerequisites and outgoing learning outcomes,
  split into engineering, ML, security, and theory;
- `-` for any prerequisite category with no section-specific requirement;
- a Background section with direct links to required concise background and
  clearly marked optional longer reading; and
- author-facing current-state TODOs kept separate from participant instructions.
