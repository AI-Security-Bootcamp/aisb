# CLAUDE.md

This project is for creating content for AI Security Bootcamp, a 1-week program designed to teach participants about AI-relevant cybersecurity. Each day is preceded by recommended reading, starts with 1 hour of lecture, followed by 6 hours of hands-on labs. (The bootcamp is loosely inspired by other bootamps, MLAB2 and ARENA, which were focused on teaching ML skills and concepts relevant for AI alignment.) The participants work in pairs to complete the labs, and have instructors available to answer questions and debug issues.

The participants will be experienced cybersecurity experts who need upskilling in the AI-relevant topics.

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

Each day's solution file follows a consistent layout:

### Overall structure

1. **Title block** — first `# %%` cell with a triple-quoted string containing:
   - `# W1DX - Day Title`
   - Introductory paragraph (1-3 sentences framing the day)
   - `<!-- toc -->` placeholder for auto-generated table of contents
   - `## Content & Learning Objectives` section with numbered subsections and learning objectives in blockquotes
2. **Setup section** — imports, shared constants, API clients, utility functions
3. **Content sections** — each introduced with `## Section Name` (plain text, no leading emoji digit — see below), containing explanatory prose and exercises
4. **Summary section** — key takeaways (bulleted list) and further reading (linked list)
5. Use `# %%` cell markers between logical sections for readability

### Section and exercise headers

- Top-level content sections: `## Section Name`
- Exercises: `### Exercise N.M: Title` (N = section number, M = exercise within section)
- Sub-exercises or optional variants: `### Exercise N.Ma (Optional): Title`

**Do not put emoji digits (`1️⃣`, `2️⃣`, …) in section headers.** They were used previously but drift out of order whenever sections are added, removed, or moved between days (which happens often), and require manual reconciliation that is easy to get wrong. Use plain `## Section Name`. The exercise numbers (`N.M`) still encode ordering; keep those accurate, but section *headers* carry no number.

### Setup section
Include these instructions (replacing N and M with week/day numbers):

```markdown
Create a file named `wNdM_answers.py` in the `wNdM` directory. This will be your answer file for today.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your wNdM_answers.py file.**
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
### Exercise 1.1: Title

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
### Exercise 1.1: Do the Thing

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
- [ ] File starts with a title block (`# W1DX - Title`) containing intro, `<!-- toc -->`, and learning objectives
- [ ] Exercises use `### Exercise N.M: Title` format
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

Each day folder has a `README.md` with:
- `- [ ]` / `- [x]` checkboxes for topic tracking
- `*Italics*` for ML prerequisites
- `**Bold**` for day themes
- Nested bullets for sub-topics and exercise steps

## Language

Language should be easy to follow, avoid dense jargon, and avoid concepts they haven't learned yet.

Exercise instructions should roughly look like (inside the skeleton):

"""
### Exercise 1.1: Do the Thing

Explanation of what you need to do, and why we need to do it

Explanation of any gotchas in implementation, if there are any

Hints

Outcome. e.g. "The function should return ...". "By the end of this exercise...". Here is a good place to specify the format things
need to be returned in if there is any particular format.
"""

In general, focus on what needs to be done, not on how to do it (unless in the hints). Feel free to link official documentation and tutorials outside of the hints.