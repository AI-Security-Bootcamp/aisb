# Day 4.3 — quick handoff

New day replacing `4.3-safety-finetuning`, built on Arditi et al. 2024
([arXiv 2406.11717](https://arxiv.org/abs/2406.11717)): find the refusal direction, ablate it
to bypass refusal, add it to induce refusal, bake it into the weights, measure the cost.

**Branch:** `vegas-4.3-dev` (pushed to `github.com/davidquarel/aisb`). Old day preserved in `OLD/`.

## The files (in `4.3-refusal-direction/`)

| file | what | edit? |
|---|---|---|
| `section3_solution.py` | **source of truth** — all prose + exercises + tests | **yes, only this** |
| `section3_instructions.md` | participant markdown (generated) | no |
| `section3_test.py` | extracted tests (generated) | no |

You edit *only* `section3_solution.py`; the build system splits it into the participant
instructions (solutions swapped for TODO scaffolds) and the test file.

## Editing conventions

- Prose → top-level triple-quoted strings.
- Exercise → a function with `if "SOLUTION": <answer> else: <scaffold w/ TODO>`.
- Test → `@report`-decorated `test_*`, called at top level as `test_foo(foo)`.
- `# %%` = cell separators (cosmetic). Full rules in repo `CLAUDE.md`.

## Build & run

```bash
cd aisb/worktrees/4.3-refusal-single-dir

# build (regenerate .md + test from the solution). --watch / --force / --reference also available.
# NOTE the `env -u VIRTUAL_ENV` — required so the script uses .venv, not system python.
env -u VIRTUAL_ENV ./build-instructions.sh 4.3-refusal-direction/section3_solution.py

# run == verify (loads Qwen3-1.7B, runs all exercises+tests, demos, Inspect MMLU). GPU 0, ~6-8 min.
HF_HOME=REPOS/hf_cache CUDA_VISIBLE_DEVICES=0 .venv/bin/python 4.3-refusal-direction/section3_solution.py
```

Last verified: 12/12 tests pass, MMLU 54%→56% (capability preserved), intervention layer 19.

## Content (6 sections, 12 tested exercises)

1. Find the direction (format prompts, read residual stream, difference-in-means)
2. Bypass refusal (project-out, ablation hooks)
3. Induce refusal (activation-addition steering)
4. Bake into weights (orthogonalize, standalone abliterated model)
5. Measure cost from scratch (logit MCQ probe, standard error, random baseline)
6. Measure cost with Inspect (MMLU, original vs abliterated)

Model **Qwen3-1.7B**, intervention **layer 19** (chosen by a full sweep — fully removes refusal
with the least benign-behaviour disturbance; `REPOS/work/layer_sweep.py` reproduces it).

## Your audit checklist (none blocking)

- [ ] Review the ~16 inline harmful instructions in `section3_solution.py` before shipping.
- [ ] Decide: keep generated `.md`/`test.py` committed, or gitignore + build on demand?
- [ ] Optional: add a Gemma-3n-E2B bonus (E4B only partially bypasses — good discussion point).
- [ ] Fix repo-wide `requirements.txt` conflict (`control-arena` needs `datasets>=4.4.2`, pinned 3.6.0).
- [ ] Strip `HANDOFF.md` / `HANDOFF_DAVID.md` before merging to `vegas`.

Everything gitignored: `REPOS/` (dev scripts, model cache, 3.3 GB abliterated model), `.venv`,
the day's large outputs. Full R&D log (6 models verified, gotchas, file map) is in `HANDOFF.md`.
