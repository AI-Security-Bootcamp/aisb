# Handoff — Day 4.3 "Refusal is a single direction"

**Goal:** replace the old `4.3-safety-finetuning` section with a new day built around
*"Refusal in Language Models Is Mediated by a Single Direction"* (Arditi et al., 2024,
[arXiv 2406.11717](https://arxiv.org/abs/2406.11717)).

**Location:** git worktree at `aisb/worktrees/4.3-refusal-single-dir/`, checked out on branch
**`vegas-4.3-dev`** (created from `vegas`). The worktree directory name (`4.3-refusal-single-dir`)
and the content folder name (`4.3-refusal-direction/`) differ — don't be confused by that.

## Status: content DRAFTED, verified, committed, pushed

The day is written and working (this section supersedes the earlier "not yet written" note):

- **`4.3-refusal-direction/section3_solution.py`** — the single source of truth. 12 exercises,
  each with a hardened `@report` test. Builds cleanly and **executes end-to-end with all 12
  tests passing** (verified on GPU, including the in-file Inspect MMLU eval).
- Committed to `vegas-4.3-dev` in 5 small commits and **pushed to the fork**
  `https://github.com/davidquarel/aisb` (remote `fork`; `origin` still points at upstream
  `AI-Security-Bootcamp/aisb`). Awaiting the user's audit before any merge to `vegas`.

Remaining work is audit + polish, not core implementation — see "Open items".

---

## Cold-start for a new Claude (read this first)

You are picking up someone else's work. Orient before touching anything:

1. **Absolute path:** `/workspace/HOME/guest/david_quarel/aisb/worktrees/4.3-refusal-single-dir/`.
   This is a **git worktree** of the main repo at `/workspace/HOME/guest/david_quarel/aisb/`
   (the main checkout is on branch `vegas`; this worktree is on `vegas-4.3-dev`). Run all
   commands from the worktree root unless a step says otherwise.
2. **Read `CLAUDE.md`** (repo root) — it governs how content is authored and built. Non-obvious
   build gotchas it does *not* cover are in "Authoring lessons" below.
3. **Read `HANDOFF_DAVID.md`** — the short version for the human owner (David). This file is the
   long version for you.
4. **A project memory** at `~/.claude/projects/-workspace-HOME-guest-david-quarel-aisb/memory/`
   (`day-4.3-refusal-direction.md`) points here — treat it as a stale pointer, trust this file.
5. **Confirm the branch and that it's clean:** `git -C <worktree> status` and
   `git log --oneline vegas..HEAD`. Then **sanity-check you can build and run** (commands below)
   before making changes — the env drifts (torch pins, GPU availability) between sessions.
6. **Commit style:** Conventional-Commit prefixes (`content(4.3):`, `chore(4.3):`, `docs(4.3):`),
   a `Co-Authored-By:` trailer, and **your own** session trailer — do not copy the previous
   session's `Claude-Session:` URL. Small, logical commits; the user audits before merging.

---

## The day itself (`4.3-refusal-direction/`)

Model: **Qwen3-1.7B** (ungated, clean, fast). Layer: **19** (of 28), hardcoded and given for
free. Chosen by a full sweep (`REPOS/work/layer_sweep.py`): among the many layers that fully
remove refusal, layer 19 has the lowest collateral damage (KL 0.086 on benign prompts vs e.g.
0.92 at layer 14) while keeping MMLU at baseline. Most mid-to-late layers work; layer 0 removes
refusal but destroys capability (MMLU ≈ random).

Section arc, each exercise with a self-contained test:
1. **Find the direction** — `format_instructions`, `get_mean_activations` (hooks), `compute_refusal_directions` (difference-in-means).
2. **Bypass refusal** — `project_out` (projection onto orthogonal complement), `get_ablation_hooks`.
3. **Induce refusal** — `get_steering_hook` (activation addition).
4. **Bake into weights** — `orthogonalize_matrix`, `abliterate_model` (permanent, no hooks).
5. **Measure cost (from scratch)** — `mcq_predict` (logit-based MCQ), `standard_error`, `expected_random_accuracy`.
6. **Measure cost (properly)** — `mmlu_record_to_sample` + a provided Inspect MMLU harness comparing original vs abliterated.

In-file result (MMLU, 50 Q for speed): original 54% ±7% → abliterated 54% ±7%, random 25% — capability preserved. Behavioural demos (baseline vs ablated / steered / baked) print inline.

Files (the committed triple): `section3_solution.py` (source), `section3_instructions.md`,
`section3_test.py` (both generated — do not hand-edit). Large local outputs
`4.3-refusal-direction/{abliterated_qwen3,inspect_logs}/` are gitignored.

---

## Authoring lessons for the aisb build system (READ before writing/auditing content)

These cost real debugging time; they are not obvious from `CLAUDE.md`.

- **Run the builder without `VIRTUAL_ENV` set:** `env -u VIRTUAL_ENV ./build-instructions.sh <file>`.
  The script trusts a set `VIRTUAL_ENV` and runs the *system* python (which lacks `libcst`);
  unset it so the script activates `.venv`. The venv needs `libcst black termcolor` installed.
- **What lands in the generated `*_test.py`:** only (a) collected imports, (b) `if "TEST_FIXTURE":`
  bodies, and (c) `test_*` functions. Consequences:
  - A `TEST_FIXTURE` helper must **not** call an exercise function (they're absent from the test
    file). E.g. the `generate` helper inlines its own tokenization rather than calling the
    `format_instructions` exercise.
  - A test body must **not** reference exercise-computed globals (`refusal_dir`, `harmful_means`,
    …) — they don't exist in the test file. Make every test self-contained: synthetic inputs, or
    the fixture model + a random direction, or compute-from-fixtures. **Behavioural end-to-end
    validation lives in the solution file's top-level demo prints, not in asserts.**
  - Solution functions passed into a test carry their *own* module globals, so a participant's
    exercise calling their other exercise works — the only breakage is test-file-resident code.
- **bf16 batched matmuls are not bit-identical across batch sizes.** Don't assert exact equality
  between batch-1 and batch-2 results (a "mean of a repeated prompt == single prompt" test failed
  this way). Compare at the same batch size, or use tolerant/structural checks.
- **Destructive exercises:** `abliterate_model` mutates weights in place, so its test runs on a
  tiny `SimpleNamespace` + `nn.Linear/Embedding` stand-in, not the real model (orthogonalization
  is idempotent, which the test also checks).

Inspect (`inspect_ai`) gotchas:
- `temperature=0.0` is rejected by the HF provider — pass `do_sample=False` as a **model_arg**
  (`GenerateConfig` has no `do_sample`).
- Two 1.7B models don't co-fit on one 16 GB GPU if loaded float32 / large batch. Free in-memory
  models first; use `batch_size` + `max_connections` limits; keep bf16.
- `save_pretrained` records dtype under the new `dtype` key; **also set legacy `torch_dtype`** in
  the saved `config.json` or the loader defaults to float32 (2× memory → OOM).
- Qwen3: pass `enable_thinking=False` (model_arg) so MCQ answers are a few tokens, not a long CoT.

---

## R&D verification (the exploration behind the day) — all under gitignored `REPOS/`

The technique was verified on 6 models before the lab was written:

| model | how | refusal ASR baseline → ablation |
|---|---|---|
| gemma-2b-it (paper's original) | committed direction, substring judge | 0.09 → 0.98 *(committed 0.09→1.00)* |
| Qwen-1.8B-Chat (paper's original) | **removed from HF**; validated vs committed | 0.30 → 0.99 (committed) |
| Qwen3-0.6B | gen-based layer select (L8) | 0.77 → 0.77 augmented *(metric noisy; samples flip)* |
| Qwen3-1.7B | gen-based layer select (L14) | **0.39 → 0.99** augmented |
| gemma-3n-E2B-it | custom AltUp adapter (L15) | **0.32 → 1.00** augmented |
| gemma-3n-E4B-it | custom AltUp adapter (L16) | 0.40 → 0.94 augmented *(bomb only partially bypasses)* |

R&D capability (full, 150 Q, ±1 SE): ARC-Challenge 70.7±3.7% → 70.0±3.7%; MMLU 55.3±4.1% →
56.7±4.0%. (ARC-Easy 94%→94% — dropped, near-ceiling.)

Key R&D findings (also drive the lab design):
1. **`Qwen/Qwen-1_8B-Chat` is gone from HF** (only base + Int4/Int8 remain).
2. **The paper's `P(first token ∈ {"I","As"})` selection metric doesn't transfer** to modern
   soft-refusal models — it empties the dataset and filters out every candidate
   (`AssertionError: All scores have been filtered out!`). Fix: select the layer by *generation
   bypass*. Extraction + directional ablation are still exactly the paper's method.
3. **Substring refusal judges are unreliable for modern refusals** (both under/over-count). Use
   an augmented judge + qualitative samples. (A real LlamaGuard/Together judge was out of scope.)
4. **Gemma 3n uses AltUp** — decoder input is `[num_altup=4, B, S, D]` (active idx 0); the repo's
   mean-capture assumes `[B,S,D]`. Custom adapter captures the active stream; ablation hooks work
   unchanged. Needs `timm`; loads via `Gemma3nForConditionalGeneration` at
   `model.model.language_model.layers`.
5. **E4B is more robust** — single-layer ablation only partially bypasses the bomb prompt
   (deflects to a "baking-soda volcano"). Good discussion point on scale.

### `REPOS/` file map (all gitignored)

```
REPOS/
├── refusal_2406.11717.pdf, paper_src/     paper PDF + arXiv TeX source
├── hf_cache/                              HF model cache (HF_HOME points here)
├── refusal_direction/                     cloned github.com/andyrdt/refusal_direction, MY EDITS:
│   ├── pipeline/model_utils/qwen3_model.py       NEW: Qwen3 adapter
│   ├── pipeline/model_utils/model_factory.py     EDITED: route 'qwen3' → Qwen3Model
│   └── pipeline/submodules/evaluate_jailbreak.py EDITED: vllm/litellm imports optional
└── work/                                  scripts + outputs
    ├── smoke_test.py, reproduce_ablation.py, run_pipeline_lite.py
    ├── abliterate.py            Qwen/Llama gen-based select + eval + samples
    ├── gemma3n_abliterate.py    Gemma 3n (AltUp-aware) abliteration
    ├── bake_model.py            orthogonalize weights → save standalone model
    ├── generate_rollouts.py     baseline/steering/ablation rollouts → qwen3_1.7b_rollouts.md
    ├── mcq_eval.py              inspect ARC-Challenge + MMLU, one model per process
    ├── layer_sweep.py           per-layer bypass + MMLU + KL sweep → best layer (19 for Qwen3-1.7B)
    ├── plot_results.py          capability_plot.png (SE bars + random/human refs)
    ├── timing.py                extraction/ablation/steering timing
    ├── qwen3-1.7b-abliterated/  saved baked-in model (~3.3 GB bf16)
    ├── direction_*.pt, *_*.json, *.log, inspect_logs/
    ├── venv_freeze.txt          exact `uv pip freeze` of the working .venv
    └── (sent to user: qwen3_1.7b_rollouts.md, capability_plot.png)
```

---

## Environment & how to run

- `.venv` at the worktree root. It was installed from a **subset** of `requirements.txt` because
  that file is **currently uninstallable**: `control-arena==17.1.1` needs `datasets>=4.4.2` but it
  pins `datasets==3.6.0`. **Bump that pin for the full env.** (Also: installing `timm` silently
  bumps torch, so torch/torchvision/triton must be pinned in the *same* install command.)
- **If `.venv` is missing/broken, recreate it with this exact command** (verified working set;
  full snapshot in `REPOS/work/venv_freeze.txt`). Pinning torch+torchvision+triton up front stops
  `timm` from clobbering torch:
  ```bash
  cd /workspace/HOME/guest/david_quarel/aisb/worktrees/4.3-refusal-single-dir
  uv venv --python 3.12 .venv
  VIRTUAL_ENV=.venv uv pip install \
    torch==2.4.0 torchvision==0.19.0 triton==3.0.0 \
    transformers==4.57.6 accelerate==1.10.1 datasets==3.6.0 huggingface_hub==0.35.3 \
    sentencepiece tokenizers protobuf tqdm tiktoken numpy scipy matplotlib \
    jaxtyping einops transformers-stream-generator inspect-ai==0.3.244 timm \
    libcst black termcolor \
    --extra-index-url https://download.pytorch.org/whl/cu124
  ```
  Key installed versions: torch 2.4.0+cu124, transformers 4.57.6, inspect-ai 0.3.244, datasets
  3.6.0, timm 1.0.28, libcst 1.8.6.
- Build the day (regenerate `.md` + `_test.py` from the solution) — note `env -u VIRTUAL_ENV`,
  see Authoring lessons for why:
  ```bash
  env -u VIRTUAL_ENV ./build-instructions.sh 4.3-refusal-direction/section3_solution.py
  ```
- Run the day content directly (this is also the verification — runs every exercise + test +
  the Inspect eval; working dir = repo/worktree root; ~6–8 min, GPU 0):
  ```bash
  HF_HOME=REPOS/hf_cache CUDA_VISIBLE_DEVICES=0 .venv/bin/python 4.3-refusal-direction/section3_solution.py
  ```
  R&D scripts run from the cloned repo (cwd drifts — always `cd`):
  ```bash
  cd REPOS/refusal_direction && HF_HOME=<ABS>/REPOS/hf_cache PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
      ../../.venv/bin/python ../work/<script>.py [args]
  ```
- GPUs: 3× RTX A4000 (16 GB). **GPU 0** is the free one; GPU 1 throws "Unknown Error"; GPU 2/3 are
  used by others. Two 1.7B models don't co-fit on one card → one model per process. GPU
  availability drifts between sessions — check `nvidia-smi` and pick a free card via
  `CUDA_VISIBLE_DEVICES` (Qwen3-1.7B needs it; the paper's original **Gemma models are gated**).
- **HuggingFace token:** an active token for user `davidquarel` is present in the environment
  (gives gated Gemma access). Verify with:
  `.venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"`.
  Qwen3 is ungated and needs no token; only the Gemma R&D scripts do.

## Git / pushing

- Remotes: `origin` = upstream `AI-Security-Bootcamp/aisb` (**do not push here**); `fork` =
  `https://github.com/davidquarel/aisb` (the user's fork — push here). `gh` is authenticated as
  `davidquarel` over HTTPS (`gh auth setup-git` already run).
- Push the branch with: `git push fork vegas-4.3-dev`. The branch is **not** to be merged into
  `vegas` by you — the user audits first.
- Committed files are small text only. `REPOS/`, `.venv`, and the day's large outputs
  (`abliterated_qwen3/`, `inspect_logs/`) are gitignored; never `git add -f` them.

---

## Open items / next steps

1. **User audit** of the day before merging `vegas-4.3-dev` → `vegas`.
2. **Optional exercise to consider:** a Gemma-3n-E2B "current-architecture" demo with E4B's
   partial bomb-refusal as a discussion point. (The layer sweep is done — layer 19 is baked in
   for free; `layer_sweep.py` reproduces it if you want to show the landscape.)
3. **Review the inline harmful dataset** (~16 AdvBench-style instructions in the solution file) —
   confirm it's acceptable to ship verbatim in course content.
4. **Fix `requirements.txt`** `datasets` pin conflict for the full bootcamp env.
5. Decide whether the generated `*_instructions.md` / `*_test.py` should be committed (currently
   are, for review convenience) or gitignored and built on demand, per repo convention.
6. `HANDOFF.md` and `HANDOFF_DAVID.md` are dev clutter — strip both before merge to `vegas`.
