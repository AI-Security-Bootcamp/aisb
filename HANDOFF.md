# Handoff — Day 4.3 "Refusal is a single direction" (worktree `4.3-refusal-single-dir`)

**Goal:** replace the old `4.3-safety-finetuning` section with a new day built around
*"Refusal in Language Models Is Mediated by a Single Direction"* (Arditi et al., 2024,
[arXiv 2406.11717](https://arxiv.org/abs/2406.11717)). This handoff covers the R&D /
verification phase. **No `section*_solution.py` has been written yet** — that's the next step.

Branch `4.3-refusal-single-dir` (worktree of the `aisb` repo), based on `vegas`.

---

## TL;DR of what's verified

The paper's technique reproduces. Ablating one difference-in-means direction turns refusal
into compliance; adding it induces refusal on benign prompts; baking it into the weights
gives a permanently "abliterated" model with **no capability loss**.

| model | how | refusal ASR baseline → ablation |
|---|---|---|
| gemma-2b-it (paper's original) | committed direction, substring judge | 0.09 → 0.98 *(committed target 0.09→1.00)* |
| Qwen-1.8B-Chat (paper's original) | **removed from HF**; validated scoring vs committed | 0.30 → 0.99 (committed) |
| Qwen3-0.6B | full pipeline, gen-based layer select (L8) | 0.77 → 0.77 augmented *(metric noisy; samples flip)* |
| Qwen3-1.7B | gen-based layer select (L14) | **0.39 → 0.99** augmented |
| gemma-3n-E2B-it | custom AltUp adapter (L15) | **0.32 → 1.00** augmented |
| gemma-3n-E4B-it | custom AltUp adapter (L16) | 0.40 → 0.94 augmented *(bomb only partially bypasses)* |

**Capability of baked-in abliterated Qwen3-1.7B** (inspect_ai, 150 Q, ±1 SE):
- ARC-Challenge: 70.7±3.7% → 70.0±3.7%
- MMLU: 55.3±4.1% → 56.7±4.0%
- (ARC-Easy 94%→94% — **dropped**, too easy/near-ceiling to be informative.)

Differences are within noise → ablation is surgical.

---

## Where everything lives

Everything R&D is under **`REPOS/`**, which is **gitignored** (see `.gitignore`). Only two
things are staged in the repo: the `OLD/` move and the `.gitignore` edit.

```
worktrees/4.3-refusal-single-dir/
├── HANDOFF.md                         ← this file (untracked)
├── OLD/4.3-safety-finetuning/         ← old section, preserved via `git mv` (STAGED)
├── .gitignore                         ← added `REPOS/` (STAGED)
├── .venv/                             ← isolated env (gitignored). torch 2.4.0, transformers 4.57.6,
│                                        inspect-ai 0.3.244, timm. See "Environment" below.
└── REPOS/                             ← ALL gitignored working area
    ├── refusal_2406.11717.pdf         ← paper (5-page NeurIPS version)
    ├── paper_src/                     ← full arXiv TeX source (refusal.tex etc.)
    ├── hf_cache/                      ← HF model cache (HF_HOME points here)
    ├── refusal_direction/             ← cloned github.com/andyrdt/refusal_direction, WITH MY EDITS:
    │   ├── pipeline/model_utils/qwen3_model.py        ← NEW: Qwen3 adapter I wrote
    │   ├── pipeline/model_utils/model_factory.py      ← EDITED: routes 'qwen3' → Qwen3Model
    │   └── pipeline/submodules/evaluate_jailbreak.py  ← EDITED: made vllm/litellm imports optional
    │   └── pipeline/runs/{gemma-2b-it,qwen-1_8b-chat,...}/  ← upstream's COMMITTED artifacts (direction.pt etc.)
    └── work/                          ← all my scripts + outputs
        ├── smoke_test.py              ← load + generate + ablate (quick sanity)
        ├── reproduce_ablation.py      ← gemma-2b-it repro using committed direction (→ reproduce_gemma-2b-it.json)
        ├── run_pipeline_lite.py       ← full pipeline trimmed to refusal-score-only, --no_filter
        ├── abliterate.py              ← Qwen/Llama-family: gen-based layer select + eval + samples
        ├── gemma3n_abliterate.py      ← Gemma 3n (AltUp-aware) abliteration
        ├── bake_model.py              ← orthogonalize weights → save standalone abliterated model
        ├── generate_rollouts.py       ← baseline/steering/ablation rollouts → qwen3_1.7b_rollouts.md
        ├── mcq_eval.py                ← KEEP: inspect_ai ARC-Challenge + MMLU eval (per model, per benchmark)
        ├── arc_eval.py                ← superseded (ARC-Easy); kept for reference
        ├── plot_results.py            ← capability_plot.png with SE error bars + random/human refs
        ├── timing.py                  ← timing of extraction / ablation / steering
        ├── qwen3-1.7b-abliterated/    ← SAVED baked-in model (standalone HF model, ~3.3GB bf16)
        ├── direction_*.pt             ← saved refusal directions per model
        ├── *_*.json                   ← eval results + completions per model
        ├── qwen3_1.7b_rollouts.md     ← sent to user
        ├── capability_plot.png        ← sent to user
        ├── inspect_logs/              ← inspect .eval logs
        └── *.log                      ← run logs
```

---

## Key technical findings (bake these into the lab)

1. **The paper's original Qwen 1.8B (`Qwen/Qwen-1_8B-Chat`) is GONE from HF.** Only base +
   Int4/Int8 remain. Use gemma-2b-it for a faithful original-model repro, or lean on Qwen3.

2. **The paper's direction-SELECTION metric doesn't transfer to modern models.** It scores
   directions by `P(first generated token ∈ {"I","As"})`. Qwen3/Gemma-3n refuse with topic
   words ("*Hotwiring a car is illegal...*"), so that metric collapses the dataset during
   filtering and then filters out every candidate direction (`AssertionError: All scores have
   been filtered out!`). **Fix used:** `abliterate.py` / `gemma3n_abliterate.py` select the
   layer by *actual generation bypass* on a held-out harmful set (the standard "abliteration"
   approach). The extraction + directional ablation are still exactly the paper's method.

3. **Substring refusal judges are unreliable for modern refusals** (both under- and
   over-counting; e.g. an ablated compliance that says "note this is illegal" is scored as a
   refusal). Qwen3-0.6B shows a flat 0.77→0.77 despite samples clearly flipping. Report an
   augmented judge AND lean on qualitative samples. A proper judge (LlamaGuard/Together) was
   out of scope (user chose refusal-score-only, no paid API).

4. **Gemma 3n uses AltUp** — each decoder layer's residual input is `[num_altup=4, B, S, D]`,
   active stream index 0. The repo's mean-activation capture assumes `[B,S,D]` and breaks.
   `gemma3n_abliterate.py` captures the active stream for extraction; the ablation hooks work
   unchanged (they broadcast over the last dim). Gemma 3n also needs `timm` (vision tower) and
   loads via `Gemma3nForConditionalGeneration` at `model.model.language_model.layers`.

5. **E4B is more robust:** at a single layer the bomb prompt only *partially* bypasses
   (deflects to a "baking-soda volcano"). Good discussion point on scale vs. abliteration.

6. **Timing (Qwen3-1.7B, one RTX A4000, bf16):** direction extraction (256 prompts) 1.3s;
   ablation adds ~28% per-token overhead (0.84→1.08 s/prompt at 256 tok); steering (1 hook) is
   free; each 150-Q inspect eval ~20–40s + ~15s load. All well within an interactive lab.

---

## Environment & how to run

- `.venv` at the worktree root. Built from a **subset** of `requirements.txt` because the
  repo's `requirements.txt` is **currently uninstallable**: `control-arena==17.1.1` needs
  `datasets>=4.4.2` but the file pins `datasets==3.6.0`. **Bump that pin for the full env.**
- Installed on top: `inspect-ai==0.3.244`, `timm` (for Gemma 3n). Installing timm silently
  upgraded torch → had to re-pin `torch==2.4.0 torchvision==0.19.0 triton==3.0.0`.
- Standard run pattern (working dir must be the cloned repo; it drifts, so always `cd`):
  ```bash
  cd REPOS/refusal_direction
  HF_HOME=<ABS>/REPOS/hf_cache PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
      ../../.venv/bin/python ../work/<script>.py [args]
  ```
- GPUs: box has 3× RTX A4000 (16GB). **GPU 0** is the free one; GPU 1 throws "Unknown Error",
  GPU 2/3 are used by others. Two 1.7B models don't co-fit on one 16GB card → run one model
  per process (see `mcq_eval.py`, invoked once per model).
- HF token for gated Gemma is already active in the environment (user `davidquarel`).

---

## Open items / next steps

1. **Write `4.3-refusal-single-dir/section*_solution.py`** — the actual day content. Suggested
   arc: extract difference-in-means direction → runtime steering & ablation (interactive `# %%`
   cells) → **bake into weights** (orthogonalization, save new model) → **MMLU before/after
   capability check** with inspect_ai. Recommended model: **Qwen3-1.7B** (ungated, clean, fast);
   optionally Gemma-3n-E2B as a "current-architecture" demo, with E4B's partial bomb-refusal as
   a discussion point. Follow `CLAUDE.md` conventions (title block, `if "SOLUTION":` markers,
   `@report` tests, plain `## headers`, `N/5` ratings, canonical `sys.path` snippet).
2. Use **MMLU + ARC-Challenge** for the capability eval (drop ARC-Easy). `mcq_eval.py` is ready.
3. Decide whether to commit `HANDOFF.md` and whether the new day's shared code (a `*_setup.py`,
   the Qwen3 adapter, bake/eval helpers) lives in the `4.3-*` content folder (per CLAUDE.md,
   shared modules go in the `X.Y-*` folder, not legacy `dayN-*`).
4. Fix the `requirements.txt` `datasets` pin conflict for the full bootcamp env.
