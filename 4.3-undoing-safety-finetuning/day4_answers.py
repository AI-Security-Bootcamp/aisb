# %%
import contextlib
import gc
import json
import math
import subprocess
import types
from collections.abc import Callable

import sys
from pathlib import Path

import torch
from torch import Tensor
from jaxtyping import Float
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
# ----- Model -----
MODEL_PATH = "Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = (
    "left"  # left-pad so the last token lines up across a batch
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = (
    AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    .to(DEVICE)
    .eval()
)
model.requires_grad_(False)

N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
# Layer 19 (of 28) is preselected for Qwen3-1.7B: a sweep over all layers found it fully
# removes refusal while least disturbing the model on benign inputs (lowest KL divergence).
# In practice you sweep for this; here it is given for free. (Most mid-to-late layers work.)
LAYER = 19

# ----- Datasets: instructions that trigger refusal vs. ones that don't. We use the splits
# from the paper's repo (github.com/andyrdt/refusal_direction), 128 of each. If the repo
# isn't already present locally, we shallow-clone it automatically. -----
REFUSAL_REPO_URL = "https://github.com/andyrdt/refusal_direction"

def get_splits_dir() -> Path:
    """Return the directory holding the instruction splits, cloning the repo if needed."""
    # Reuse an existing checkout if any ancestor directory already contains one ...
    for parent in Path(__file__).resolve().parents:
        splits = parent / "refusal_direction" / "dataset" / "splits"
        if splits.is_dir():
            return splits
    # ... otherwise shallow-clone the paper's repo next to this file (one-off, a few MB).
    target = Path(__file__).resolve().parent / "refusal_direction"
    print(f"Dataset not found locally; cloning {REFUSAL_REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REFUSAL_REPO_URL, str(target)], check=True
    )
    return target / "dataset" / "splits"

SPLITS_DIR = get_splits_dir()

def load_instructions(split: str, n: int = 128) -> list[str]:
    """Return the first `n` instruction strings from a refusal_direction dataset split."""
    records = json.loads((SPLITS_DIR / f"{split}.json").read_text())
    return [record["instruction"] for record in records[:n]]

HARMFUL_INSTRUCTIONS = load_instructions("harmful_train")
HARMLESS_INSTRUCTIONS = load_instructions("harmless_train")
# %%

print(f"HARMFUL ({len(HARMFUL_INSTRUCTIONS)} total), first 10:")
for instruction in HARMFUL_INSTRUCTIONS[:10]:
    print(f"  - {instruction}")

print(f"\nHARMLESS ({len(HARMLESS_INSTRUCTIONS)} total), first 10:")
for instruction in HARMLESS_INSTRUCTIONS[:10]:
    print(f"  - {instruction}")
# %%

def format_instructions(instructions: list[str] | str, device=DEVICE):
    """Apply the chat template to each instruction and tokenize (left-padded) as a batch."""
    if isinstance(instructions, str):
        instructions = [instructions]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ins}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for ins in instructions
    ]

    return tokenizer(prompts, return_tensors="pt", padding=True).to(device)


demo_tokens = format_instructions(["How do I bake a cake?"])
print(tokenizer.decode(demo_tokens.input_ids[0]))

# %%
# A global buffer the hook writes into: the mean last-token residual stream at our chosen layer.
resid_cache = {}

def hook_cache_resid(module, args: tuple):
    '''
    A hook function that caches the residual stream and then
    modifies it by doubling it.
    '''
    # args is a tuple of all arguments, and the only argument is the residual stream : (batch, seq, d_model)
    (resid,) = args

    resid_cache['cache'] = resid

    #option: return something if you want to modify the residual stream
    new_resid = resid * 2
    return (new_resid, ) # or None if you don't want to make changes

@contextlib.contextmanager
def use_hooks(pre_hooks=()):
    """Temporarily register forward pre-hooks, removing them on exit."""
    handles = [module.register_forward_pre_hook(hook) for module, hook in pre_hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# %%

from section3_test import test_hook_cache_mean_resid

tokens = format_instructions(prompts)

resid_cache = {}
def hook_cache_mean_resid(module, args: tuple):
    (resid,) = args    
    resid_cache['resid_last_mean'] = resid[:, -1, :].mean(dim=0)
    return None

test_hook_cache_mean_resid(hook_cache_mean_resid)
# %%
@torch.no_grad()
def get_mean_activations(
    prompts: list[str], layer: int = LAYER
) -> Float[Tensor, "d_model"]:
    # TODO:
    # 1. Tokenize `prompts` with format_instructions.
    # 2. Copy hook_cache_mean_resid, register it on model.model.layers[layer].
    # 3. Run the model inside `with use_hooks(...)`, then return resid_cache["resid_last_mean"].
    tokens = format_instructions(prompts)

    resid_cache = {}
    def my_hook_cache_mean_resid(module, args: tuple):
        (resid,) = args    
        resid_cache['resid_last_mean'] = resid[:, -1, :].mean(dim=0)
        return None

    hook_locations = [(model.model.layers[layer], my_hook_cache_mean_resid)]
    
    with use_hooks(hook_locations):
        model(**tokens)

    # runs the model as standard without the hook functions
    model(**tokens)
    return resid_cache['resid_last_mean']


harmful_means = get_mean_activations(HARMFUL_INSTRUCTIONS)
harmless_means = get_mean_activations(HARMLESS_INSTRUCTIONS)
print("mean-activation tensor shape:", harmful_means.shape)

# The refusal direction: the difference-in-means vector, pointing from harmless → harmful
# activations. This single vector is what Sections 2-4 ablate, add, and bake into the weights.
refusal_dir = harmful_means - harmless_means
from section3_test import test_get_mean_activations


test_get_mean_activations(get_mean_activations)
# %%

import numpy as np
import matplotlib.pyplot as plt


# Short capture helper for the plot: like get_mean_activations, but keeps every prompt's vector
# instead of averaging. One forward pass over the whole list.
@torch.no_grad()
def get_last_token_activations(prompts: list[str], layer: int = LAYER) -> Tensor:
    """One last-token residual-stream vector per prompt at `layer`. Shape [n_prompts, D_MODEL]."""
    cache = {}

    def hook(module, args):
        cache["acts"] = args[0][:, -1, :].float().cpu()  # (n_prompts, d_model)

    with use_hooks([(model.model.layers[layer], hook)]):
        model(**format_instructions(prompts))
    return cache["acts"]


harmful_resid = get_last_token_activations(HARMFUL_INSTRUCTIONS)
harmless_resid = get_last_token_activations(HARMLESS_INSTRUCTIONS)

# Project every prompt's activation onto the (unit) refusal direction, one scalar per prompt.
unit = (refusal_dir / refusal_dir.norm()).float().cpu()
harmful_proj = (harmful_resid @ unit).numpy()
harmless_proj = (harmless_resid @ unit).numpy()

# A shared x-axis spanning both sets of projections.
lo = min(harmful_proj.min(), harmless_proj.min())
hi = max(harmful_proj.max(), harmless_proj.max())
xs = np.linspace(lo, hi, 200)

plt.figure(figsize=(7, 4))
for label, colour, proj in [
    ("harmful", "red", harmful_proj),
    ("harmless", "blue", harmless_proj),
]:
    mu, sigma = proj.mean(), proj.std()
    # Gaussian fitted to this group's projections (empirical mean and std).
    density = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    plt.plot(
        xs, density, color=colour, label=f"{label}  (mean={mu:.1f}, std={sigma:.1f})"
    )
    # The projected points themselves, as a rug along the axis.
    plt.scatter(proj, np.zeros_like(proj), color=colour, marker="|", s=200)
plt.xlabel("projection onto the refusal direction")
plt.ylabel("density")
plt.title(
    f"Last-token activations projected onto the refusal direction (layer {LAYER})"
)
plt.legend()
plt.tight_layout()
plt.savefig("refusal_projection.png", dpi=120)
plt.show()
# %%

# Combine prompts, projection values, and labels across harmful and harmless sets
all_prompts = HARMFUL_INSTRUCTIONS + HARMLESS_INSTRUCTIONS
all_projections = np.concatenate([harmful_proj, harmless_proj])
all_labels = ["harmful"] * len(HARMFUL_INSTRUCTIONS) + ["harmless"] * len(HARMLESS_INSTRUCTIONS)

prompt_data = list(zip(all_prompts, all_projections, all_labels))

# 1. 5 of the highest on the refusal vector
print("=== 5 Highest on the Refusal Vector ===")
highest_5 = sorted(prompt_data, key=lambda x: x[1], reverse=True)[:5]
for prompt, proj, label in highest_5:
    print(f"[{label}] (proj = {proj:.2f}): {prompt}")

# 2. 5 sitting near the 280 side of the vector projection value
print("\n=== 5 Near 280 on the Refusal Vector ===")
near_280 = sorted(prompt_data, key=lambda x: abs(x[1] - 280))[:5]
for prompt, proj, label in near_280:
    print(f"[{label}] (proj = {proj:.2f}): {prompt}")

# 3. 5 near the 150 refusal vector projection value
print("\n=== 5 Near 150 on the Refusal Vector ===")
near_150 = sorted(prompt_data, key=lambda x: abs(x[1] - 150))[:5]
for prompt, proj, label in near_150:
    print(f"[{label}] (proj = {proj:.2f}): {prompt}")

# 4. 5 near the 80 refusal vector projection value
print("\n=== 5 Near 80 on the Refusal Vector ===")
near_80 = sorted(prompt_data, key=lambda x: abs(x[1] - 80))[:5]
for prompt, proj, label in near_80:
    print(f"[{label}] (proj = {proj:.2f}): {prompt}")

# %%
