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
# %%
# my_hook_locations = [(model.model.layers[1], hook1), (model.model.layers[2], hook2)]

@contextlib.contextmanager
def use_hooks(pre_hooks=()):
    """Temporarily register forward pre-hooks, removing them on exit."""
    handles = [module.register_forward_pre_hook(hook) for module, hook in pre_hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# runs the model with the hook functions enabled
# with use_hooks(my_hook_locations):
#     model(**demo_tokens)

# # runs the model as standard without the hook functions
# model(**demo_tokens)
# %%

resid_cache = {}
def hook_cache_mean_resid(module, args: tuple):
    # args is a tuple of all arguments, and the only argument is the residual stream : (batch, seq, d_model)
    (resid,) = args
    resid_last_tok = resid[:, -1, :]  # Get the last token's residual stream
    resid_cache['resid_last_mean'] = resid_last_tok.mean(dim=0)

from section3_test import test_hook_cache_mean_resid


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
    
    def hook_cache_mean_resid(module, args: tuple):
        # args is a tuple of all arguments, and the only argument is the residual stream : (batch, seq, d_model)
        (resid_stream,) = args
        # extract out the last token position
        resid_stream_last_token = resid_stream[:, -1, :]  # (batch, d_model)
        # average over the batch dimension and store (one forward pass sees every prompt at once)
        resid_cache["resid_last_mean"] = resid_stream_last_token.mean(
            dim=0
        )  # (d_model)

    pre_hooks = [(model.model.layers[layer], hook_cache_mean_resid)]

    with use_hooks(pre_hooks):
        model(**tokens)
        # equivalent to model(input_ids=input_ids, attention_mask=attention_mask)
    return resid_cache["resid_last_mean"]

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

def oproj(x: Float[Tensor, "... d_model"], r: Float[Tensor, "d_model"]) -> Tensor:
    """Remove the component of `x` along `r` (operates on the last dim; any leading shape)."""
    # TODO:
    # 1. Normalize direction to unit length: r_hat = r / r.norm().
    # 2. Compute the projection coefficient x · r_hat along the last dim (keepdim=True).
    # 3. Return a NEW tensor x - coeff * r_hat (don't modify x in place).
    r_hat = r / r.norm()
    coeff = torch.sum(x * r_hat, dim=-1, keepdim=True)
    return x - coeff * r_hat
    
from section3_test import test_oproj


test_oproj(oproj)
# %%
def get_ablation_hooks(direction: Tensor) -> list:
    '''Forward pre-hooks that project `direction` out of every layer's residual input.
    Returns a list [(layer, hook)] for every layer in the model.'''
    def hook(module, args):
        return (oproj(args[0], direction),)

    return [(model.model.layers[i], hook) for i in range(N_LAYERS)]
    pass
from section3_test import test_get_ablation_hooks


test_get_ablation_hooks(get_ablation_hooks)
# %%


from tqdm import tqdm
from transformers import StoppingCriteria

class GenerationProgressBar(StoppingCriteria):
    def __init__(self, max_new_tokens: int, desc: str = "generating"):
        self.bar = tqdm(total=max_new_tokens, desc=desc, leave=False)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.bar.update(1)
        if self.bar.n >= self.bar.total:
            self.bar.close()
        return False

@torch.no_grad()
def generate(instructions, pre_hooks=(), max_new_tokens=256):
    """Greedy-generate a completion for each instruction, with optional hooks applied."""
    enc = format_instructions(instructions).to(DEVICE)
    with use_hooks(pre_hooks):
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([GenerationProgressBar(max_new_tokens)]),
        )
    out = out[:, enc.input_ids.shape[1] :]
    return [
        text.strip()
        for text in tokenizer.batch_decode(out, skip_special_tokens=True)
    ]


print("\n=== Harmful prompts: baseline vs. ablation ===")
harmful_prompt = HARMFUL_INSTRUCTIONS[1] #tax fraud
pre_hooks = get_ablation_hooks(refusal_dir)

print(f"=== Prompt ===\n{harmful_prompt}")
print(f"=== BASELINE ===\n{generate(harmful_prompt)[0]}")
print(f"=== ABLATION ===\n{generate(harmful_prompt, pre_hooks=pre_hooks)[0]}")

# %%
def get_steering_hook(
    direction: Tensor, coeff: float = 1.0, layer: int = LAYER
) -> list:
    """A forward pre-hook that adds `coeff * direction` to the residual at `layer`."""
    # TODO: return [(model.model.layers[layer], hook)] where hook adds coeff*direction to the residual stream.
    def hook(module, args):
        return (coeff * direction + args[0],)

    return [(model.model.layers[layer], hook)]
from section3_test import test_get_steering_hook


test_get_steering_hook(get_steering_hook)
# %%
print("\n=== Harmless prompts: baseline vs. steering (add refusal direction) ===")
harmless_prompts = HARMLESS_INSTRUCTIONS[:1]  # a list (chocolate cake); generate expects a list
pre_hooks = get_steering_hook(refusal_dir, coeff=1.0)

print(f"=== Prompt ===\n{harmless_prompts[0]}")
print(f"=== BASELINE ===\n{generate(harmless_prompts)[0]}")
print(f"=== STEERING ===\n{generate(harmless_prompts, pre_hooks)[0]}")



# %%
def orthogonalize_matrix(matrix: Tensor, direction: Tensor) -> Tensor:
    """Remove `direction` from every row of `matrix` (last dim = D_MODEL). Same maths as oproj."""
    return oproj(matrix, direction)




def abliterate_model(model, direction: Tensor) -> None:
    """Permanently remove `direction` from all residual-stream write matrices, in place."""
    # TODO: orthogonalize embed_tokens.weight, and each layer's o_proj.weight.T and
    # down_proj.weight.T (transpose so the last dim is D_MODEL, then transpose back).
    model.model.embed_tokens.weight.data = orthogonalize_matrix(
                model.model.embed_tokens.weight.data, direction
            )
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = orthogonalize_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = orthogonalize_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T
from section3_test import test_abliterate_model


test_abliterate_model(abliterate_model)

# %%


# Abliterate `model` in place; this permanently removes refusal from the weights. Section 6
# compares it against the original, which it reloads fresh from HuggingFace (so we keep no copy here).
abliterate_model(model, refusal_dir)
print("\n=== After baking (NO hooks): the weights themselves are changed ===")
print(f"{harmful_prompt}\n==========================")
print(generate(harmful_prompt)[0])
# %%


from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

# %%

ABLITERATED_DIR = str(Path(__file__).resolve().parent / "abliterated_qwen3")
model.save_pretrained(ABLITERATED_DIR)
tokenizer.save_pretrained(ABLITERATED_DIR)

# `save_pretrained` records the dtype under a new "dtype" key; also write the legacy
# "torch_dtype" key so the loader restores bf16 (otherwise it defaults to float32 and uses 2x
# the memory).
_cfg_path = Path(ABLITERATED_DIR) / "config.json"
_cfg = json.loads(_cfg_path.read_text())
_cfg["torch_dtype"] = "bfloat16"
_cfg_path.write_text(json.dumps(_cfg, indent=2))

# Free the in-memory model and everything holding a reference to its layers.
del model, pre_hooks, refusal_dir, harmful_means, harmless_means
gc.collect()
torch.cuda.empty_cache()
print(f"Saved abliterated model to {ABLITERATED_DIR}")

# %%


def mmlu_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        choices=list(record["choices"]),
        target="ABCD"[record["answer"]],
    )

from section3_test import test_mmlu_record_to_sample


test_mmlu_record_to_sample(mmlu_record_to_sample)


MMLU_LIMIT = (
    50  # small so this runs in a couple of minutes; raise it for a tighter estimate
)


@task
def mmlu_task():
    return Task(
        dataset=hf_dataset(
            "cais/mmlu",
            name="all",
            split="test",
            sample_fields=mmlu_record_to_sample,
            limit=MMLU_LIMIT,
            shuffle=True,
            seed=0,
        ),
        solver=[multiple_choice()],
        scorer=choice(),
        config=GenerateConfig(max_tokens=32, max_connections=8),
    )


def run_eval(model_id: str) -> float:
    """Evaluate a HuggingFace model id/path on the MMLU task, returning its accuracy."""
    log = inspect_eval(
        mmlu_task(),
        model=model_id,
        # "full" shows a live progress bar over the questions (needs ipywidgets in a notebook);
        # it degrades to plain progress lines when run as a script. Use "none" to silence it.
        display="full",
        model_args={
            "device": DEVICE,
            "do_sample": False,
            "batch_size": 8,
            "enable_thinking": False,
        },
        log_dir=str(Path(__file__).resolve().parent / "inspect_logs"),
    )[0]
    for s in log.results.scores:
        if "accuracy" in s.metrics:
            return s.metrics["accuracy"].value
    return float("nan")

def standard_error(p: float, n: int) -> float:
    """Standard error of a proportion p estimated from n trials: sqrt(p*(1-p)/n)."""
    return math.sqrt(p * (1 - p) / n)

original_acc = run_eval("hf/Qwen/Qwen3-1.7B")
abliterated_acc = run_eval("hf/" + ABLITERATED_DIR)
random_acc = 0.25

print(f"\nMMLU ({MMLU_LIMIT} questions):")
print(
    f"  original    : {original_acc:.0%} ± {standard_error(original_acc, MMLU_LIMIT):.0%}"
)
print(
    f"  abliterated : {abliterated_acc:.0%} ± {standard_error(abliterated_acc, MMLU_LIMIT):.0%}"
)
print(f"  random guess: {random_acc:.0%}")
print("If the two intervals overlap, abliteration did not measurably cost capability.")

# %%
