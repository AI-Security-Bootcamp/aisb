# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from aisb_utils import report
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import StoppingCriteria
from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

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




@report
def test_hook_cache_mean_resid(solution: Callable):
    torch.manual_seed(0)
    x = torch.randn(4, 5, D_MODEL)  # a synthetic residual stream: (batch, seq, d_model)
    # The hook stashes its result in the `resid_cache` dict living in its own module scope; reach
    # that dict through the hook's globals so this works wherever the hook is defined.
    cache = solution.__globals__["resid_cache"]
    cache.pop("resid_last_mean", None)

    out = solution(None, (x,))  # the module argument is unused by the hook
    assert "resid_last_mean" in cache, (
        "Hook should store its result in resid_cache['resid_last_mean']"
    )
    assert out is None, (
        "A capture hook should return None (it reads the stream, it must not modify it)"
    )

    got = cache["resid_last_mean"]
    assert got.shape == (D_MODEL,), (
        f"Expected shape {(D_MODEL,)}, got {tuple(got.shape)}"
    )
    # It must be the LAST token position, averaged over the batch.
    assert torch.allclose(got, x[:, -1, :].mean(dim=0), atol=1e-5), (
        "Should be the last-token vector averaged over the batch dimension"
    )
    # Guard against the common slips: using the first token, or averaging over positions too.
    assert not torch.allclose(got, x[:, 0, :].mean(dim=0), atol=1e-5), (
        "Use the last token, not the first"
    )
    assert not torch.allclose(got, x.mean(dim=(0, 1)), atol=1e-5), (
        "Average over the batch only, not over the sequence positions"
    )
    print("  All tests passed!")




@report
def test_get_mean_activations(solution: Callable):
    resid = solution(HARMLESS_INSTRUCTIONS[:4])
    assert resid.shape == (D_MODEL,), (
        f"Expected shape {(D_MODEL,)}, got {tuple(resid.shape)}"
    )
    assert torch.isfinite(resid).all(), "Activations should be finite"
    assert resid.norm() > 0, "Activations should be non-zero"
    # For a single prompt, the "mean" is just that prompt's last-token activation. Check the
    # returned vector against a manual capture at LAYER (same batch size, so bf16 matches).
    p = HARMLESS_INSTRUCTIONS[0]
    one = solution([p])
    grabbed = {}

    def cap(module, args):
        grabbed["x"] = args[0][:, -1, :][0].double()

    handle = model.model.layers[LAYER].register_forward_pre_hook(cap)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    with torch.no_grad():
        model(**tokenizer(prompt, return_tensors="pt").to(DEVICE))
    handle.remove()
    assert torch.allclose(one, grabbed["x"].to(one), atol=1e-2), (
        "Should be the last-token activation at LAYER"
    )
    # Harmful and harmless prompts must produce different activations (the premise of the method).
    diff = (solution(HARMFUL_INSTRUCTIONS[:4]) - resid).norm().item()
    assert diff > 0.1, (
        f"Harmful vs harmless activations should differ (got norm {diff:.3f})"
    )
    print("  All tests passed!")




@report
def test_oproj(solution: Callable):
    torch.manual_seed(0)
    d = torch.randn(8)
    x = torch.randn(4, 3, 8)  # [batch, seq, d_model]
    out = solution(x, d)
    unit = d / d.norm()
    # Result must be orthogonal to the direction ...
    assert torch.allclose((out * unit).sum(-1), torch.zeros(4, 3), atol=1e-5), (
        "Result must be orthogonal to direction"
    )
    # ... and equal to x minus its projection (i.e. unchanged in the orthogonal complement).
    expected = x - (x * unit).sum(-1, keepdim=True) * unit
    assert torch.allclose(out, expected, atol=1e-5), (
        "Should subtract exactly the projection"
    )
    # Idempotent: projecting again changes nothing.
    assert torch.allclose(solution(out, d), out, atol=1e-5), (
        "Projection should be idempotent"
    )
    # Projecting the direction itself out leaves ~0.
    assert solution(d, d).norm() < 1e-5, (
        "Projecting the direction onto its own complement should be ~0"
    )
    # A vector already orthogonal to d is unchanged.
    v = torch.randn(8)
    v = v - (v @ unit) * unit
    assert torch.allclose(solution(v, d), v, atol=1e-5), (
        "Vectors orthogonal to the direction must be unchanged"
    )
    # The direction need not be unit length (it is normalized internally).
    assert torch.allclose(solution(x, d), solution(x, 10.0 * d), atol=1e-5), (
        "Result must not depend on |direction|"
    )
    print("  All tests passed!")




@report
def test_get_ablation_hooks(solution: Callable):
    torch.manual_seed(0)
    d = torch.randn(D_MODEL)
    hooks = solution(d)
    assert len(hooks) == N_LAYERS, (
        f"Expected one hook per layer ({N_LAYERS}), got {len(hooks)}"
    )
    unit = d / d.norm()
    x = torch.randn(2, 3, D_MODEL)
    # Every hook must be attached to the corresponding layer and project the direction out.
    for i in range(0, N_LAYERS, max(1, N_LAYERS // 4)):
        module, hook = hooks[i]
        assert module is model.model.layers[i], (
            f"Hook {i} should be attached to layer {i}"
        )
        out = hook(module, (x,))[0]
        assert torch.allclose((out * unit).sum(-1), torch.zeros(2, 3), atol=1e-4), (
            f"After hook {i}, the activation must be orthogonal to the direction"
        )
    print("  All tests passed!")



@report
def test_get_steering_hook(solution: Callable):
    # Correctness on a synthetic activation: the hook must add exactly coeff*direction.
    d = torch.ones(D_MODEL)
    hooks = solution(d, coeff=2.0)
    assert len(hooks) == 1, "Steering resid at a single layer"
    module, hook = hooks[0]
    # Use batch > 1: a hook that returns tuple(tensor) instead of (tensor,) unpacks the batch
    # dimension and passes the wrong shape on, which only shows up when batch != 1.
    x = torch.randn(2, 4, D_MODEL)
    result = hook(module, (x,))
    assert isinstance(result, tuple) and len(result) == 1, (
        "Hook must return a one-element tuple (the modified residual), not the unpacked tensor"
    )
    out = result[0]
    assert out.shape == x.shape, (
        f"Hook must preserve the activation shape {tuple(x.shape)}, got {tuple(out.shape)}"
    )
    assert torch.allclose(out, x + 2.0 * d.to(x)), (
        "Hook should add coeff*direction to the activation"
    )
    # It must act at the requested layer ...
    assert solution(d, coeff=1.0, layer=3)[0][0] is model.model.layers[3], (
        "Steering should act at the requested layer"
    )
    # ... scale linearly with coeff ...
    delta1 = solution(d, coeff=1.0)[0][1](module, (x,))[0] - x
    delta3 = solution(d, coeff=3.0)[0][1](module, (x,))[0] - x
    assert torch.allclose(delta3, 3.0 * delta1, atol=1e-5), (
        "The added vector should scale with coeff"
    )
    print("  All tests passed!")




@report
def test_abliterate_model(solution: Callable):
    # Build a tiny stand-in with the same attribute structure as the real model, so we can
    # check the weight edit without mutating (or needing) the full 1.7B model.
    D = 16
    torch.manual_seed(0)

    def make_block():
        return types.SimpleNamespace(
            self_attn=types.SimpleNamespace(o_proj=torch.nn.Linear(4, D, bias=False)),
            mlp=types.SimpleNamespace(down_proj=torch.nn.Linear(7, D, bias=False)),
        )

    fake = types.SimpleNamespace(
        model=types.SimpleNamespace(
            embed_tokens=torch.nn.Embedding(6, D), layers=[make_block(), make_block()]
        )
    )
    d = torch.randn(D)
    unit = d / d.norm()
    emb_shape = fake.model.embed_tokens.weight.shape
    solution(fake, d)
    assert fake.model.embed_tokens.weight.shape == emb_shape, (
        "Weight shapes must be preserved"
    )
    assert (fake.model.embed_tokens.weight.data @ unit).abs().max() < 1e-4, (
        "Embeddings not orthogonalized"
    )
    for blk in fake.model.layers:
        assert (blk.self_attn.o_proj.weight.data.T @ unit).abs().max() < 1e-4, (
            "o_proj not orthogonalized"
        )
        assert (blk.mlp.down_proj.weight.data.T @ unit).abs().max() < 1e-4, (
            "down_proj not orthogonalized"
        )
    # Idempotent: re-applying it leaves the (already-orthogonalized) weights unchanged.
    emb_before = fake.model.embed_tokens.weight.data.clone()
    solution(fake, d)
    assert torch.allclose(fake.model.embed_tokens.weight.data, emb_before, atol=1e-5), (
        "Abliteration should be idempotent"
    )
    print("  All tests passed!")




@report
def test_mmlu_record_to_sample(solution: Callable):
    s = solution(
        {"question": "2 + 2 = ?", "choices": ["3", "4", "5", "6"], "answer": 1}
    )
    assert s.input == "2 + 2 = ?", "input should be the question text"
    assert list(s.choices) == ["3", "4", "5", "6"], (
        "choices should be carried through in order"
    )
    assert s.target == "B", f"answer index 1 should map to letter 'B', got {s.target!r}"
    # Index 0 -> "A" and the last index -> the last letter.
    assert (
        solution({"question": "q", "choices": ["a", "b", "c", "d"], "answer": 0}).target
        == "A"
    )
    assert (
        solution({"question": "q", "choices": ["a", "b", "c", "d"], "answer": 3}).target
        == "D"
    )
    print("  All tests passed!")
