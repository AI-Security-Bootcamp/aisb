# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Tuple, List


@report
def test_build_attack_batch(solution):
    """requires: GPU (uses the tokenizer/model device from setup).

    Each tokenized example must be a triple of 1-D token-id tensors
    (tokens-before-prefix, tokens-after-prefix, target).
    """
    batch = solution(tokenizer, attack_pairs, device)
    assert len(batch) == len(attack_pairs), (
        f"Expected {len(attack_pairs)} examples, got {len(batch)}"
    )
    for i, example in enumerate(batch):
        assert len(example) == 3, f"Example {i} should be a 3-tuple, got length {len(example)}"
        before_ids, after_ids, tgt_ids = example
        assert before_ids.ndim == 1, f"Example {i}: prompt-before ids should be 1-D, got {before_ids.ndim}-D"
        assert after_ids.ndim == 1, f"Example {i}: prompt-after ids should be 1-D, got {after_ids.ndim}-D"
        assert tgt_ids.ndim == 1, f"Example {i}: target ids should be 1-D, got {tgt_ids.ndim}-D"
        assert tgt_ids.shape[0] > 0, f"Example {i}: target should be non-empty"
    print("  All tests passed!")




@report
def test_initialize_latent_prefix(solution):
    """requires: GPU (reads the model's embedding dimension / device).

    The latent prefix must be a (prefix_length, embed_dim) float32 tensor on the
    model's device, so Adam has the precision it needs during optimization.
    """
    prefix_length = 7
    prefix = solution(model, prefix_length=prefix_length, device=device)
    embed_dim = model.get_input_embeddings().weight.shape[1]
    assert prefix.shape == (prefix_length, embed_dim), (
        f"Expected shape {(prefix_length, embed_dim)}, got {tuple(prefix.shape)}"
    )
    assert prefix.dtype == torch.float32, f"Expected float32, got {prefix.dtype}"
    assert prefix.device.type == device.type, (
        f"Expected device {device.type}, got {prefix.device.type}"
    )
    print("  All tests passed!")




@report
def test_latent_target_loss(solution):
    """requires: GPU (runs forward passes through the model).

    On a single-example batch, the shared-prefix loss must equal a direct,
    hand-computed cross-entropy over the same target tokens. We use a
    zero-length prefix so we can reconstruct the exact forward pass by hand.
    """
    before_ids, after_ids, tgt_ids = attack_batch[0]
    empty_prefix = initialize_latent_prefix(model, prefix_length=0, device=device)

    loss = solution(model, [(before_ids, after_ids, tgt_ids)], empty_prefix)
    assert loss.ndim == 0, f"Expected a scalar loss, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss) and loss.item() > 0, f"Loss must be finite and positive, got {loss.item()}"

    # Hand-computed reference: with an empty prefix the input is just the token ids.
    full_ids = torch.cat([before_ids, after_ids, tgt_ids]).unsqueeze(0)
    context_length = before_ids.shape[0] + after_ids.shape[0]
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits
    target_logits = logits[0, context_length - 1 : -1, :]
    expected = F.cross_entropy(target_logits, tgt_ids).item()

    assert abs(loss.item() - expected) < 1e-2, (
        f"latent_target_loss ({loss.item():.4f}) does not match the hand-computed "
        f"single-example cross-entropy ({expected:.4f})"
    )
    print("  All tests passed!")




@report
def test_optimize_latent_prefix(solution):
    """requires: GPU (runs the Adam optimization loop over the model).

    Note: `final_loss <= initial_loss` is essentially tautological for gradient
    descent on this objective. Instead we require the optimizer to make a
    *substantive* dent in the batch-mean target loss over the given steps, and
    to return a loss history of the expected length.
    """
    start_prefix = initialize_latent_prefix(model, prefix_length=32, device=device)
    steps = 50
    optimized, history = solution(model, attack_batch, start_prefix, steps=steps, lr=5e-5)

    assert optimized.shape == start_prefix.shape, (
        f"Optimized prefix changed shape: {tuple(optimized.shape)} vs {tuple(start_prefix.shape)}"
    )
    assert len(history) == steps, f"Expected {steps} recorded losses, got {len(history)}"

    reduction = (history[0] - history[-1]) / history[0]
    assert reduction > 0.05, (
        f"Optimization barely reduced the loss (initial={history[0]:.3f}, "
        f"final={history[-1]:.3f}, reduction={reduction:.1%}); expected >5%"
    )
    print("  All tests passed!")
