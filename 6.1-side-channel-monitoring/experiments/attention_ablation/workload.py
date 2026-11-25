"""Controlled Qwen3 attention/backbone ablations for the instrument-host runner.

Load with workshop/capture/capture.py --workload /path/to/this/file.py.
AISB_ATTENTION_VARIANT selects efficient, flash, eager, no-attention, no-mlp,
or no-blocks. Every condition uses the same initial checkpoint, generated data,
and one common memory-efficient-attention optimizer warm-up before switching.
"""
from contextlib import nullcontext
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import types

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
import transformers

VARIANTS = ("efficient", "flash", "eager", "no-attention", "no-mlp", "no-blocks")


class Tokens(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with gzip.open(self.paths[index], "rt") as stream:
            return torch.tensor(json.load(stream), dtype=torch.long)


def bypassed_decoder_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, use_cache=False, cache_position=None,
    position_embeddings=None, **kwargs,
):
    """Qwen3's two residual branches, with selected branches completely skipped.

    Identity means x -> x, including its gradient. Returning x from an attention
    or MLP module would instead make the outer residual addition double x.
    A removed branch also skips its own pre-normalization and residual addition.
    Embedding, final normalization, vocabulary head, and loss remain active.
    """
    if use_cache or past_key_value is not None:
        raise ValueError("These training ablations require use_cache=False")
    if not self._aisb_skip_attention:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = residual + hidden_states
    if not self._aisb_skip_mlp:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    return hidden_states


def annotate(forward, label):
    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return forward(*args, **kwargs)
    return wrapped


class Workload:
    def __init__(self, args):
        self.variant = os.environ.get("AISB_ATTENTION_VARIANT", "efficient")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").train()
        if self.model.config.model_type != "qwen3" or len(self.model.model.layers) != 28:
            raise ValueError("This experiment requires the unchanged Qwen3-0.6B baseline")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, fused=True)
        self.backend = "efficient"
        self.ready = False
        self.warmup_tokens = None
        self.output_dir = args.output
        self.truth = dict(
            model=str(args.model), layers=28, batch=args.batch,
            sequence_length=args.sequence_length, precision="BF16",
            optimizer="AdamW fused", seed=args.seed, variant=self.variant,
            attention="eager" if self.variant == "eager" else "sdpa",
            requested_sdpa_backend=(None if self.variant == "eager" else
                                    "flash" if self.variant == "flash" else "efficient"),
            skip_attention=self.variant in ("no-attention", "no-blocks"),
            skip_mlp=self.variant in ("no-mlp", "no-blocks"),
            ablation_semantics="Skip the entire selected residual branch, including its pre-norm; identity residual stream",
            common_optimizer_warmup_steps=1, additional_kernel_warmup_without_update=1,
            torch_version=torch.__version__, transformers_version=transformers.__version__,
            gpu=torch.cuda.get_device_name(0),
            flash_mask="No padding/cache/sliding window; use SDPA is_causal=True instead of materializing a causal mask",
        )
        directory = args.output / "inputs"
        directory.mkdir()
        rng = np.random.default_rng(args.seed)
        paths, values = [], []
        for index in range(2 * args.batch):
            tokens = rng.integers(0, self.model.config.vocab_size, args.sequence_length)
            path = directory / f"{index:03d}.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(tokens.tolist(), stream)
            paths.append(path)
            values.append(tokens)
        self.truth["input_tokens_sha256"] = hashlib.sha256(np.stack(values).astype("<i8").tobytes()).hexdigest()
        self.loader = iter(DataLoader(Tokens(paths), batch_size=args.batch, num_workers=0, pin_memory=True))

    def attention_context(self):
        if self.backend == "eager":
            return nullcontext()
        backend = SDPBackend.FLASH_ATTENTION if self.backend == "flash" else SDPBackend.EFFICIENT_ATTENTION
        # A single allowed backend makes an unsupported request fail explicitly.
        return sdpa_kernel(backend)

    def next_batch(self):
        return next(self.loader)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, tokens):
        if not self.ready:
            self.warmup_tokens = tokens.detach()
        options = {}
        if self.backend == "flash":
            if self.model.model.has_sliding_layers:
                raise ValueError("The FlashAttention comparison requires full causal attention")
            # Transformers 4.55 materializes a causal mask for this workload.
            # Flash SDPA rejects explicit masks. With our unpadded, fixed-length
            # batch and no KV cache, its is_causal=True path is equivalent.
            # Qwen3 accepts an already-prepared mask mapping; the SDPA adapter
            # sets is_causal=True when this entry is None and query length > 1.
            options["attention_mask"] = {"full_attention": None}
        with self.attention_context():
            return self.model(input_ids=tokens, labels=tokens, use_cache=False, **options)

    def backward(self, output):
        with self.attention_context():
            output.loss.backward()

    def step(self):
        self.optimizer.step()

    def gradient_inventory(self):
        result = {}
        for name, part in (("attention", "self_attn"), ("mlp", "mlp")):
            parameters = [p for layer in self.model.model.layers for p in getattr(layer, part).parameters()]
            result[name] = dict(parameters=sum(p.numel() for p in parameters),
                                parameters_with_grad=sum(p.numel() for p in parameters if p.grad is not None))
        return result

    def annotate_layers(self):
        self.backend = self.variant if self.variant in ("flash", "eager") else "efficient"
        self.model.set_attn_implementation("eager" if self.backend == "eager" else "sdpa")
        skip_attention, skip_mlp = self.truth["skip_attention"], self.truth["skip_mlp"]
        for index, layer in enumerate(self.model.model.layers):
            if skip_attention or skip_mlp:
                layer._aisb_skip_attention = skip_attention
                layer._aisb_skip_mlp = skip_mlp
                layer.forward = types.MethodType(bypassed_decoder_forward, layer)
            # No GPU layer span should be invented for a pure identity layer.
            prefix = "Bypassed decoder" if skip_attention and skip_mlp else "Layer"
            layer.forward = annotate(layer.forward, f"{prefix} {index:02d} forward")
            layer.self_attn.forward = annotate(layer.self_attn.forward, f"Attention {index:02d} forward")
            layer.mlp.forward = annotate(layer.mlp.forward, f"MLP {index:02d} forward")
        self.ready = True
        # Warm the selected implementation without changing the common weights
        # or AdamW state produced by the initial shared warm-up.
        output = self.forward(self.warmup_tokens)
        self.backward(output)
        torch.cuda.synchronize()
        if not torch.isfinite(output.loss).item():
            raise ValueError("Non-finite loss during implementation warm-up")
        inventory = self.gradient_inventory()
        for name, skipped in (("attention", skip_attention), ("mlp", skip_mlp)):
            assert (inventory[name]["parameters_with_grad"] == 0) == skipped, inventory
        (self.output_dir / "variant-validation.json").write_text(json.dumps({
            **self.truth, "warmup_loss": output.loss.item(), "gradients": inventory,
        }, indent=2) + "\n")
        self.zero_grad()
        del output
        self.warmup_tokens = None
        torch.cuda.synchronize()
