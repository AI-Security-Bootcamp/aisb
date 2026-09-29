"""Capture one training step with PyTorch profiling and the oscilloscope.

Edit the settings below, then run on the prepared GPU/PicoSDK host:
  python train.py
  python merge.py  # Set RUN in merge.py to match OUTPUT below.
The merger includes a September 24 calibration estimate for this H200/Pico at
0.4 ns / 420 ms. Recalibrate if the hardware or acquisition settings change.
Requires CUDA PyTorch and transformers; capture_scope also needs numpy and PicoSDK.
"""

import functools
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM

from capture_scope import Scope

MODEL = "Qwen3-0.6B"
OUTPUT = Path("/tmp/power-trace-pranav")  # replace with your name
BATCH_SIZE = 8
SEQUENCE_LENGTH = 512
SEED = 4158
DURATION_S = 0.42
INTERVAL_NS = 0.4

def warmup(model, optimizer):
    tokens = torch.randint(model.config.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH)).to("cuda")
    output = model(input_ids=tokens, labels=tokens, use_cache=False)
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def main():
    torch.manual_seed(SEED)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL
    ).to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

    # Optional: Warm up CUDA and allocate optimizer state before recording.
    warmup(model, optimizer)

    # Qwen3/Granite decoder layout. Adapt this line for another model architecture.
    for index, layer in enumerate(model.model.layers):
        def wrap(forward, index):
            @functools.wraps(forward)
            def annotated(*a, **kw):
                with record_function(f"Layer {index:02d} forward"):
                    return forward(*a, **kw)
            return annotated
        layer.forward = wrap(layer.forward, index)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        torch.cuda.synchronize()
        scope = Scope(OUTPUT, DURATION_S, INTERVAL_NS)
        with scope:
            time.sleep(0.02)  # An idle baseline before the measured step.
            with record_function("Step data_loading"):
                # Synthetic CPU batch preparation; no dataset or disk loader needed.
                tokens = torch.randint(model.config.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH)).pin_memory()
            with record_function("Step host_to_device"):
                tokens = tokens.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
            with record_function("Step zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with record_function("Step forward"):
                output = model(input_ids=tokens, labels=tokens, use_cache=False)
                torch.cuda.synchronize()
            with record_function("Step backward"):
                output.loss.backward()
                torch.cuda.synchronize()
            with record_function("Step optimizer"):
                optimizer.step()
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(OUTPUT / "torch.json"))
    print(f"Saved {OUTPUT}; loss = {output.loss.item():g}")


if __name__ == "__main__":
    main()
