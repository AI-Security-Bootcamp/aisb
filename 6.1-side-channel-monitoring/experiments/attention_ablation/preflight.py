"""GPU-only preflight: one common warm-up, one variant warm-up, one profiled step."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from workload import Workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batch, args.sequence_length, args.seed = 8, 512, 20260922
    args.output.mkdir(parents=True, exist_ok=False)
    work = Workload(args)
    tokens = work.next_batch().to("cuda")
    output = work.forward(tokens)
    work.backward(output)
    work.step()
    torch.cuda.synchronize()
    common_loss = output.loss.item()
    work.zero_grad()
    del output, tokens
    work.annotate_layers()
    tokens = work.next_batch().to("cuda")
    torch.cuda.synchronize()
    elapsed = {}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for phase in ("forward", "backward", "optimizer"):
            begin = time.perf_counter()
            if phase == "forward":
                output = work.forward(tokens)
            elif phase == "backward":
                work.backward(output)
            else:
                work.step()
            torch.cuda.synchronize()
            elapsed[phase] = (time.perf_counter() - begin) * 1000
    prof.export_chrome_trace(str(args.output / "torch.json"))
    trace = json.loads((args.output / "torch.json").read_text())
    operations = Counter(e["name"] for e in trace["traceEvents"] if e.get("cat") == "cpu_op" and "attention" in e.get("name", "").lower())
    result = {**work.truth, "common_warmup_loss": common_loss, "loss": output.loss.item(),
              "duration_ms": elapsed, "attention_ops": dict(operations),
              "gradients": work.gradient_inventory()}
    result["logit_sample"] = output.logits[0, [0, 1, 127, 255, 511], ::4096].detach().float().cpu().tolist()
    if work.variant == "flash":
        # Verify that replacing an explicit mask with is_causal does not expose
        # later tokens to the prefix. Both evaluations use the same weights.
        changed = tokens.clone()
        changed[:, 256:] = (changed[:, 256:] + 17) % work.model.config.vocab_size
        with torch.no_grad():
            prefix = work.forward(tokens).logits[:, :128, ::4096].clone()
            other = work.forward(changed).logits[:, :128, ::4096]
        error = (prefix.float() - other.float()).abs().max().item()
        assert error <= 0.01, f"Future tokens affected the causal prefix: {error}"
        result["causal_prefix_max_abs_diff"] = error
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
