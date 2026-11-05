"""Export validated ablations through the existing acquisition/bundle boundary.

Run on the archive host after Slurm releases the GPU. Adds attention/MLP CUDA
spans and exact 0.1 ms min/max tracks to the standard RMS + execution trace.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import numpy as np

TITLES = {
    "efficient": "Attention comparison · memory-efficient baseline",
    "flash": "Attention comparison · FlashAttention",
    "eager": "Attention comparison · eager attention",
    "no-attention": "Ablation · attention branches bypassed",
    "no-mlp": "Ablation · MLP branches bypassed",
    "no-blocks": "Ablation · attention and MLP branches bypassed",
}


def branch_spans(exporter, events, prefix, pid, rebase):
    renamed = []
    for event in events:
        name = event.get("name", "")
        if event.get("cat") == "user_annotation":
            if name.startswith(prefix + " ") and name.endswith(" forward"):
                event = dict(event, name="Layer " + name[len(prefix) + 1:])
            elif name.startswith("Layer "):
                continue
        renamed.append(event)
    spans = exporter.gpu_layers(renamed)
    for event in spans:
        event.update(pid=pid, name=event["name"].replace("Decoder block", prefix),
                     ts=round(event["ts"] - rebase, 3))
    return spans


def export_one(root, variant, capture_tools):
    source = root / "runs" / variant
    metadata = json.loads((source / "metadata.json").read_text())
    validation = json.loads((source / "variant-validation.json").read_text())
    if metadata["status"] != "completed" or metadata["overflow_mask"]:
        raise ValueError(f"Invalid capture: {variant}")
    identity = "attn-" + variant + "-20261010"
    target = root / "bundles" / identity
    subprocess.run([
        "/home/tenant/envs/pico/bin/python", str(capture_tools / "export.py"),
        str(source), "--out", str(target), "--title", TITLES[variant],
        "--alignment", str(capture_tools.parent / "instructor/timing-profile.json"),
    ], check=True)
    spec = importlib.util.spec_from_file_location("archive_export", capture_tools / "export.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    manifest = json.loads((target / "manifest.json").read_text())
    original = json.loads((source / "torch-trace.json").read_text())
    events = original["traceEvents"]
    rebase = math.floor(min(e["ts"] for e in events if "ts" in e) / 1000) * 1000
    kernels = [e for e in events if e.get("cat") == "kernel"]
    capture_zero_us = manifest["perfetto"]["capture_zero_trace_ms"] * 1000
    uncertainty_us = manifest["timing"].get("reference_uncertainty_us", 100)
    if (min(e["ts"] for e in kernels) - rebase < capture_zero_us + uncertainty_us
            or max(e["ts"] + e["dur"] for e in kernels) - rebase
            > capture_zero_us + manifest["duration_ms"] * 1000 - uncertainty_us - 5000):
        raise ValueError("CUDA work is not contained within the calibrated capture")

    with gzip.open(target / "trace.json.gz", "rt") as stream:
        trace = json.load(stream)
    out = trace["traceEvents"]
    with np.load(target / "rms.npz", allow_pickle=False) as rms:
        # Reuse the original 10 us smoothing window at every 1 us output bin.
        # Keep the acquisition clock; narrower smoothing does not recalibrate it.
        for time_ms, value in zip(rms["time_ms"], rms["rms_0.01ms_A"]):
            out.append(dict(ph="C", pid=900000, tid=0,
                            ts=round(capture_zero_us + float(time_ms) * 1000, 3),
                            name="AC current RMS 0.01 ms (A)", args={"value": float(value)}))
    branch_counts = {}
    for prefix, pid in (("Attention", 900003), ("MLP", 900004)):
        spans = branch_spans(exporter, events, prefix, pid, rebase)
        branch_counts[prefix] = len(spans)
        expected = 0 if validation["skip_attention" if prefix == "Attention" else "skip_mlp"] else 28
        assert len(spans) == expected, (variant, prefix, len(spans), expected)
        if spans:
            out.extend([
                dict(ph="M", pid=pid, tid=1, name="process_name", args={"name": "GPU " + prefix + " branches"}),
                dict(ph="M", pid=pid, tid=1, name="thread_name", args={"name": prefix + " forward"}),
                *spans,
            ])
    gain = manifest["current_A_per_adc_count"]
    raw = np.load(target / "channel-a-adc.npy", mmap_mode="r", allow_pickle=False)
    stride = round(0.1e-3 / metadata["interval_s"])
    assert len(raw) % stride == 0
    blocks = raw.reshape(-1, stride)
    low, high = blocks.min(axis=1) * gain, blocks.max(axis=1) * gain
    for index, (minimum, maximum) in enumerate(zip(low, high)):
        timestamp = capture_zero_us + (index + 0.5) * 100
        for name, value in (("Minimum", minimum), ("Maximum", maximum)):
            out.append(dict(ph="C", pid=900000, tid=0, ts=round(timestamp, 3),
                            name=f"AC current {name.lower()} per 0.1 ms (A)", args={"value": float(value)}))

    operations = Counter(e["name"] for e in events if e.get("cat") == "cpu_op")
    expected_op = {
        "efficient": "aten::_scaled_dot_product_efficient_attention",
        "flash": "aten::_scaled_dot_product_flash_attention",
    }
    if variant in ("efficient", "flash", "no-mlp"):
        name = expected_op["efficient" if variant == "no-mlp" else variant]
        assert operations[name] == 28, (variant, name, operations[name])
    elif variant in ("no-attention", "no-blocks"):
        assert operations["aten::scaled_dot_product_attention"] == 0
    elif variant == "eager":
        assert operations["aten::scaled_dot_product_attention"] == 0
        assert operations["aten::_softmax"] == 28

    phases = {}
    for phase in manifest["phases"]:
        if phase["phase"] not in ("forward", "backward", "optimizer"):
            continue
        # Phase RMS uses the raw signal; it is an AC-current statistic, not DC power.
        first = math.ceil(phase["capture_start_ms"] * 1e-3 / metadata["interval_s"])
        end = math.floor(phase["capture_end_ms"] * 1e-3 / metadata["interval_s"])
        energy, count = 0.0, 0
        for start in range(first, end, 2_000_000):
            values = raw[start:min(end, start + 2_000_000)].astype(np.float64)
            energy += float(values @ values)
            count += len(values)
        phases[phase["phase"]] = dict(
            duration_ms=phase["capture_end_ms"] - phase["capture_start_ms"],
            current_rms_A=math.sqrt(energy / count) * gain,
        )
    experiment = dict(
        variant=variant, model="Qwen/Qwen3-0.6B", layers=28, batch=8, sequence_length=512,
        precision="BF16", seed=metadata["seed"], loss=metadata["loss"],
        skip_attention=validation["skip_attention"], skip_mlp=validation["skip_mlp"],
        ablation_semantics=validation["ablation_semantics"],
        input_tokens_sha256=validation["input_tokens_sha256"],
        common_optimizer_warmup_steps=1, additional_kernel_warmup_without_update=1,
        gradients=validation["gradients"], branch_spans=branch_counts,
        attention_ops={k: v for k, v in operations.items() if "attention" in k.lower()},
        phases=phases, torch_version=metadata["torch_version"],
        transformers_version=metadata["transformers_version"],
    )
    trace["metadata"].update(experiment=experiment, minmax_window_ms=0.1,
                             perfetto_rms_windows_ms=[0.01, 0.1])
    with gzip.open(target / "trace.json.gz", "wt", compresslevel=4) as stream:
        json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
    manifest["experiment"] = experiment
    manifest["perfetto"]["envelope_tracks_included"] = True
    manifest["perfetto"]["rms_windows_ms"] = [0.01, 0.1]
    manifest["description"] = (f"Qwen3-0.6B, 28 layers, batch 8, sequence 512, BF16; {variant}. "
                               "One forward/backward/optimizer step. RCP120XS, 50 mV/A; measured AC current.")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exported": identity, "phases": phases, "branches": branch_counts}), flush=True)
    return {"capture": identity, **experiment}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--capture-tools", type=Path, default=Path('/home/tenant/experiments/pico-access/workshop/capture'))
    parser.add_argument("--variants", nargs="+", default=list(TITLES))
    args = parser.parse_args()
    results = [export_one(args.root, variant, args.capture_tools) for variant in args.variants]
    (args.root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
