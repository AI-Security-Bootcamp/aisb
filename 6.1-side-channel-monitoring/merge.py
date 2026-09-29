"""Merge torch.json and scope.npy/scope.json into a compressed Perfetto trace.

Requires numpy; no GPU or SDK. Set RUN below, then run `python merge.py`.
Open RUN/perfetto.json.gz at ui.perfetto.dev.
The calibration is a separate September 24 estimate for this instrument/settings,
not a universal device constant. Set CALIBRATION = None for uncalibrated viewing.

Every ADC sample contributes to centered 0.1 ms RMS, output every 1 us.
INCLUDE_ENVELOPE adds extrema in 0.1 ms buckets. Raw samples stay in scope.npy.
The RCP120XS records nominal AC current, not total GPU DC power.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np

RUN = Path("/tmp/power-trace-pranav")  # Match OUTPUT in train.py.
INCLUDE_ENVELOPE = False
CALIBRATION = {
    "offset_us": 109.812, "uncertainty_us": 100,
    "interval_ns": 0.4, "duration_s": 0.42, "serial": "12669/0029",
}


def gpu_layers(events):
    """CPU layer -> operator External id -> CUDA kernels; never align by CPU duration."""
    layers = sorted((e for e in events if e.get("cat") == "user_annotation"
                     and e.get("name", "").startswith("Layer ")
                     and e["name"].endswith(" forward")), key=lambda e: e["ts"])
    operators = [e for e in events if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    spans = []
    for layer in layers:
        # Find CPU operators inside this layer, then follow their IDs to GPU execution.
        ids = {e["args"]["External id"] for e in operators
               if (e.get("pid"), e.get("tid")) == (layer.get("pid"), layer.get("tid"))
               and layer["ts"] <= e["ts"]
               and e["ts"] + e.get("dur", 0) <= layer["ts"] + layer["dur"] + 0.001}
        matched = [e for e in kernels if e.get("args", {}).get("External id") in ids]
        if not matched:
            raise ValueError(f"No CUDA kernels attributed to {layer['name']}")
        lo = min(e["ts"] for e in matched)
        hi = max(e["ts"] + e["dur"] for e in matched)
        spans.append(dict(ph="X", cat="gpu_layer_span", name=layer["name"], pid=900002, tid=1,
                          ts=lo, dur=hi-lo, args={"kernel_count": len(matched),
                          "meaning": "First to last attributed CUDA kernel; includes scheduling gaps"}))
    return spans


def reduce_samples(raw, stride, amperes_per_count):
    """Reduce to 1-us energy/extrema bins in bounded memory; include every sample."""
    if len(raw) % stride:
        raise ValueError("Recording must contain whole 1-us bins")
    n = len(raw) // stride
    energy, low, high = (np.empty(n, dtype=np.float64) for _ in range(3))
    for start in range(0, n, 2048):
        end = min(start + 2048, n)
        values = raw[start*stride:end*stride].astype(np.float64).reshape(-1, stride)
        low[start:end] = values.min(axis=1) * amperes_per_count
        high[start:end] = values.max(axis=1) * amperes_per_count
        values *= amperes_per_count
        np.square(values, out=values)
        energy[start:end] = values.mean(axis=1)
    return energy, low, high


def centered_rms(energy, width=100):
    """100 x 1-us bins: window [-50, +50) us; repeat edge-bin energy at ends."""
    padded = np.pad(energy, (width // 2, width - 1 - width // 2), mode="edge")
    total = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return np.sqrt(np.maximum((total[width:] - total[:-width]) / width, 0))


def main():
    out = RUN / "perfetto.json.gz"
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((RUN / "scope.json").read_text())
    if meta["status"] != "completed" or meta["overflow_mask"]:
        raise ValueError("Capture incomplete or clipped")
    raw = np.load(RUN / meta["raw_file"], mmap_mode="r")
    if raw.dtype != np.int16 or raw.ndim != 1 or len(raw) != meta["samples"]:
        raise ValueError("Raw samples disagree with scope metadata")
    stride = round(1e-6 / meta["interval_s"])
    if stride < 1 or not math.isclose(stride * meta["interval_s"], 1e-6, rel_tol=1e-8):
        raise ValueError("Sample interval must divide 1 us exactly")
    offset, uncertainty = 0, 0
    if CALIBRATION is not None:
        offset, uncertainty = CALIBRATION["offset_us"], CALIBRATION["uncertainty_us"]
        if not math.isfinite(offset) or not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("Invalid calibration offset/uncertainty")
        if (meta["serial"] != CALIBRATION["serial"]
                or not math.isclose(meta["interval_s"] * 1e9, CALIBRATION["interval_ns"], rel_tol=1e-8)
                or not math.isclose(len(raw) * meta["interval_s"], CALIBRATION["duration_s"], rel_tol=1e-8)):
            raise ValueError("Calibration does not match this instrument/acquisition")
    trace = json.loads((RUN / "torch.json").read_text())
    events = trace["traceEvents"]
    base = trace["baseTimeNanoseconds"]  # PyTorch timestamps are microseconds relative to this epoch.
    origin_ns = meta["start_return_unix_ns"] + round(offset * 1000)
    scope_us = (origin_ns - base) / 1000
    duration_us = len(raw) * meta["interval_s"] * 1e6
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("Trace contains no CUDA kernels")
    # Fail instead of presenting a truncated pass as a complete recording.
    if (min(e["ts"] for e in kernels) < scope_us + uncertainty
            or max(e["ts"] + e["dur"] for e in kernels) > scope_us + duration_us - uncertainty - 5000):
        raise ValueError("CUDA work does not fit inside the acquisition with a 5-ms tail")
    spans = gpu_layers(events)
    gain = meta["range_v"] / meta["adc_max"] / (meta["sensitivity_mV_per_A"] / 1000)
    energy, low, high = reduce_samples(raw, stride, gain)
    rms = centered_rms(energy)
    rebase = math.floor(min(scope_us, min(e["ts"] for e in events if "ts" in e)) / 1000) * 1000
    timing = {"status": "transferred-estimate" if CALIBRATION is not None else "uncalibrated",
              "calibration": CALIBRATION,
              "offset_from_runblock_return_us": offset if CALIBRATION is not None else None,
              "uncertainty_us": uncertainty if CALIBRATION is not None else None,
              "scope_origin_unix_ns": origin_ns, "raw_sample_interval_ns": meta["interval_s"] * 1e9,
              "rms_window_ms": 0.1, "rms_output_interval_us": 1,
              "rms_edge_padding": "nearest 1-us energy bin", "raw_tracks_included": False}
    partial = out.with_name(out.name + ".partial")
    with gzip.open(partial, "xt", compresslevel=4) as stream:
        stream.write('{"traceEvents":[')
        first = True

        def emit(event):
            nonlocal first
            event = dict(event)
            if "ts" in event:
                event["ts"] = round(event["ts"] - rebase, 3)
            stream.write(("" if first else ",") + json.dumps(event, separators=(",", ":"), allow_nan=False))
            first = False

        for event in events:
            emit(event)
        for pid, name in ((900000, f"AC current ({timing['status']} timing)"),
                          (900001, "Workload phases"), (900002, "GPU decoder layers")):
            emit(dict(ph="M", pid=pid, tid=1, name="process_name", args={"name": name}))
        for event in spans:
            emit(event)
        for event in events:
            if event.get("cat") == "user_annotation" and event.get("name", "").startswith("Step "):
                emit(dict(event, pid=900001, tid=1))

        def counter(name, index, value):
            emit(dict(ph="C", pid=900000, tid=1, name=name, ts=scope_us + index, args={"value": float(value)}))

        for i, value in enumerate(rms):
            counter("AC current RMS 0.1 ms (A)", i, value)
        if INCLUDE_ENVELOPE:
            for i in range(0, len(energy), 100):
                center = i + min(100, len(energy) - i) / 2
                counter("AC current minimum per 0.1 ms (A)", center, low[i:i+100].min())
                counter("AC current maximum per 0.1 ms (A)", center, high[i:i+100].max())
        stream.write('],"displayTimeUnit":"ms","metadata":' + json.dumps(timing) + "}")
    partial.rename(out)
    print(f"Saved {out}: {len(kernels)} CUDA kernels, {len(spans)} GPU layer spans, {len(rms)} RMS points; {timing['status']}")


if __name__ == "__main__":
    main()
