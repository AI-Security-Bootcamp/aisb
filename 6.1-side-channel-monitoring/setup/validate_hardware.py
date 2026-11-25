"""Exercise reference + real GPU/scope acceptance test. Submit as a Slurm job."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import socket
import sys
import tempfile
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynvml

SECTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECTION))
import merge


def monitor(stop, rows, errors):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            rows.append({"unix_ns": time.time_ns(),
                         "driver_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                         "gpu_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu})
            stop.wait(.02)
    except Exception as error:
        errors.append(str(error))
    finally:
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path, help="Optional separate viewer data directory")
    parser.add_argument("--existing-run", type=Path, help="Analyze a saved acceptance run without using the GPU or scope")
    args = parser.parse_args()
    if not args.existing_run and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit this test with sbatch --gres=gpu:1,scope:1 --exclusive")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.existing_run:
        recording, = args.existing_run.glob("side-channel-*/profiled-01")
        with (args.existing_run / "nvidia-driver.csv").open() as stream:
            rows = [{"unix_ns": int(r["unix_ns"]), "driver_power_w": float(r["driver_power_w"]),
                     "gpu_utilization_percent": int(r["gpu_utilization_percent"])} for r in csv.DictReader(stream)]
        shutil.copy2(args.existing_run / "nvidia-driver.csv", args.output / "nvidia-driver.csv")
        config = Path("/home/tenant/experiments/pico-access/model/Qwen3-0.6B/config.json")
        expected_layers = json.loads(config.read_text())["num_hidden_layers"]
        errors = []
    else:
        os.environ["TMPDIR"] = str(args.output)
        tempfile.tempdir = None
        stop, rows, errors = threading.Event(), [], []
        thread = threading.Thread(target=monitor, args=(stop, rows, errors), daemon=True)
        thread.start()
        try:
            # Run all generated tests and the actual idle/training/profiler exercises.
            namespace = runpy.run_path(str(SECTION / "section1_reference.py"), run_name="__main__")
            expected_layers = namespace["model"].config.num_hidden_layers
            recording = namespace["OUTPUT"] / "profiled-01"
            merge.RUN = recording
            merge.INCLUDE_ENVELOPE = True
            merge.main()
        finally:
            stop.set()
            thread.join(timeout=10)
            with (args.output / "nvidia-driver.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["unix_ns", "driver_power_w", "gpu_utilization_percent"])
                writer.writeheader()
                writer.writerows(rows)
    if errors or not rows:
        raise RuntimeError(f"NVIDIA monitoring failed: {errors}")

    meta = json.loads((recording / "scope.json").read_text())
    settings = json.loads(Path("/etc/aisb/probe.json").read_text())
    assert meta["sensitivity_mV_per_A"] == settings["sensitivity_mV_per_A"], "Wrong probe gain"
    assert meta["status"] == "completed" and meta["overflow_mask"] == 0
    raw = np.load(recording / "scope.npy", mmap_mode="r")
    gain = meta["range_v"] / meta["adc_max"] / (meta["sensitivity_mV_per_A"] / 1000)
    energy, low, high = merge.reduce_samples(raw, round(1e-6 / meta["interval_s"]), gain)
    rms = merge.centered_rms(energy)
    with gzip.open(recording / "perfetto.json.gz", "rt") as stream:
        exported = json.load(stream)
    trace = json.loads((recording / "torch.json").read_text())
    origin_ns = exported["metadata"]["scope_origin_unix_ns"]
    scope_us = (origin_ns - trace["baseTimeNanoseconds"]) / 1000
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    spans = [event for event in exported["traceEvents"] if event.get("cat") == "gpu_layer_span"]
    assert kernels and len(spans) == expected_layers
    times = np.arange(len(rms)) / 1000
    first_ms = (min(event["ts"] for event in kernels) - scope_us) / 1000
    last_ms = (max(event["ts"] + event["dur"] for event in kernels) - scope_us) / 1000
    idle = (times >= 2) & (times < min(10, first_ms - 2))
    active = (times >= first_ms + 2) & (times < last_ms - 2)
    assert idle.any() and active.any(), "Missing baseline or active comparison window"
    idle_rms = float(np.median(rms[idle]))
    active_rms = float(np.median(rms[active]))
    ratio = active_rms / max(idle_rms, 1e-12)
    failures = []
    if ratio <= 3:
        failures.append(f"No clear current response during CUDA work: active/idle={ratio:.2f}")
    assert last_ms < len(raw) * meta["interval_s"] * 1000 - 5, "Missing trailing margin"
    assert max(row["gpu_utilization_percent"] for row in rows) > 0
    # Independent direct RMS check, including promotion before squaring int16 data.
    index = int((first_ms + last_ms) * 500)
    stride = round(1e-6 / meta["interval_s"])
    window = raw[(index - 50) * stride:(index + 50) * stride].astype(float) * gain
    np.testing.assert_allclose(rms[index], np.sqrt(np.mean(window * window)), rtol=1e-8)

    phases = [{"phase": event["name"].removeprefix("Step "),
               "capture_start_ms": (event["ts"] - scope_us) / 1000,
               "capture_end_ms": (event["ts"] + event["dur"] - scope_us) / 1000}
              for event in trace["traceEvents"]
              if event.get("cat") == "user_annotation" and event.get("name", "").startswith("Step ")]
    rebase = math.floor(min(scope_us, min(event["ts"] for event in trace["traceEvents"] if "ts" in event)) / 1000) * 1000
    timing_status = exported["metadata"]["status"]
    title = "probe check required" if failures else "verified training capture"
    summary = dict(schema_version=1, title=f"{socket.gethostname()} · {title}",
                   description=f"{settings['probe']}, {meta['sensitivity_mV_per_A']:g} mV/A; measured AC-current component.",
                   raw_file="channel-a-adc.npy", samples=len(raw), interval_ns=meta["interval_s"] * 1e9,
                   sample_rate_hz=1/meta["interval_s"], duration_ms=len(raw)*meta["interval_s"]*1000,
                   resolution_bits=meta["resolution_bits"], current_A_per_adc_count=gain,
                   probe=settings["probe"], sensitivity_mV_per_A=meta["sensitivity_mV_per_A"],
                   overflow_mask=0, phases=phases,
                   timing=dict(status=timing_status, origin_unix_ns=origin_ns,
                               display_note="Timing is uncalibrated on this scope; do not infer fine alignment." if timing_status == "uncalibrated" else "Transferred timing estimate; approximately ±0.1 ms reference uncertainty."),
                   perfetto=dict(file="trace.json.gz", capture_zero_trace_ms=(scope_us-rebase)/1000,
                                 layer_annotations=True, cuda_kernels=len(kernels), raw_tracks_included=False))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output / "analysis.npz", time_ms=times, rms_0_1ms_A=rms,
                        envelope_time_ms=times[::100] + .05,
                        minimum_A=np.minimum.reduceat(low, np.arange(0, len(low), 100)),
                        maximum_A=np.maximum.reduceat(high, np.arange(0, len(high), 100)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(times, rms, lw=.7)
    axes[0].set_ylabel("AC current RMS 0.1 ms (A)")
    nt = [(row["unix_ns"] - origin_ns) / 1e6 for row in rows]
    axes[1].plot(nt, [row["driver_power_w"] for row in rows], marker=".")
    axes[1].set_ylabel("NVIDIA driver power (W)")
    axes[1].set_xlabel("Time since estimated first scope sample (ms)")
    for phase in phases:
        if phase["phase"] in ("forward", "backward", "optimizer"):
            for ax in axes:
                ax.axvspan(phase["capture_start_ms"], phase["capture_end_ms"], alpha=.1)
            axes[0].text((phase["capture_start_ms"] + phase["capture_end_ms"])/2, .96,
                         phase["phase"], transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, summary["duration_ms"])
    fig.suptitle(f"{socket.gethostname()} · {settings['probe']} · {meta['sensitivity_mV_per_A']:g} mV/A\nOne training cycle; {timing_status} timing")
    fig.savefig(args.output / "validation.png", dpi=160)
    plt.close(fig)
    report = dict(status="failed" if failures else "passed", failures=failures,
                  host=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  probe=settings["probe"], sensitivity_mV_per_A=meta["sensitivity_mV_per_A"],
                  samples=len(raw), interval_ns=meta["interval_s"]*1e9, overflow_mask=0,
                  idle_median_rms_A=idle_rms, active_median_rms_A=active_rms, active_idle_ratio=ratio,
                  nvidia_driver_min_W=min(row["driver_power_w"] for row in rows),
                  nvidia_driver_peak_W=max(row["driver_power_w"] for row in rows),
                  peak_gpu_utilization_percent=max(row["gpu_utilization_percent"] for row in rows),
                  cuda_kernels=len(kernels), gpu_layer_spans=len(spans), timing=timing_status,
                  raw_recording=str(recording), output=str(args.output))
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.publish:
        # Publish only the approved visualization files, never acquisition source or credentials.
        destination = args.publish / "recordings" / "reference"
        traces = args.publish / "traces" / "reference"
        destination.mkdir(parents=True, exist_ok=False)
        traces.mkdir(parents=True, exist_ok=False)
        os.link(recording / "scope.npy", destination / "channel-a-adc.npy")
        shutil.copy2(args.output / "summary.json", destination / "summary.json")
        shutil.copy2(args.output / "summary.json", traces / "summary.json")
        shutil.copy2(recording / "perfetto.json.gz", traces / "trace.json.gz")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
