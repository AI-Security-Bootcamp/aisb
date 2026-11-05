"""Measure possible multi-second scope/workload lag with one continuous block.

Run inside a Slurm GPU + scope allocation. The slow native timebase aliases the
probe's high-frequency output: this is a timing diagnostic, not calibrated
broadband RMS or a measurement of total GPU DC power.
"""

import argparse
import csv
import ctypes as ct
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_scope import SIGNATURES, f64, i16, probe_settings, u32, u64


def timestamp():
    return {"monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()}


def monitor(stop, rows, errors):
    """Poll both NVIDIA power APIs, preserving device field timestamps."""
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        fields = {
            pynvml.NVML_VALUE_TYPE_DOUBLE: "dVal",
            pynvml.NVML_VALUE_TYPE_UNSIGNED_INT: "uiVal",
            pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG: "ulVal",
            pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: "ullVal",
            pynvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG: "sllVal",
        }
        while not stop.is_set():
            row = timestamp()
            row["driver_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            row["gpu_utilization_percent"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            field = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
            row["instant_status"] = field.nvmlReturn
            row["instant_timestamp_us"] = field.timestamp
            row["instant_query_latency_us"] = field.latencyUsec
            row["instant_power_w"] = (
                float(getattr(field.value, fields[field.valueType])) / 1000
                if field.nvmlReturn == pynvml.NVML_SUCCESS else None
            )
            rows.append(row)
            stop.wait(0.02)
    except Exception as error:
        errors.append(repr(error))
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def reduce_channels(raw, interval_s, adc_max, output):
    """Retain exact min/max and mean square of each native 10 ms sample bin."""
    bin_samples = max(1, round(0.01 / interval_s))
    complete = raw.shape[1] // bin_samples
    times = (np.arange(complete) + 0.5) * bin_samples * interval_s
    summary = {}
    for channel, label in enumerate("ABCD"):
        values = raw[channel, :complete * bin_samples].reshape(complete, bin_samples)
        scale_mV = 10_000 / adc_max
        summary[f"{label}_min_mV"] = values.min(axis=1) * scale_mV
        summary[f"{label}_max_mV"] = values.max(axis=1) * scale_mV
        summary[f"{label}_mean_mV"] = values.mean(axis=1) * scale_mV
        # At most one channel's float buffer is materialized at a time.
        squared = values.astype(np.float64) ** 2
        summary[f"{label}_rms_mV"] = np.sqrt(squared.mean(axis=1)) * scale_mV
        del squared
    np.savez_compressed(output / "reduced.npz", seconds=times, **summary)
    return times, summary


def analyse(raw, metadata, rows, output):
    interval_s = metadata["interval_s"]
    origin_ns = metadata["events"]["run_block_return"]["monotonic_ns"]
    times, reduced = reduce_channels(raw, interval_s, metadata["adc_max"], output)
    driver_times = np.array([(r["monotonic_ns"] - origin_ns) / 1e9 for r in rows])
    driver_watts = np.array([r["driver_power_w"] for r in rows])
    phases = metadata["load_windows_s"]
    active = np.zeros(times.size, dtype=bool)
    for start, end in phases:
        active |= (times > start + 0.3) & (times < end - 0.3)
    idle = ((times > 2) & (times < 8)) | ((times > 45) & (times < 58))
    lags = np.arange(-20, 20.0001, 0.05)
    lag_scores, statistics = {}, {}
    for label in "ABCD":
        rms = reduced[f"{label}_rms_mV"]
        scores = []
        for lag in lags:
            # Positive lag means a feature occurs in scope data later than load.
            shifted_times = times - lag
            valid = (shifted_times >= 0) & (shifted_times <= metadata["duration_s"])
            predicted = np.zeros(valid.sum())
            for start, end in phases:
                predicted += ((shifted_times[valid] >= start) & (shifted_times[valid] < end))
            scores.append(float(np.corrcoef(rms[valid], predicted)[0, 1]))
        scores = np.array(scores)
        lag_scores[label] = scores
        best = int(np.nanargmax(np.abs(scores)))
        baseline, activity = float(np.median(rms[idle])), float(np.median(rms[active]))
        statistics[label] = {
            "idle_median_rms_mV": baseline,
            "zero_lag_active_median_rms_mV": activity,
            "zero_lag_active_idle_ratio": activity / baseline,
            "rms_percentiles_mV": dict(zip(("p1", "p50", "p99"), np.quantile(rms, [0.01, 0.5, 0.99]).tolist())),
            "best_absolute_correlation": float(scores[best]),
            "best_lag_s": float(lags[best]),
            "zero_lag_correlation": float(scores[np.argmin(np.abs(lags))]),
        }

    for with_envelope, filename in ((False, "rms.png"), (True, "envelope.png")):
        fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
        for index, label in enumerate("ABCD"):
            ax = axes[index]
            if with_envelope:
                ax.fill_between(times, reduced[f"{label}_min_mV"], reduced[f"{label}_max_mV"], alpha=0.3, label="native sample min/max per 10 ms")
            ax.plot(times, reduced[f"{label}_rms_mV"], lw=0.8, label="RMS of native samples per 10 ms")
            ax.set_ylabel(f"Channel {label}\n(mV)")
            if index == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[4].plot(driver_times, driver_watts, label="nvmlDeviceGetPowerUsage", lw=1)
        instant = np.array([np.nan if r["instant_power_w"] is None else r["instant_power_w"] for r in rows])
        axes[4].plot(driver_times, instant, label="NVML_FI_DEV_POWER_INSTANT", alpha=0.6, lw=0.8)
        axes[4].set_ylabel("NVIDIA (W)")
        axes[4].legend(loc="upper right", fontsize=8)
        axes[5].plot(driver_times, [r["gpu_utilization_percent"] for r in rows], lw=1)
        axes[5].set_ylabel("GPU busy (%)")
        axes[5].set_xlabel("Seconds from RunBlock return (scope sample-zero offset is uncalibrated)")
        for ax in axes:
            for start, end in phases:
                ax.axvspan(start, end, color="orange", alpha=0.16)
            ax.set_xlim(0, metadata["duration_s"])
            ax.grid(alpha=0.2)
        fig.suptitle(f"{metadata['hostname']}: continuous 60 s capture, 2 / 4 / 6 s CUDA bursts\nNative low-rate samples can alias high-frequency current; use for lag diagnosis only")
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for label in "ABCD":
        ax.plot(lags, lag_scores[label], label=f"Channel {label}")
    ax.set(xlabel="Scope feature lag after workload (s)", ylabel="Correlation with load schedule", ylim=(-1, 1))
    ax.axvline(0, color="black", alpha=0.3)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "lag-search.png", dpi=150)
    plt.close(fig)
    report = dict(metadata)
    report["channels"] = statistics
    report["nvidia"] = {
        "samples": len(rows),
        "poll_median_ms": float(np.median(np.diff(driver_times)) * 1000),
        "driver_min_W": float(driver_watts.min()),
        "driver_peak_W": float(driver_watts.max()),
        "peak_gpu_utilization_percent": max(r["gpu_utilization_percent"] for r in rows),
        "instant_success_count": sum(r["instant_status"] == 0 for r in rows),
        "instant_unique_values": len({r["instant_power_w"] for r in rows if r["instant_power_w"] is not None}),
        "instant_unique_timestamps": len({r["instant_timestamp_us"] for r in rows}),
    }
    report["lag_interpretation"] = "Best lag is descriptive, not a calibration: noise always has a largest correlation. No claimed offset without a visible matching pulse pattern."
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-us", type=float, default=5.0,
                        help="Requested native interval; actual nearest interval is recorded")
    args = parser.parse_args()
    if not np.isfinite(args.interval_us) or args.interval_us <= 0:
        parser.error("--interval-us must be positive and finite")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use sbatch --gres=gpu:1,scope:1 --exclusive --time=00:05:00")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "started", "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "events": {}, "requested_duration_s": 60, "requested_interval_s": args.interval_us * 1e-6,
        "resolution_bits": 16, "range_v": 10, "channels_enabled": list("ABCD"),
        "raw_file": "channels-adc.npy", "limitations": __doc__,
    }
    probe, sensitivity, source = probe_settings()
    metadata.update(probe=probe, sensitivity_mV_per_A=sensitivity, probe_settings_source=source)
    torch.set_num_threads(2)
    a = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    b, c = torch.randn_like(a), torch.empty_like(a)
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    time.sleep(0.5)
    library = ct.CDLL("/opt/picoscope/lib/libpsospa.so")
    for name, signature in SIGNATURES.items():
        function = getattr(library, "psospa" + name)
        function.argtypes, function.restype = signature, u32

    def call(name, *values):
        status = getattr(library, "psospa" + name)(*values)
        if status:
            raise RuntimeError(f"{name}: 0x{status:08x}")

    lock = open("/tmp/pico-workshop-capture.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle, ready = i16(), i16()
    stop, rows, errors = threading.Event(), [], []
    thread = threading.Thread(target=monitor, args=(stop, rows, errors), daemon=True)
    try:
        call("OpenUnit", ct.byref(handle), None, 4, None)
        for name, field in (("model", 3), ("serial", 4), ("driver", 0)):
            buffer, required = ct.create_string_buffer(256), i16()
            call("GetUnitInfo", handle, buffer, 256, ct.byref(required), field)
            metadata[name] = buffer.value.decode()
        for channel in range(4):
            call("SetChannelOn", handle, channel, 1, -10_000_000_000, 10_000_000_000, 0, 0.0, 0)
        for port in (128, 129):
            call("SetDigitalPortOff", handle, port)
        call("SetSimpleTrigger", handle, 0, 0, 0, 2, 0, 0)
        maximum, timebase, interval = u64(), u32(), f64()
        call("MemorySegments", handle, 1, ct.byref(maximum))
        call("NearestSampleIntervalStateless", handle, 15, args.interval_us * 1e-6, 0, 4, ct.byref(timebase), ct.byref(interval))
        n = round(60 / interval.value)
        actual_ns = f64()
        call("GetTimebase", handle, timebase, n, ct.byref(actual_ns), ct.byref(maximum), 0)
        assert n <= maximum.value, (n, maximum.value)
        assert np.isclose(actual_ns.value, interval.value * 1e9, rtol=1e-9)
        adc_min, adc_max = i16(), i16()
        call("GetAdcLimits", handle, 4, ct.byref(adc_min), ct.byref(adc_max))
        metadata.update(interval_s=interval.value, duration_s=n * interval.value,
                        samples_per_channel=n, sample_rate_hz=1 / interval.value,
                        adc_min=adc_min.value, adc_max=adc_max.value, timebase=timebase.value)
        raw = np.lib.format.open_memmap(args.output / "channels-adc.npy", mode="w+", dtype=np.int16, shape=(4, n))
        for channel in range(4):
            call("SetDataBuffer", handle, channel, raw[channel].ctypes.data, n, 1, 0, 0x80000000, 3 if channel == 0 else 2)
        duration = f64()
        thread.start()
        metadata["events"]["run_block_call"] = timestamp()
        call("RunBlock", handle, 0, n, timebase, ct.byref(duration), 0, None, None)
        metadata["events"]["run_block_return"] = timestamp()
        metadata["driver_reported_time_indisposed_ms"] = duration.value
        origin_ns = metadata["events"]["run_block_return"]["monotonic_ns"]
        metadata["load_windows_s"] = []
        print(f"Capturing {n} points/channel, {interval.value*1e6:.4f} us interval for {n*interval.value:.3f} s", flush=True)

        def elapsed():
            return (time.monotonic_ns() - origin_ns) / 1e9

        def poll_ready():
            if not ready.value:
                call("IsReady", handle, ct.byref(ready))
                if ready.value:
                    metadata["events"]["capture_ready"] = timestamp()

        def idle_until(target):
            while elapsed() < target:
                poll_ready()
                time.sleep(max(0, min(0.01, target - elapsed())))

        for index, (target_start, length) in enumerate(((10, 2), (17, 4), (26, 6)), start=1):
            idle_until(target_start)
            metadata["events"][f"load_{index}_begin"] = timestamp()
            start = elapsed()
            operations = 0
            while elapsed() < target_start + length:
                torch.mm(a, b, out=c)
                torch.cuda.synchronize()
                operations += 1
                poll_ready()
            end = elapsed()
            metadata["events"][f"load_{index}_end"] = timestamp()
            metadata["load_windows_s"].append([start, end])
            metadata[f"load_{index}_matmuls"] = operations
            print(f"Load {index}: {start:.3f}–{end:.3f} s, {operations} matmuls", flush=True)
        idle_until(60)
        while not ready.value:
            poll_ready()
            if elapsed() > 90:
                raise TimeoutError("Scope not ready within 90 seconds of RunBlock return")
            time.sleep(0.01)
        metadata["events"]["get_values_call"] = timestamp()
        count, overflow = u64(n), i16()
        call("GetValues", handle, 0, ct.byref(count), 1, 0x80000000, 0, ct.byref(overflow))
        metadata["events"]["get_values_return"] = timestamp()
        raw.flush()
        metadata.update(returned_samples_per_channel=count.value, overflow_mask=overflow.value)
        assert count.value == n, f"Expected {n} samples, got {count.value}"
        assert not overflow.value, f"Scope overflow: {overflow.value}"
        metadata["status"] = "captured"
    finally:
        stop.set()
        if thread.ident:
            thread.join(timeout=10)
        if handle.value > 0:
            for method in ("Stop", "CloseUnit"):
                status = getattr(library, "psospa" + method)(handle)
                if status:
                    errors.append(f"{method}: 0x{status:08x}")
        lock.close()
        metadata["nvidia_errors"] = errors
        (args.output / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if rows:
            with (args.output / "nvidia.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if errors or not rows:
        raise RuntimeError(f"NVIDIA monitor failed: {errors}")
    analyse(raw, metadata, rows, args.output)


if __name__ == "__main__":
    main()
