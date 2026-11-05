"""Short Slurm-only channel/wiring diagnostic: record A–D during GPU load pulses."""
import argparse
import ctypes as ct
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_scope import SIGNATURES, i16, u32, u64, f64
from merge import centered_rms
from validate_hardware import monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run under Slurm with gpu:1,scope:1 and --exclusive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    a = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    time.sleep(.5)
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
    handle = i16()
    stop, rows, errors = threading.Event(), [], []
    thread = threading.Thread(target=monitor, args=(stop, rows, errors), daemon=True)
    try:
        call("OpenUnit", ct.byref(handle), None, 4, None)
        for channel in range(4):
            call("SetChannelOn", handle, channel, 1, -10_000_000_000, 10_000_000_000, 0, 0., 0)
        for port in (128, 129):
            call("SetDigitalPortOff", handle, port)
        call("SetSimpleTrigger", handle, 0, 0, 0, 2, 0, 0)
        maximum, timebase, interval = u64(), u32(), f64()
        call("MemorySegments", handle, 1, ct.byref(maximum))
        call("NearestSampleIntervalStateless", handle, 15, 1e-6, 0, 4, ct.byref(timebase), ct.byref(interval))
        n = round(2.5 / interval.value)
        actual_ns = f64()
        call("GetTimebase", handle, timebase, n, ct.byref(actual_ns), ct.byref(maximum), 0)
        assert n <= maximum.value
        adc_min, adc_max = i16(), i16()
        call("GetAdcLimits", handle, 4, ct.byref(adc_min), ct.byref(adc_max))
        raw = np.empty((4, n), dtype=np.int16)
        for channel in range(4):
            call("SetDataBuffer", handle, channel, raw[channel].ctypes.data, n, 1, 0, 0x80000000, 3 if channel == 0 else 2)
        duration = f64()
        thread.start()
        call("RunBlock", handle, 0, n, timebase, ct.byref(duration), 0, None, None)
        origin_unix_ns, origin = time.time_ns(), time.monotonic()
        phases = []
        time.sleep(.2)
        for _ in range(4):
            start = time.monotonic() - origin
            deadline = time.monotonic() + .2
            while time.monotonic() < deadline:
                torch.mm(a, b, out=c)
                torch.cuda.synchronize()
            end = time.monotonic() - origin
            phases.append((start, end))
            time.sleep(.2)
        ready, deadline = i16(), time.monotonic() + 10
        while not ready.value:
            call("IsReady", handle, ct.byref(ready))
            if time.monotonic() > deadline:
                raise TimeoutError("Block did not finish")
            time.sleep(.002)
        count, overflow = u64(n), i16()
        call("GetValues", handle, 0, ct.byref(count), 1, 0x80000000, 0, ct.byref(overflow))
        assert count.value == n and overflow.value == 0
    finally:
        stop.set()
        if thread.ident:
            thread.join(timeout=10)
        if handle.value > 0:
            call("Stop", handle)
            call("CloseUnit", handle)
        lock.close()
    np.save(args.output / "channels-adc.npy", raw)
    times = np.arange(n) * interval.value
    idle = (times > .04) & (times < .15)
    active = np.zeros(n, dtype=bool)
    for start, end in phases:
        active |= (times > start + .03) & (times < end - .03)
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    statistics = {}
    for channel, label in enumerate("ABCD"):
        volts = raw[channel].astype(float) * 10 / adc_max.value
        rms = centered_rms(volts * volts, round(.0001 / interval.value)) * 1000
        baseline, activity = float(np.median(rms[idle])), float(np.median(rms[active]))
        statistics[label] = dict(idle_rms_mV=baseline, active_rms_mV=activity,
                                 active_idle_ratio=activity / max(baseline, 1e-12),
                                 min_mV=float(volts.min()*1000), max_mV=float(volts.max()*1000))
        axes[channel].plot(times[::100], rms[::100], lw=.6)
        axes[channel].set_ylabel(f"{label} RMS (mV)")
    axes[4].plot([(r["unix_ns"]-origin_unix_ns)/1e9 for r in rows], [r["driver_power_w"] for r in rows], marker=".")
    axes[4].set_ylabel("NVIDIA (W)")
    axes[4].set_xlabel("Seconds since RunBlock returned (uncalibrated)")
    for ax in axes:
        for start, end in phases:
            ax.axvspan(start, end, alpha=.1)
        ax.set_xlim(0, 2.5)
        ax.grid(alpha=.2)
    fig.savefig(args.output / "channels.png", dpi=140)
    report = dict(interval_ns=interval.value*1e9, samples_per_channel=n, overflow_mask=overflow.value,
                  channels=statistics, load_windows_s=phases, nvidia_errors=errors,
                  driver_min_W=min(r["driver_power_w"] for r in rows),
                  driver_peak_W=max(r["driver_power_w"] for r in rows))
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
