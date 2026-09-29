"""Record PicoScope 5464E channel A to raw int16 .npy plus timing/scale metadata.

Requires numpy and /opt/picoscope/lib/libpsospa.so (the installed PicoSDK driver).
RCP120XS probe: nominal 50 mV/A, AC only; the measured conductor is unconfirmed.
This is current, not total GPU DC power. No raw samples are averaged or discarded.
Edit the settings below and run `python capture_scope.py` for an idle recording.
Use `with scope:` around the workload, synchronizing CUDA before leaving the block.
Entering starts acquisition; normal exit downloads it; every exit closes the device.
420 ms at 0.4 ns takes 2.1 GB. Longer work needs a larger native sample interval.
"""

import ctypes as ct
import fcntl
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

OUTPUT = Path("/tmp/idle")
DURATION_S = 0.42
INTERVAL_NS = 0.4
RANGE_V = 10
SENSITIVITY_MV_PER_A = 50

# Native argument types must match PicoSDK, especially 64-bit sample counts/pointers.
i16, i32, u32, u64, f64, ptr = ct.c_int16, ct.c_int32, ct.c_uint32, ct.c_uint64, ct.c_double, ct.POINTER
SIGNATURES = {
    "OpenUnit": [ptr(i16), ct.c_char_p, i32, ct.c_void_p],
    "GetUnitInfo": [i16, ct.c_void_p, i16, ptr(i16), i32],
    "SetChannelOn": [i16, i32, i32, ct.c_int64, ct.c_int64, i32, f64, i32],
    "SetChannelOff": [i16, i32], "SetDigitalPortOff": [i16, i32],
    "SetSimpleTrigger": [i16, i16, i32, i16, i32, u64, u32],
    "MemorySegments": [i16, u64, ptr(u64)],
    "NearestSampleIntervalStateless": [i16, u32, f64, ct.c_uint8, i32, ptr(u32), ptr(f64)],
    "GetTimebase": [i16, u32, u64, ptr(f64), ptr(u64), u64],
    "GetAdcLimits": [i16, i32, ptr(i16), ptr(i16)],
    "RunBlock": [i16, u64, u64, u32, ptr(f64), u64, ct.c_void_p, ct.c_void_p],
    "IsReady": [i16, ptr(i16)],
    "SetDataBuffer": [i16, i32, ct.c_void_p, u64, i32, u64, u32, u32],
    "GetValues": [i16, u64, ptr(u64), u64, u32, u64, ptr(i16)],
    "Stop": [i16], "CloseUnit": [i16],
}


class Scope:
    def __init__(self, out, duration_s=DURATION_S, interval_ns=INTERVAL_NS):
        if not 0 < duration_s <= 4 or not math.isfinite(interval_ns) or interval_ns <= 0:
            raise ValueError("Need duration in (0, 4] seconds and a positive finite interval")
        self.out = Path(out)
        self.interval_ns = interval_ns
        self.samples = round(duration_s / (interval_ns * 1e-9))
        self.handle = i16()
        self.meta = {"status": "failed", "range_v": RANGE_V,
                     "sensitivity_mV_per_A": SENSITIVITY_MV_PER_A,
                     "probe": "RCP120XS; AC only; conductor/rail coverage unconfirmed",
                     "resolution_bits": 16, "raw_file": "scope.npy"}

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "scope.npy").exists() or (self.out / "scope.json").exists():
            raise FileExistsError("Scope output already exists")
        if self.samples < 1 or shutil.disk_usage(self.out).free < self.samples * 2 + 2**30:
            raise ValueError("Invalid sample count or insufficient disk space")
        self.lock = open("/tmp/pico-workshop-capture.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lib = ct.CDLL("/opt/picoscope/lib/libpsospa.so")
            for name, signature in SIGNATURES.items():
                function = getattr(self.lib, "psospa" + name)
                function.argtypes, function.restype = signature, u32
            self.call("OpenUnit", ct.byref(self.handle), None, 4, None)  # Resolution enum 4 = 16 bits.
            for name, field in (("model", 3), ("serial", 4), ("driver", 0)):
                buffer, required = ct.create_string_buffer(256), i16()
                self.call("GetUnitInfo", self.handle, buffer, 256, ct.byref(required), field)
                self.meta[name] = buffer.value.decode()
            # Channel A, DC 1 Mohm, limits in nanovolts, no offset/filter.
            limit = round(RANGE_V * 1e9)
            self.call("SetChannelOn", self.handle, 0, 1, -limit, limit, 0, 0.0, 0)
            for channel in (1, 2, 3):
                self.call("SetChannelOff", self.handle, channel)
            for port in (128, 129):
                self.call("SetDigitalPortOff", self.handle, port)
            self.call("SetSimpleTrigger", self.handle, 0, 0, 0, 2, 0, 0)  # Free-running block.
            maximum = u64()
            self.call("MemorySegments", self.handle, 1, ct.byref(maximum))
            self.timebase, interval = u32(), f64()
            self.call("NearestSampleIntervalStateless", self.handle, 1, self.interval_ns * 1e-9,
                      0, 4, ct.byref(self.timebase), ct.byref(interval))
            if not math.isclose(interval.value, self.interval_ns * 1e-9, rel_tol=1e-8):
                raise ValueError(f"Unsupported interval; nearest is {interval.value * 1e9:g} ns")
            actual_ns, adc_min, adc_max = f64(), i16(), i16()
            self.call("GetTimebase", self.handle, self.timebase, self.samples, ct.byref(actual_ns), ct.byref(maximum), 0)
            if self.samples > maximum.value:
                raise ValueError("Capture exceeds scope memory; increase the sample interval")
            if not math.isclose(actual_ns.value, self.interval_ns, rel_tol=1e-8):
                raise ValueError("Hardware timebase disagrees with requested interval")
            self.call("GetAdcLimits", self.handle, 4, ct.byref(adc_min), ct.byref(adc_max))
            self.meta.update(samples=self.samples, interval_s=interval.value, adc_max=adc_max.value)
            duration = f64()
            self.meta["start_call_unix_ns"] = time.time_ns()
            self.call("RunBlock", self.handle, 0, self.samples, self.timebase, ct.byref(duration), 0, None, None)
            # Host return time requires a separate offset estimate to locate sample zero.
            self.meta["start_return_unix_ns"] = time.time_ns()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None:
                return False  # Preserve the workload exception; stop the scope below.
            ready, deadline = i16(), time.monotonic() + 10
            while not ready.value:
                self.call("IsReady", self.handle, ct.byref(ready))
                if time.monotonic() > deadline:
                    raise TimeoutError("Scope acquisition did not complete")
                if not ready.value:
                    time.sleep(0.001)
            # Keep the native buffer alive until CloseUnit; store every int16 sample.
            self.raw = np.lib.format.open_memmap(self.out / "scope.npy", mode="w+", dtype=np.int16, shape=(self.samples,))
            # int16 (1), RAW mode (0x80000000), clear/add buffer (3), ratio 1.
            self.call("SetDataBuffer", self.handle, 0, self.raw.ctypes.data, self.samples, 1, 0, 0x80000000, 3)
            count, overflow = u64(self.samples), i16()
            self.call("GetValues", self.handle, 0, ct.byref(count), 1, 0x80000000, 0, ct.byref(overflow))
            self.raw.flush()
            self.meta["overflow_mask"] = overflow.value
            if count.value != self.samples or overflow.value:
                raise RuntimeError("Incomplete or clipped capture; raw data retained")
            self.meta["status"] = "completed"
        finally:
            self.close()

    def call(self, name, *args):
        status = getattr(self.lib, "psospa" + name)(*args)
        if status:
            raise RuntimeError(f"psospa{name}: 0x{status:08x}")

    def close(self):
        if self.handle.value > 0:
            for name in ("Stop", "CloseUnit"):
                getattr(self.lib, "psospa" + name)(self.handle)
            self.handle.value = 0
        try:
            (self.out / "scope.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        finally:
            self.lock.close()


if __name__ == "__main__":
    with Scope(OUTPUT):
        pass  # An idle recording; exit waits for the finite block and downloads it.
    print(f"Saved {OUTPUT / 'scope.npy'}")
