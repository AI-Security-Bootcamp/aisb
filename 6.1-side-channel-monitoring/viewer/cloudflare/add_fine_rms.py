"""Derive 1 us and 100 ns RMS from archived ADC samples, without acquisition.

Each value uses every raw sample in a nonoverlapping window, with its timestamp
at the mean sample time. Peak indexes let the viewer preserve brief excursions
at wide zoom without downloading millions of values. Nothing is interpolated
from the older, coarser RMS curves.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WINDOWS_MS = ("0.001", "0.0001")
PEAK_STRIDES = (16, 64, 256, 1024, 4096, 16384)
PEAK_DTYPE = np.dtype([
    ("low_index", "<u4"), ("low", "<f8"),
    ("high_index", "<u4"), ("high", "<f8"),
])


def rms_values(raw, width, gain):
    """Compute exact integer square sums before converting to current RMS.

    Only complete windows are emitted. Work in bounded chunks so a multi-GB
    memory-mapped recording never needs to fit in RAM. int64 avoids ADC square
    overflow and preserves the sum exactly for the supported short windows.
    """
    count = len(raw) // width
    values = np.empty(count, dtype="<f8")
    chunk_windows = max(1, 2_000_000 // width)
    for first in range(0, count, chunk_windows):
        stop = min(count, first + chunk_windows)
        samples = np.asarray(raw[first * width:stop * width], dtype=np.int64)
        np.square(samples, out=samples)
        sums = samples.reshape(-1, width).sum(axis=1, dtype=np.int64)
        values[first:stop] = np.sqrt(sums / width) * abs(gain)
    return values


def peak_records(values, stride):
    """Store each display bucket's extrema AND their original RMS indexes."""
    count = len(values)
    records = np.empty((count + stride - 1) // stride, dtype=PEAK_DTYPE)
    complete = count // stride
    if complete:
        groups = values[:complete * stride].reshape(-1, stride)
        starts = np.arange(complete) * stride
        records["low_index"][:complete] = starts + groups.argmin(axis=1)
        records["high_index"][:complete] = starts + groups.argmax(axis=1)
    if count % stride:
        tail = values[complete * stride:]
        records["low_index"][-1] = complete * stride + tail.argmin()
        records["high_index"][-1] = complete * stride + tail.argmax()
    records["low"] = values[records["low_index"]]
    records["high"] = values[records["high_index"]]
    return records


def prepare(catalog_path, raw_dir, output, prefix, capture_ids):
    catalog = json.loads(catalog_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    uploads = []

    def save(array, destination, key):
        array.tofile(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        uploads.append(dict(file=str(destination.resolve()), key=key,
                            bytes=destination.stat().st_size, sha256=digest,
                            content_type="application/octet-stream"))
        return key

    for identity in capture_ids:
        capture = catalog[identity]
        summary = capture["summary"]
        raw = np.load(raw_dir / f"{identity}.npy", mmap_mode="r", allow_pickle=False)
        if raw.dtype != np.dtype("<i2") or raw.ndim != 1 or len(raw) != summary["samples"]:
            raise ValueError(f"ADC layout does not match {identity}'s metadata")
        destination = output / identity
        destination.mkdir(exist_ok=True)
        base = f"{prefix.rstrip('/')}/{identity}"
        fine = {}
        for window in WINDOWS_MS:
            requested_width = float(window) / (summary["interval_ns"] * 1e-6)
            width = round(requested_width)
            if width < 1 or not np.isclose(width, requested_width, rtol=0, atol=1e-6):
                raise ValueError("RMS window must contain an integer number of raw samples")
            values = rms_values(raw, width, summary["current_A_per_adc_count"])
            name = f"rms-{window}.f64"
            item = dict(window_ms=float(window), interval_ms=float(window),
                        origin_ms=(width - 1) * summary["interval_ns"] * 0.5e-6,
                        samples_per_window=width, count=len(values),
                        omitted_tail_samples=len(raw) % width,
                        method="Nonoverlapping RMS over every raw sample; timestamp at mean sample time",
                        values_key=save(values, destination / name, f"{base}/{name}"), peaks={})
            for stride in PEAK_STRIDES:
                name = f"rms-{window}-peaks-{stride}.bin"
                item["peaks"][str(stride)] = save(
                    peak_records(values, stride), destination / name, f"{base}/{name}")
            fine[window] = item
        capture["fine_rms"] = fine
        summary["rms"]["fine_windows_ms"] = [float(w) for w in WINDOWS_MS]
        summary["rms"]["fine_method"] = fine[WINDOWS_MS[0]]["method"]
        print(json.dumps({"prepared": identity,
                          "windows": {w: f["count"] for w, f in fine.items()}}), flush=True)
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "uploads.json").write_text(json.dumps(uploads, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--captures", nargs="+", required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.raw_dir, args.output, args.prefix, args.captures)
