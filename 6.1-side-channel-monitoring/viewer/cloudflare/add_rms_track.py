"""Add the archived 0.01 ms RMS array to each catalogued Perfetto recording.

Reads existing US archive objects without credentials. Writes new trace objects
and a new catalog; never changes ADC samples, timestamps, or existing tracks.
Run before build.py, then upload and verify the resulting upload inventory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

TRACK = "AC current RMS 0.01 ms (A)"
REVISION = "rms-001-20261010"


def download(key, destination):
    subprocess.run([
        "curl", "--fail", "--silent", "--show-error", "--retry", "3",
        "https://traces.aisb.dev/" + key, "-o", str(destination),
    ], check=True)


def add_track(source, destination, summary, values, origin_ms, interval_ms):
    """Preserve existing events; stream new counters without a second huge list."""
    with gzip.open(source, "rt") as stream:
        trace = json.load(stream)
    events = trace.pop("traceEvents")
    template = next(event for event in events if event.get("ph") == "C"
                    and "RMS 0.1 ms" in event.get("name", ""))
    start_us = (summary["perfetto"]["capture_zero_trace_ms"] + origin_ms) * 1000
    step_us = interval_ms * 1000
    # Both windows must describe the same acquisition clock and sample grid.
    assert abs(template["ts"] - start_us) < 0.01
    metadata = trace.setdefault("metadata", {})
    metadata["perfetto_rms_windows_ms"] = [0.01, 0.1]
    metadata["rms_001_provenance"] = {
        "source": "Original precomputed 0.01 ms RMS array, every recorded output bin",
        "output_interval_us": step_us,
        "count": len(values),
        "timing": "Original acquisition origin; no realignment",
    }
    with gzip.open(destination, "wt", compresslevel=4) as stream:
        stream.write('{"traceEvents":[')
        first = True
        for event in events:
            if event.get("ph") == "C" and event.get("name") == TRACK:
                continue
            if not first:
                stream.write(",")
            json.dump(event, stream, separators=(",", ":"), allow_nan=False)
            first = False
        for index, value in enumerate(values):
            event = dict(ph="C", pid=template["pid"], tid=template.get("tid", 0),
                         ts=round(start_us + index * step_us, 3), name=TRACK,
                         args={"value": float(value)})
            stream.write(",")
            json.dump(event, stream, separators=(",", ":"), allow_nan=False)
        stream.write("]")
        for key, value in trace.items():
            stream.write("," + json.dumps(key) + ":")
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("}")


def prepare(catalog_path, output, prefix):
    captures = json.loads(catalog_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    uploads, provenance = [], []
    for identity, capture in captures.items():
        folder = output / identity
        folder.mkdir()
        source, rms, target = (folder / name for name in
                               ("original-trace.json.gz", "rms-0.01.f64", "trace.json.gz"))
        download(capture["trace_key"], source)
        download(capture["rms"]["0.01"]["1"], rms)
        values = np.fromfile(rms, dtype="<f8")
        assert len(values) == capture["rms_count"] and np.isfinite(values).all()
        add_track(source, target, capture["summary"], values,
                  capture["rms_origin_ms"], capture["rms_interval_ms"])
        provenance.append(dict(id=identity, original_key=capture["trace_key"],
                               original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               rms_key=capture["rms"]["0.01"]["1"],
                               rms_sha256=hashlib.sha256(rms.read_bytes()).hexdigest()))
        capture["trace_key"] = prefix.rstrip("/") + "/" + identity + "/trace.json.gz"
        capture["summary"]["perfetto"].update(rms_windows_ms=[0.01, 0.1], trace_revision=REVISION)
        uploads.append(dict(file=str(target.resolve()), key=capture["trace_key"],
                            bytes=target.stat().st_size, content_type="application/gzip",
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(json.dumps({"prepared": identity, "rms_001_points": len(values),
                          "bytes": target.stat().st_size}), flush=True)
    for name, value in (("catalog.json", captures), ("uploads.json", uploads),
                        ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.output, args.prefix)
