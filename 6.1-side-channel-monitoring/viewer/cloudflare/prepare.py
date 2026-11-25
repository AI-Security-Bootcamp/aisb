"""Export existing capture bundles for the read-only Cloudflare/R2 viewer.

Keep raw NPY files and original RMS values unchanged. Envelope files store pairs
of int16 extrema; RMS levels select existing values, without recomputing them.
Run on the archive host, not as part of acquisition.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RMS_STRIDES = (1, 4, 16, 64, 256, 1024)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(catalog, output, prefix):
    output.mkdir(parents=True, exist_ok=False)
    captures, uploads = {}, []

    def register(path, key, content_type="application/octet-stream"):
        uploads.append(dict(file=str(path.resolve()), key=key, bytes=path.stat().st_size,
                            sha256=sha256(path), content_type=content_type))
        return key

    for folder in sorted(catalog.iterdir()):
        if not (folder / "manifest.json").is_file():
            continue
        identity = folder.name
        destination = output / identity
        destination.mkdir()
        summary = json.loads((folder / "manifest.json").read_text())
        raw = np.load(folder / "channel-a-adc.npy", mmap_mode="r", allow_pickle=False)
        assert raw.dtype == np.dtype("<i2") and raw.ndim == 1
        assert len(raw) == summary["samples"]
        summary["has_raw"] = True
        summary["perfetto"]["file"] = "trace.json.gz"
        base = f"{prefix.rstrip('/')}/{identity}"
        item = dict(summary=summary, raw_offset=int(raw.offset),
                    raw_bytes=(folder / "channel-a-adc.npy").stat().st_size,
                    raw_key=register(folder / "channel-a-adc.npy", f"{base}/channel-a-adc.npy"),
                    trace_key=register(folder / "trace.json.gz", f"{base}/trace.json.gz", "application/gzip"),
                    envelopes={}, rms={})
        for stride in summary["envelope_strides"]:
            envelope = np.load(folder / f"envelope-{stride}.npy", mmap_mode="r", allow_pickle=False)
            pairs = np.column_stack((envelope["min"], envelope["max"]))
            assert np.isfinite(pairs).all() and np.equal(pairs, pairs.astype("<i2")).all()
            path = destination / f"envelope-{stride}.i16"
            pairs.astype("<i2").tofile(path)
            item["envelopes"][str(stride)] = register(path, f"{base}/{path.name}")
        with np.load(folder / "rms.npz", allow_pickle=False) as archive:
            times = archive["time_ms"]
            step = float(times[1] - times[0])
            assert len(times) > 1 and step > 0
            assert np.allclose(times, times[0] + np.arange(len(times)) * step, atol=1e-9, rtol=1e-10)
            item["rms_origin_ms"] = float(times[0])
            item["rms_interval_ms"] = step
            item["rms_count"] = len(times)
            for window in ("0.01", "0.1", "0.5", "2"):
                values = archive[f"rms_{window}ms_A"]
                assert values.shape == times.shape and np.isfinite(values).all()
                item["rms"][window] = {}
                for stride in RMS_STRIDES:
                    path = destination / f"rms-{window}-{stride}.f64"
                    values[::stride].astype("<f8").tofile(path)
                    item["rms"][window][str(stride)] = register(path, f"{base}/{path.name}")
        captures[identity] = item
        print(json.dumps({"prepared": identity, "samples": len(raw)}), flush=True)
    if not captures:
        raise ValueError("No capture bundles found")
    (output / "catalog.json").write_text(json.dumps(captures, indent=2) + "\n")
    (output / "uploads.json").write_text(json.dumps(uploads, indent=2) + "\n")
    return captures, uploads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="waveform-viewer/2026-10-10")
    args = parser.parse_args()
    prepare(args.catalog, args.output, args.prefix)
