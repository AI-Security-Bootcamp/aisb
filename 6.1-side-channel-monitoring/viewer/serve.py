"""Serve archived scope waveforms and their matching Perfetto traces.

This portable backend implements the API used by the archived port-6008 UI.
It needs only NumPy; it never connects to CUDA or the oscilloscope.
"""

import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parent
MAX_POINTS = 20_000
STRIDES = (256, 4096, 65536, 1048576)
WINDOWS_MS = (0.01, 0.1, 0.5, 2.0)
STATIC_FILES = {
    "index.html",
    "style.css",
    "app.js",
    "perfetto-launch.js",
    "plotly.min.js",
}


class Waveform:
    """Memory-map ADC counts; reduce wide views without discarding spikes."""

    def __init__(self, folder, summary):
        self.raw = np.load(
            folder / "channel-a-adc.npy", mmap_mode="r", allow_pickle=False
        )
        if self.raw.ndim != 1 or self.raw.dtype != np.dtype("<i2"):
            raise ValueError(f"{folder}: expected a one-dimensional int16 ADC array")
        if len(self.raw) != summary["samples"] or not len(self.raw):
            raise ValueError(f"{folder}: sample count does not match summary.json")
        self.dt_ms = summary["interval_ns"] / 1e6
        self.gain = summary["current_A_per_adc_count"]
        if not self.dt_ms > 0 or not self.gain > 0:
            raise ValueError("Sample interval and ADC-to-current gain must be positive")
        self.duration_ms = len(self.raw) * self.dt_ms
        self.envelopes = {}
        low = high = self.raw
        previous = 1
        for stride in STRIDES:
            starts = np.arange(0, len(low), stride // previous)
            low = np.minimum.reduceat(low, starts)
            high = np.maximum.reduceat(high, starts)
            self.envelopes[stride] = (low, high)
            previous = stride
        self.rms_time, self.rms = self._rms_curves()

    def _rms_curves(self):
        # Mean square in ~1 µs bins bounds memory use even for billion-sample
        # captures. Every ADC sample contributes; smoothing operates on bins.
        samples_per_bin = max(1, round(0.001 / self.dt_ms))
        bin_ms = samples_per_bin * self.dt_ms
        n_bins = math.ceil(len(self.raw) / samples_per_bin)
        mean_square = np.empty(n_bins, dtype=np.float64)
        chunk_size = samples_per_bin * 4096
        for start in range(0, len(self.raw), chunk_size):
            values = self.raw[start : start + chunk_size].astype(np.float64)
            values *= self.gain
            values *= values
            bins = np.arange(0, len(values), samples_per_bin)
            counts = np.minimum(samples_per_bin, len(values) - bins)
            first = start // samples_per_bin
            mean_square[first : first + len(bins)] = (
                np.add.reduceat(values, bins) / counts
            )
        curves = {}
        for window in WINDOWS_MS:
            width = max(1, round(window / bin_ms))
            left = width // 2
            padded = np.pad(mean_square, (left, width - left - 1), mode="edge")
            cumulative = np.concatenate(([0.0], np.cumsum(padded)))
            curves[window] = np.sqrt(
                np.maximum(0, (cumulative[width:] - cumulative[:-width]) / width)
            )
        starts = np.arange(n_bins) * samples_per_bin
        return starts * self.dt_ms, curves

    def query(self, lo_ms, hi_ms, mode):
        if not math.isfinite(lo_ms) or not math.isfinite(hi_ms):
            raise ValueError("Time limits must be finite")
        lo_ms, hi_ms = max(0, lo_ms), min(self.duration_ms, hi_ms)
        if hi_ms <= lo_ms:
            raise ValueError("Select a nonempty interval inside the recording")
        # Samples are at index * dt. Include those in [lo_ms, hi_ms).
        first = min(len(self.raw), math.ceil(lo_ms / self.dt_ms))
        stop = min(len(self.raw), math.ceil(hi_ms / self.dt_ms))
        count = stop - first
        if count <= 0:
            raise ValueError("The interval contains no ADC samples")
        result = {"raw_samples_in_range": count}
        if mode != "raw":
            window = float(mode)
            if window not in self.rms:
                raise ValueError("RMS window must be 0.01, 0.1, 0.5, or 2 ms")
            begin, end = np.searchsorted(self.rms_time, [lo_ms, hi_ms])
            step = max(1, math.ceil((end - begin) / MAX_POINTS))
            return {
                **result,
                "kind": "rms",
                "x_ms": self.rms_time[begin:end:step].tolist(),
                "y": self.rms[window][begin:end:step].tolist(),
                "description": f"RMS {window:g} ms; display every {step} mean-square bins",
            }
        if count <= MAX_POINTS:
            return {
                **result,
                "kind": "raw",
                "x_ms": (np.arange(first, stop) * self.dt_ms).tolist(),
                "y": (self.raw[first:stop] * self.gain).tolist(),
                "description": f"Every ADC sample · {self.dt_ms * 1e6:g} ns spacing",
            }
        stride = next(s for s in STRIDES if math.ceil(count / s) + 1 <= MAX_POINTS)
        begin, end = first // stride, math.ceil(stop / stride)
        cached_low, cached_high = self.envelopes[stride]
        low, high = cached_low[begin:end].copy(), cached_high[begin:end].copy()
        starts = np.maximum(np.arange(begin, end) * stride, first)
        stops = np.minimum((np.arange(begin, end) + 1) * stride, stop)
        # Recalculate the two boundary bins: never include extrema outside the
        # requested range simply because they share a display bucket.
        for index in {0, len(starts) - 1}:
            values = self.raw[starts[index] : stops[index]]
            low[index], high[index] = values.min(), values.max()
        return {
            **result,
            "kind": "envelope",
            "x_ms": ((starts + stops - 1) * 0.5 * self.dt_ms).tolist(),
            "low": (low * self.gain).tolist(),
            "high": (high * self.gain).tolist(),
            "description": f"Exact min/max envelope · up to {stride:,} ADC samples per bucket",
        }


def load_captures(recordings):
    """Pair committed traces with optional, untracked raw recordings by ID."""
    captures = {}
    for path in sorted((ROOT / "traces").glob("*/summary.json")):
        name = path.parent.name
        raw_folder = recordings / name
        has_raw = (raw_folder / "channel-a-adc.npy").is_file()
        summary_path = raw_folder / "summary.json" if has_raw else path
        summary = json.loads(summary_path.read_text())
        summary["has_raw"] = has_raw
        # Both archives and cropped bundles use capture-relative timestamps.
        # The trace itself keeps its original clock and is never rewritten.
        summary["perfetto"]["file"] = "trace.json.gz"
        waveform = None
        if has_raw:
            print(f"Preparing {name}: {summary['samples']:,} ADC samples…", flush=True)
            waveform = Waveform(raw_folder, summary)
        captures[name] = {
            "summary": summary,
            "waveform": waveform,
            "raw": raw_folder / "channel-a-adc.npy",
            "trace": path.parent / "trace.json.gz",
        }
    if not captures:
        raise ValueError("No archived traces found in viewer/traces/")
    return captures


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, captures, **kwargs):
        self.captures = captures
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self._route()

    def do_HEAD(self):
        self._route()

    def _route(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        default = (
            "reference" if "reference" in self.captures else next(iter(self.captures))
        )
        name = query.get("capture", [default])[0]
        try:
            if url.path == "/api/catalog":
                self._json(
                    [
                        {
                            "id": key,
                            "title": item["summary"]["title"],
                            "default": key == default,
                        }
                        for key, item in self.captures.items()
                    ]
                )
                return
            filename = url.path.lstrip("/") or "index.html"
            if filename in STATIC_FILES:
                self._file(ROOT / "static" / filename)
                return
            if name not in self.captures:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown recording")
                return
            capture = self.captures[name]
            if url.path == "/api/summary":
                self._json(capture["summary"])
            elif url.path == "/api/waveform":
                if capture["waveform"] is None:
                    self.send_error(
                        HTTPStatus.NOT_FOUND,
                        "Raw samples are not installed; open this recording in Perfetto",
                    )
                    return
                lo = float(query.get("lo_ms", [0])[0])
                hi = float(query.get("hi_ms", [capture["summary"]["duration_ms"]])[0])
                mode = query.get("mode", ["raw"])[0]
                self._json(capture["waveform"].query(lo, hi, mode))
            elif url.path == "/data/trace.json.gz":
                self._file(capture["trace"], download=True)
            elif url.path == "/data/channel-a-adc.npy":
                self._file(capture["raw"], download=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, OverflowError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass  # Zooming cancels the previous fetch; this is normal.

    def _json(self, data):
        payload = json.dumps(data, allow_nan=False).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _file(self, path, download=False):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        first, last = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.send_error(HTTPStatus.BAD_REQUEST, "Use a single byte range")
                return
            start, end = match.groups()
            if start:
                first = int(start)
                last = min(last, int(end)) if end else last
            else:
                first = max(0, size - int(end))
            if first > last or first >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
        media_type = (
            "application/gzip"
            if path.suffix == ".gz"
            else mimetypes.guess_type(path)[0]
        )
        self.send_header("Content-Type", media_type or "application/octet-stream")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Accept-Ranges", "bytes")
        if requested:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        if download:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            source.seek(first)
            remaining = last - first + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    args = parser.parse_args()
    captures = load_captures(args.recordings)
    server = ThreadingHTTPServer(
        (args.host, args.port), partial(Handler, captures=captures)
    )
    print(f"Viewer: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
