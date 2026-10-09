"""Exercise bundle integrity, the upload boundary, and scientific round trips."""

from functools import partial
import gzip
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from recording_bundle import unpack_bundle
from serve import Handler, load_captures


def bundle(raw=None, corrupt=False, extra=None):
    raw = np.array([-12, 0, 22, 9], dtype=np.int16) if raw is None else raw
    array = io.BytesIO()
    np.save(array, raw)
    summary = {"title": "Test recording", "samples": len(raw), "interval_ns": 0.4,
               "duration_ms": len(raw) * 0.4 / 1e6, "sample_rate_hz": 2.5e9,
               "current_A_per_adc_count": 0.05, "phases": [],
               "perfetto": {"capture_zero_trace_ms": 23}, "timing": {"display_note": "Test"}}
    files = {"summary.json": json.dumps(summary).encode(), "channel-a-adc.npy": array.getvalue(),
             "trace.json.gz": gzip.compress(b'{"traceEvents": []}')}
    manifest = {"format": "aisb-waveform-v1", "id": "test-recording", "files": {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()}}
    if corrupt:
        files["channel-a-adc.npy"] = files["channel-a-adc.npy"][:-1] + b'X'
    files["bundle.json"] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        if extra:
            archive.writestr(extra, b"unexpected")
    return output.getvalue()


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def server(self, enabled=True):
        captures = {}
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(
            Handler, captures=captures, imports=self.root / "imports" if enabled else None))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", captures

    def upload(self, base, payload, header=True):
        headers = {"Content-Type": "application/zip"}
        if header:
            headers["X-AISB-Import"] = "1"
        return urlopen(Request(base + "/api/import", data=payload, headers=headers))

    def test_upload_preserves_samples_metadata_trace_and_survives_restart(self):
        base, captures = self.server()
        with self.upload(base, bundle()) as response:
            identity = json.load(response)["id"]
        with urlopen(base + "/api/waveform?capture=" + identity) as response:
            waveform = json.load(response)
        self.assertEqual(waveform["kind"], "raw")
        np.testing.assert_allclose(waveform["y"], [-0.6, 0, 1.1, 0.45])
        self.assertEqual(captures[identity]["summary"]["perfetto"]["capture_zero_trace_ms"], 23)
        self.assertEqual(gzip.decompress(captures[identity]["trace"].read_bytes()), b'{"traceEvents": []}')
        restored = load_captures(self.root / "imports", self.root / "imports")
        self.assertTrue(restored[identity]["summary"]["has_raw"])
        # Importing a second time makes a separate capture, without replacing one.
        with self.upload(base, bundle()) as response:
            second = json.load(response)["id"]
        self.assertNotEqual(identity, second)
        self.assertEqual(len(captures), 2)

    def test_read_only_server_and_cross_origin_form_boundary(self):
        base, _ = self.server(enabled=False)
        with urlopen(base + "/api/capabilities") as response:
            self.assertFalse(json.load(response)["import_recording"])
        with self.assertRaises(HTTPError) as error:
            self.upload(base, bundle())
        self.assertEqual(error.exception.code, 404)
        base, _ = self.server()
        with self.assertRaises(HTTPError) as error:
            self.upload(base, bundle(), header=False)
        self.assertEqual(error.exception.code, 403)

    def test_corrupt_and_wrong_dtype_uploads_leave_no_partial_capture(self):
        base, captures = self.server()
        for payload in (bundle(corrupt=True), bundle(raw=np.array([1, 2], dtype=np.float32))):
            with self.assertRaises(HTTPError) as error:
                self.upload(base, payload)
            self.assertEqual(error.exception.code, 400)
            self.assertFalse(captures)
            self.assertEqual(list((self.root / "imports").iterdir()), [])

    def test_archive_cannot_write_outside_staging_or_add_unexpected_files(self):
        for extra in ("../escaped", "/tmp/escaped", "unrequested.txt", "summary.json"):
            archive = self.root / "recording.zip"
            archive.write_bytes(bundle(extra=extra))
            with self.assertRaises(ValueError):
                unpack_bundle(archive, self.root / "capture")
            self.assertFalse((self.root / "capture").exists())
            self.assertFalse((self.root / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
