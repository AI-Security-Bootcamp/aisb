"""Check that adding a smoothing window preserves the capture clock and events."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from add_rms_track import TRACK, add_track


class AddRMSTrackTests(unittest.TestCase):
    def test_preserves_existing_events_and_original_time_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("in.gz", "out.gz"))
            original = [
                dict(ph="X", pid=7, tid=3, ts=123500.25, dur=18.0, name="CUDA kernel"),
                dict(ph="C", pid=900000, tid=0, ts=123456.789,
                     name="AC current RMS 0.1 ms (A)", args={"value": 9.0}),
            ]
            payload = {"traceEvents": original, "metadata": {"origin_unix_ns": 1791614980540689412}}
            with gzip.open(source, "wt") as stream:
                json.dump(payload, stream)
            summary = {"perfetto": {"capture_zero_trace_ms": 123.456789}}
            add_track(source, target, summary, np.array([2., 5., 7.]), 0., .001)
            with gzip.open(target, "rt") as stream:
                result = json.load(stream)
            self.assertEqual(result["traceEvents"][:2], original)
            self.assertEqual(result["metadata"]["origin_unix_ns"], 1791614980540689412)
            added = [event for event in result["traceEvents"] if event["name"] == TRACK]
            self.assertEqual([event["ts"] for event in added], [123456.789, 123457.789, 123458.789])
            self.assertEqual([event["args"]["value"] for event in added], [2., 5., 7.])

    def test_rejects_an_inconsistent_capture_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("in.gz", "out.gz"))
            with gzip.open(source, "wt") as stream:
                json.dump({"traceEvents": [dict(ph="C", pid=1, tid=0, ts=500.,
                          name="AC current RMS 0.1 ms (A)", args={"value": 1.})]}, stream)
            with self.assertRaises(AssertionError):
                add_track(source, target, {"perfetto": {"capture_zero_trace_ms": 2.}},
                          np.array([1.]), 0., .001)


if __name__ == "__main__":
    unittest.main()
