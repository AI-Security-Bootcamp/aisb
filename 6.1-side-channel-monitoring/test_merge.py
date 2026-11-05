"""CPU-only regression check for the participant Perfetto exporter."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import merge


class MergeTests(unittest.TestCase):
    def test_short_pulse_has_two_rms_windows_on_the_same_clock(self):
        # Two ADC samples per microsecond; a 10-us, 1-A pulse at 10 ms.
        raw = np.zeros(40_000, dtype=np.int16)
        raw[19_990:20_010] = 100
        base_ns = 1_700_000_000_000_000_000
        metadata = {
            "status": "completed", "overflow_mask": 0, "raw_file": "scope.npy",
            "samples": len(raw), "interval_s": 0.5e-6,
            "start_return_unix_ns": base_ns + 200_000,
            "range_v": 1, "adc_max": 100, "sensitivity_mV_per_A": 1000,
        }
        kernel = dict(ph="X", cat="kernel", name="fixture", pid=1, tid=1,
                      ts=6000, dur=1000)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            np.save(run / "scope.npy", raw)
            (run / "scope.json").write_text(json.dumps(metadata))
            (run / "torch.json").write_text(json.dumps({
                "baseTimeNanoseconds": base_ns, "traceEvents": [kernel],
            }))
            with patch.object(merge, "RUN", run), patch.object(merge, "CALIBRATION", None):
                merge.main()
            with gzip.open(run / "perfetto.json.gz", "rt") as stream:
                result = json.load(stream)

        self.assertEqual(result["metadata"]["rms_windows_ms"], [0.01, 0.1])
        self.assertEqual(result["traceEvents"][0], kernel)
        for window, expected in ((0.01, 1.0), (0.1, np.sqrt(0.1))):
            points = [event for event in result["traceEvents"]
                      if event.get("name") == f"AC current RMS {window:g} ms (A)"]
            self.assertEqual(len(points), 20_000)
            self.assertEqual(points[0]["ts"], 200)
            self.assertEqual(points[10_000]["ts"], 10_200)
            self.assertAlmostEqual(points[10_000]["args"]["value"], expected)


if __name__ == "__main__":
    unittest.main()
