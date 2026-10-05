"""Check scientific reductions and downloads without GPU or scope access."""

from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from serve import Handler, Waveform


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def waveform(self, raw, interval_ns=0.4, gain=0.05):
        np.save(self.folder / "channel-a-adc.npy", np.asarray(raw, dtype=np.int16))
        return Waveform(
            self.folder,
            {
                "samples": len(raw),
                "interval_ns": interval_ns,
                "current_A_per_adc_count": gain,
            },
        )

    def test_close_view_retains_every_signed_adc_sample(self):
        raw = np.array([-32768, -14, 0, 125, 32767], dtype=np.int16)
        signal = self.waveform(raw)
        result = signal.query(0, signal.duration_ms, "raw")
        self.assertEqual(result["kind"], "raw")
        self.assertEqual(result["raw_samples_in_range"], len(raw))
        np.testing.assert_allclose(result["y"], raw.astype(float) * 0.05)
        np.testing.assert_allclose(result["x_ms"], np.arange(len(raw)) * 0.4e-6)

    def test_envelope_preserves_spikes_and_excludes_outside_samples(self):
        raw = np.zeros(60_123, dtype=np.int16)
        raw[[0, 300, 31_999, 60_122]] = [32000, -23000, 19000, -32768]
        signal = self.waveform(raw, interval_ns=1_000_000, gain=1)
        result = signal.query(200, 60_000, "raw")
        self.assertEqual(result["kind"], "envelope")
        self.assertEqual(result["raw_samples_in_range"], 59_800)
        self.assertEqual(min(result["low"]), -23000)
        self.assertEqual(max(result["high"]), 19000)
        self.assertTrue(all(200 <= x < 60_000 for x in result["x_ms"]))

    def test_coarse_envelope_keeps_extrema_in_last_partial_bucket(self):
        raw = np.zeros(5_400_123, dtype=np.int16)
        raw[-1] = 32000
        raw[521] = -32768
        signal = self.waveform(raw)
        result = signal.query(0, signal.duration_ms, "raw")
        self.assertLessEqual(len(result["x_ms"]), 20_000)
        self.assertEqual(max(result["high"]), 32000 * 0.05)
        self.assertEqual(min(result["low"]), -32768 * 0.05)

    def test_rms_squares_before_averaging_and_handles_edges(self):
        # Opposite signs cancel under a mean, but must retain 0.1 A RMS.
        signal = self.waveform(np.tile([-2, 2], 25_000))
        for window in (0.01, 0.1, 0.5, 2):
            result = signal.query(0, signal.duration_ms, str(window))
            np.testing.assert_allclose(result["y"], 0.1, rtol=1e-10)

    def test_centered_rms_includes_a_narrow_pulse(self):
        raw = np.zeros(100_000, dtype=np.int16)
        raw[49_997:50_003] = 100
        signal = self.waveform(raw, interval_ns=10, gain=1)
        index = 500  # 0.5 ms, with a 10 µs window centered on it.
        expected = np.sqrt(np.mean(raw[49_500:50_500].astype(float) ** 2))
        self.assertAlmostEqual(signal.rms[0.01][index], expected)
        self.assertAlmostEqual(signal.rms_time[index], 0.5)

    def test_rejects_empty_nonfinite_and_unknown_window_requests(self):
        signal = self.waveform([1, 2, 3])
        for lo, hi, mode in [
            (1, 1, "raw"),
            (float("nan"), 1, "raw"),
            (0, float("inf"), "raw"),
            (0, 1, "0.2"),
        ]:
            with self.subTest(lo=lo, hi=hi, mode=mode), self.assertRaises(ValueError):
                signal.query(lo, hi, mode)

    def test_http_byte_ranges_and_explicit_file_boundary(self):
        payload = b"0123456789abcdefghijklmnop"
        path = self.folder / "channel-a-adc.npy"
        path.write_bytes(payload)
        captures = {
            "reference": {
                "summary": {"title": "Test", "has_raw": False},
                "raw": path,
                "trace": path,
                "waveform": None,
            }
        }
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(Handler, captures=captures)
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for header, expected in [
                ("bytes=3-6", b"3456"),
                ("bytes=-3", b"nop"),
                ("bytes=24-999", b"op"),
            ]:
                with self.subTest(header=header), urlopen(
                    Request(base + "/data/channel-a-adc.npy", headers={"Range": header})
                ) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), expected)
                    self.assertEqual(
                        int(response.headers["Content-Length"]), len(expected)
                    )
            with urlopen(
                Request(base + "/data/channel-a-adc.npy", method="HEAD")
            ) as response:
                self.assertEqual(response.read(), b"")
                self.assertEqual(int(response.headers["Content-Length"]), len(payload))
            with self.assertRaises(HTTPError) as error:
                urlopen(
                    Request(
                        base + "/data/channel-a-adc.npy", headers={"Range": "bytes=99-"}
                    )
                )
            self.assertEqual(error.exception.code, 416)
            for route in [
                "/../serve.py",
                "/serve.py",
                "/api/summary?capture=../reference",
            ]:
                with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                    urlopen(base + route)
                self.assertEqual(error.exception.code, 404)
            with urlopen(base + "/api/catalog") as response:
                self.assertEqual(json.load(response)[0]["id"], "reference")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
