"""Numerical checks for the archived raw-sample RMS export."""
import unittest

import numpy as np

from add_fine_rms import peak_records, rms_values


class FineRMSTest(unittest.TestCase):
    def test_adc_square_does_not_overflow(self):
        raw = np.array([32767, -32768, 30000, -30000], dtype=np.int16)
        actual = rms_values(raw, 2, 0.0061)
        expected = np.sqrt(np.mean(raw.astype(float).reshape(-1, 2) ** 2, axis=1)) * 0.0061
        np.testing.assert_allclose(actual, expected, rtol=1e-15)

    def test_short_pulse_survives_in_fine_window(self):
        raw = np.zeros(5000, dtype=np.int16)
        raw[2701] = 10000
        fine = rms_values(raw, 250, 0.0061)
        microsecond = rms_values(raw, 2500, 0.0061)
        self.assertEqual(np.count_nonzero(fine), 1)
        self.assertAlmostEqual(fine[10], 61 / np.sqrt(250))
        self.assertAlmostEqual(microsecond[1], 61 / np.sqrt(2500))
        self.assertGreater(fine[10], microsecond[1])

    def test_chunk_boundary_and_incomplete_window(self):
        raw = np.full(2_000_011, -123, dtype=np.int16)
        raw[2_000_002] = 456
        actual = rms_values(raw, 250, 0.5)
        # The changed sample is in the explicitly omitted, incomplete tail.
        self.assertEqual(len(actual), len(raw) // 250)
        np.testing.assert_array_equal(actual, np.full(8000, 61.5))

    def test_peak_indexes_preserve_order_ties_and_partial_tail(self):
        values = np.array([3., 7., 1., 7., 9., 5., 4., 8., 2., 11.])
        peaks = peak_records(values, 4)
        self.assertEqual(peaks.dtype.itemsize, 24)
        self.assertEqual(peaks["low_index"].tolist(), [2, 6, 8])
        self.assertEqual(peaks["high_index"].tolist(), [1, 4, 9])
        np.testing.assert_array_equal(peaks["low"], values[peaks["low_index"]])
        np.testing.assert_array_equal(peaks["high"], values[peaks["high_index"]])


if __name__ == "__main__":
    unittest.main()
