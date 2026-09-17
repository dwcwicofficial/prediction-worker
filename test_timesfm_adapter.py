#!/usr/bin/env python3
import unittest
from timesfm_adapter import TimesFMAdapter, fallback_forecast, price_path_to_prob

class TestTimesFMAdapter(unittest.TestCase):
    def test_fallback_shapes(self):
        fc = fallback_forecast([0.4, 0.41, 0.42, 0.44], horizon=6)
        self.assertEqual(len(fc["point"]), 6)
        self.assertEqual(fc["backend"], "fallback")
    def test_prob_bounds(self):
        out = price_path_to_prob([0.2 + 0.02 * i for i in range(20)], 0.45)
        self.assertGreaterEqual(out["estimated_prob"], 0.05)
        self.assertLessEqual(out["estimated_prob"], 0.95)
    def test_rising_series_positive_edge(self):
        out = price_path_to_prob([0.30 + 0.02 * i for i in range(16)], 0.40)
        self.assertGreater(out["forecast_mid"], 0.40)

if __name__ == "__main__":
    unittest.main()
