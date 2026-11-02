import unittest
import numpy as np
from metrics import threshold_metrics


class ThresholdTests(unittest.TestCase):
    def test_hand_calculated_rates_and_equal_threshold(self):
        cm, fpr, tpr = threshold_metrics([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9], 0.5)
        np.testing.assert_array_equal(cm, [[1, 1], [0, 2]])
        self.assertEqual((fpr, tpr), (0.5, 1.0))

    def test_extreme_decisions(self):
        for threshold, expected in [(0, (1, 1)), (1, (0, 0))]:
            with self.subTest(threshold=threshold):
                _, fpr, tpr = threshold_metrics([0, 1], [0.2, 0.8], threshold)
                self.assertEqual((fpr, tpr), expected)

    def test_invalid_inputs(self):
        for truth, score, threshold in [([], [], .5), ([0], [.2], .5),
                ([0, 1], [.2], .5), ([0, 1], [.2, float('nan')], .5),
                ([0, 1], [.2, 1.1], .5), ([0, 1], [.2, .9], -1)]:
            with self.subTest(truth=truth, score=score, threshold=threshold):
                with self.assertRaises(ValueError):
                    threshold_metrics(truth, score, threshold)
