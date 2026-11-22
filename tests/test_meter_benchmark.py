import unittest

import numpy as np

from MeterBenchmark import CONFIG, insert, phantom


class MeterGeometryTests(unittest.TestCase):
    def test_distinct_seed_index_and_balanced_shapes(self):
        for i in range(0, 24, 2):
            first = phantom(7, i)[1]
            other = phantom(7, i + 1)[1]
            self.assertNotEqual(first["kind"], other["kind"])
            np.testing.assert_array_equal(first["centers_cells"], phantom(7, i)[1]["centers_cells"])
            self.assertFalse(np.array_equal(first["centers_cells"], phantom(8, i)[1]["centers_cells"]))

    def test_dense_materials_match_exact_primitive(self):
        for i in (0, 1):
            shape, label = phantom(17, i)
            eps = np.ones(CONFIG.grid_shape, dtype=np.float32)
            sig = np.zeros_like(eps)
            insert(eps, sig, shape, label)
            occupied = np.argwhere(sig != 0)
            self.assertGreater(len(occupied), 50)
            np.testing.assert_array_equal(eps != 1, sig != 0)
            self.assertTrue(np.all(shape.distances(occupied) < 0))
            np.testing.assert_allclose(occupied.mean(0), label["centers_cells"], atol=.25)
            self.assertTrue(np.all(occupied.min(0) >= [13,8,8]))
            self.assertTrue(np.all(occupied.max(0) < 138))
