import unittest

import fdtd
import numpy as np

from Acquisition import _layout
from BatchFDTD import batch_metadata
from PhantomBenchmark import configuration, draw_phantom, rasterize


class PhantomTests(unittest.TestCase):
    def test_balanced_independent_reproducible_draws(self):
        for i in range(0, 32, 2):
            _, first = draw_phantom(i, 17, "dielectric")
            _, second = draw_phantom(i + 1, 17, "dielectric")
            self.assertEqual(sorted([first[0]["shape"], second[0]["shape"]]), ["box", "sphere"])
            self.assertEqual(first, draw_phantom(i, 17, "dielectric")[1])
            other = draw_phantom(i, 18, "dielectric")[1]
            self.assertNotEqual(first[0]["center_cells"], other[0]["center_cells"])
            self.assertTrue(2.5 <= first[0]["permittivity"] <= 6.0)
            self.assertTrue(0.01 <= first[0]["conductivity_s_per_m"] <= 0.12)

    def test_full_reference_and_benchmark_targets_agree(self):
        fdtd.set_backend("numpy")
        for material, i, count in (("dielectric", 0, 1), ("dielectric", 1, 1), ("thesis", 3, 2)):
            config = configuration(material)
            origin = batch_metadata(config)["grid_origin_cells"]
            world, _ = draw_phantom(i, 731, material, count)
            expected, _ = _layout(config, world, count, 1)
            si = tuple(slice(v, n - config.pml_thickness) for v, n in zip(origin, config.grid_shape))
            np.testing.assert_array_equal(rasterize(config, origin, world), expected[si])
            if count == 2:
                a, b = [obj.sdf.get_bounding_box() for obj in world]
                self.assertTrue(((a[1] + 1 < b[0]) | (b[1] + 1 < a[0])).any())


if __name__ == "__main__":
    unittest.main()
