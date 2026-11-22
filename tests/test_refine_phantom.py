import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import RefinePhantom
from Acquisition import AcquisitionConfig
from BatchFDTD import batch_metadata, record_batch
from RefinePhantom import Shape, _read_initial, refine_scene
from SDF import Box, Sphere, voxel_signed_distance


class RefinePhantomTests(unittest.TestCase):
    def test_rotated_box_seed_fits_an_occupied_volume(self):
        points = np.moveaxis(np.indices((19, 20, 21)), 0, -1)
        box = Box(np.array([9.2, 10.4, 10.3]), np.array([9., 5., 7.]), np.array([15., 33., 47.]))
        mask = box.distances(points) < 0
        model = Shape("box", points[0, 0, 0] - .5, points[-1, -1, -1] + .5)
        p = model.start(mask, points)
        fitted = model.distance(p, points) < 0
        iou = np.sum(fitted & mask) / np.sum(fitted | mask)
        self.assertGreater(iou, .98)

    def test_prediction_reader_accepts_cropped_sdf_and_full_boolean_npz(self):
        full_shape = (14, 15, 16)
        region = (slice(3, 12), slice(2, 13), slice(2, 14))
        mask = np.zeros(full_shape, dtype=bool)
        mask[6:9, 6:9, 6:9] = True
        mask[4, 3, 3] = True
        with TemporaryDirectory() as temp:
            path = Path(temp)
            np.savez(path / "prediction.npz", occupied=mask)
            np.save(path / "prediction.npy", voxel_signed_distance(mask[region]))
            np.save(path / "binary.npy", mask[region].astype(np.uint8))
            expected = mask[region].copy()
            expected[1, 1, 1] = False
            for file in path.iterdir():
                np.testing.assert_array_equal(_read_initial(file, region, full_shape), expected)
            with self.assertRaisesRegex(ValueError, "RF-only"):
                refine_scene(path, path / "grid.npy")

    def test_true_forward_echo_recovers_materials_from_raw_pair_only(self):
        config = AcquisitionConfig(grid_shape=(14, 15, 16), pml_thickness=2, standoff_cells=2,
                                   elements_per_side=2, pitch_cells=2, grid_spacing_m=.0045, samples_per_shot=64)
        meta = batch_metadata(config)
        origin = np.array(meta["grid_origin_cells"])
        upper = np.array(config.grid_shape) - config.pml_thickness
        region = tuple(slice(lo, hi) for lo, hi in zip(origin, upper))
        points = np.moveaxis(np.indices(tuple(upper - origin)), 0, -1) + origin
        mask = Sphere(np.array([8., 7., 8.]), 2.41).distances(points) < 0
        eps = np.ones((2, *config.grid_shape))
        sigma = np.zeros_like(eps)
        eps[(1, *region)][mask] = 4.91
        sigma[(1, *region)][mask] = .081
        raw = record_batch(config, eps, sigma, dtype="float64", device="cpu")
        gains = np.linspace(.85, 1.15, raw.shape[1] * raw.shape[2]).reshape(raw.shape[1], raw.shape[2], 1)
        with TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "acquisition.json").write_text(json.dumps(meta))
            np.save(path / "sensor_samples.npy", raw[1] * gains)
            np.save(path / "empty_sensor_samples.npy", raw[0] * gains)
            np.savez(path / "prediction.npz", sdf_cells=voxel_signed_distance(mask))
            schedules = []
            def check_stride(c, *args, **kwargs):
                schedules.append((len(kwargs["angles"]), c.sample_stride))
                return record_batch(c, *args, **kwargs)
            with patch.object(RefinePhantom, "record_batch", side_effect=check_stride):
                arrays, report = refine_scene(path, path / "prediction.npz", device="cpu", dtype="float64",
                                              iterations=1, batch_size=20)
        self.assertTrue(any(beams < 5 for beams, _ in schedules))
        self.assertTrue(all(stride == meta["sample_stride"] for _, stride in schedules))
        self.assertEqual(report["shape"], "sphere")
        np.testing.assert_array_equal(arrays["occupied"], mask)
        self.assertLess(report["relative_rf_l2"], 1e-4)
        self.assertAlmostEqual(report["permittivity"], 4.91, places=2)
        self.assertAlmostEqual(report["conductivity_s_per_m"], .081, places=3)
        self.assertLess(report["empty_calibration_relative_l2"], 1e-12)


if __name__ == "__main__":
    unittest.main()
