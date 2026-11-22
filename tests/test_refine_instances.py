import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from Acquisition import AcquisitionConfig
from BatchFDTD import batch_metadata, record_batch
import RefineInstances
from SDF import Sphere, voxel_signed_distance


def pair_fixture(root):
    config = AcquisitionConfig(grid_shape=(14, 16, 17), pml_thickness=2, standoff_cells=2,
                               elements_per_side=2, pitch_cells=2, grid_spacing_m=.0045,
                               samples_per_shot=64, sample_stride=2)
    meta = batch_metadata(config)
    origin = np.array(meta["grid_origin_cells"])
    upper = np.array(config.grid_shape) - config.pml_thickness
    region = tuple(slice(lo, hi) for lo, hi in zip(origin, upper))
    points = np.moveaxis(np.indices(tuple(upper - origin)), 0, -1) + origin
    masks = np.stack([Sphere(np.array([7., 5., 7.]), 1.58).distances(points) < 0,
                      Sphere(np.array([9., 10., 10.]), 2.18).distances(points) < 0])
    values = np.array([[3.21, .048], [5.28, .094]])
    eps = np.ones((4, *config.grid_shape))
    sigma = np.zeros_like(eps)
    for i, (mask, (er, conductivity)) in enumerate(zip(masks, values)):
        for row in (i+1, 3):
            eps[(row, *region)][mask] = er
            sigma[(row, *region)][mask] = conductivity
    raw = record_batch(config, eps, sigma, device="cpu", dtype="float64")
    gain = np.linspace(.85, 1.15, raw.shape[1]*raw.shape[2]).reshape(*raw.shape[1:3], 1)
    raw *= gain
    (root / "acquisition.json").write_text(json.dumps(meta))
    np.save(root / "sensor_samples.npy", raw[3])
    np.save(root / "empty_sensor_samples.npy", raw[0])
    np.savez(root / "instances.npz", masks=masks, probability_two=.8)
    return masks, values, raw


class RefineInstancesTests(unittest.TestCase):
    def test_joint_full_wave_recovers_two_different_materials_from_raw_rf_and_selected_slots(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            masks, materials, raw = pair_fixture(root)
            target = raw[3]-raw[0]
            additive_to_joint = np.linalg.norm(raw[1]+raw[2]-2*raw[0]-target) / np.linalg.norm(target)
            self.assertGreater(additive_to_joint, 1e-3)
            seed = np.stack([np.roll(masks[0], 1, axis=1), np.roll(masks[1], -1, axis=0)])
            np.savez(root / "instances.npz", masks=seed, probability_two=.8)
            seed_union, truth = seed.any(0), masks.any(0)
            self.assertLess((seed_union & truth).sum() / (seed_union | truth).sum(), .45)
            observed_schedules = []
            def forward(config, *args, **kwargs):
                observed_schedules.append((len(kwargs["angles"]), config.sample_stride))
                return record_batch(config, *args, **kwargs)
            with patch.object(RefineInstances, "record_batch", side_effect=forward):
                arrays, report = RefineInstances.refine_instances(root, root / "instances.npz", iterations=3,
                                                                 device="cpu", dtype="float64")
            self.assertTrue(any(n < 5 for n, _ in observed_schedules))
            self.assertTrue(all(stride == 2 for _, stride in observed_schedules))
            self.assertEqual(report["selected_instances"], 2)
            self.assertEqual(report["probability_two"], .8)
            self.assertLess(report["relative_rf_l2"], min(.0004, additive_to_joint / 10))
            self.assertLess(report["relative_rf_l2"], report["initial_best_relative_rf_l2"])
            np.testing.assert_allclose(arrays["predicted_echo_v"], target, atol=.0004*np.linalg.norm(target))
            np.testing.assert_array_equal(arrays["instance_masks"], masks)
            np.testing.assert_array_equal(arrays["sdf_cells"], voxel_signed_distance(masks.any(0)).astype("f4"))
            for mask, expected in zip(masks, materials):
                np.testing.assert_allclose(arrays["permittivity"][mask], expected[0], atol=.01)
                np.testing.assert_allclose(arrays["conductivity_s_per_m"][mask], expected[1], atol=.002)

    def test_one_selected_slot_delegates_to_the_existing_single_refiner(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            masks, _, _ = pair_fixture(root)
            path = root / "one.npz"
            np.savez(path, masks=masks[:1], probability_two=.2)
            def single(scene, seed, **settings):
                self.assertEqual(scene, root)
                np.testing.assert_array_equal(np.load(seed), masks[0])
                return {"occupied": masks[0]}, {"relative_rf_l2": .03, "shape": "sphere",
                                                "candidate_relative_l2": {"sphere": {"initial": .2, "final": .03}}}
            with patch.object(RefineInstances, "refine_scene", side_effect=single) as delegated:
                arrays, report = RefineInstances.refine_instances(root, path, device="cpu")
            delegated.assert_called_once()
            self.assertEqual(report["selected_instances"], 1)
            np.testing.assert_array_equal(arrays["instance_masks"], masks[:1])

    def test_empty_predicted_slot_uses_its_own_rf_center(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "slots.npz"
            region = (slice(3, 10), slice(2, 11), slice(2, 12))
            np.savez(path, masks=np.zeros((2, 7, 9, 10), dtype=bool), centers_cells=[[2, 2, 2], [5, 7, 7]])
            slots, _ = RefineInstances._slots(path, (14, 14, 14), region)
            self.assertTrue(slots[0, 2, 2, 2] and slots[1, 5, 7, 7])
            self.assertFalse((slots[0] & slots[1]).any())
            np.savez(path, masks=np.zeros((3, 7, 9, 10), dtype=bool))
            with self.assertRaisesRegex(ValueError, "one or two"):
                RefineInstances._slots(path, (14, 14, 14), region)


if __name__ == "__main__":
    unittest.main()
