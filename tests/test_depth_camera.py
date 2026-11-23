import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from BenchmarkProvenance import _DEPTH_SIDECAR_REVISIONS, same_acquisition_manifest
from DepthCamera import DepthCamera, generate


class DepthCameraTests(unittest.TestCase):
    def setUp(self):
        self.shape = (8, 9, 9)
        self.acq = dict(grid_origin_cells=[7, 4, 11], grid_spacing_m=0.01,
                        antenna_positions_cells=[[7, 7, 14], [7, 9, 16]])
        self.camera = DepthCamera.from_acquisition(self.acq, self.shape, width=5, height=5)
        self.empty = np.zeros(self.shape, dtype=bool)

    def test_first_surface_and_optical_axis_units(self):
        self.assertFalse(self.camera.render(self.empty).any())
        scene = self.empty.copy()
        scene[3] = True
        front = self.camera.render(scene)
        np.testing.assert_allclose(front, 0.025, atol=1e-8)
        self.assertEqual(front.dtype, np.float32)
        scene[5:7] = True
        np.testing.assert_array_equal(self.camera.render(scene), front)
        camera = self.camera.calibration
        np.testing.assert_allclose(camera["position_m"], np.array([7, 8, 15]) * 0.01)
        self.assertAlmostEqual(np.linalg.det(camera["camera_to_world_rotation"]), 1)

    def test_oblique_ray_can_enter_side_of_voxel(self):
        scene = self.empty.copy()
        scene[4, 2, 4] = True
        depth = self.camera.render(scene)
        # Right-hand pixel follows y=4-0.4t and crosses y=2.5 at t=3.75 cells.
        self.assertAlmostEqual(depth[2, 3], 0.0375)
        self.assertEqual(np.count_nonzero(depth), 1)
        self.assertFalse(self.camera.render(self.empty).any())

    def test_output_for_saved_scenes_contains_only_depth_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            scene = base / "archive/scene_000000"
            scene.mkdir(parents=True)
            (scene / "acquisition.json").write_text(json.dumps(self.acq))
            field = np.zeros((*self.shape, 3), np.float32)
            field[..., 2] = 1
            field[3, ..., 2] = -1
            np.save(scene / "grid.npy", field)
            output = base / "frames"
            generate(scene.parent, output, width=5, height=5)
            saved = output / scene.name / "depth.npy"
            depth = np.load(saved)
            np.testing.assert_array_equal(depth, self.camera.render(field[..., 2] < 0))
            # Vary the RF material in every occupied cell and add a hidden object.
            field[3, ..., :2] = [90, 8]
            field[6, ..., :] = [3, 0.2, -1]
            np.save(scene / "grid.npy", field)
            generate(scene, output, width=5, height=5)
            np.testing.assert_array_equal(np.load(saved), depth)
            self.assertEqual(set(output.iterdir()), {output / "camera.json", saved.parent})
            self.assertEqual(list(saved.parent.iterdir()), [saved])
            self.assertEqual(json.loads((output / "camera.json").read_text()), self.camera.calibration)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                generate(scene, output, width=7, height=5)
            np.testing.assert_array_equal(np.load(saved), depth)
            # The simulator encodes empty geometry as an all-positive-infinity SDF.
            field[..., 2] = np.inf
            np.save(scene / "grid.npy", field)
            generate(scene, output, width=5, height=5)
            self.assertFalse(np.load(saved).any())

    def test_old_rf_manifest_still_requires_identical_physics(self):
        previous, updated = _DEPTH_SIDECAR_REVISIONS["Acquisition.py"]
        old = {"seed": 42, "sources": {"Acquisition.py": previous, "Sensor.py": "same"}}
        new = {"seed": 42, "sources": {"Acquisition.py": updated, "Sensor.py": "same"}}
        self.assertTrue(same_acquisition_manifest(old, new, "sources"))
        self.assertFalse(same_acquisition_manifest(old, {**new, "seed": 43}, "sources"))
        new["sources"]["Sensor.py"] = "different"
        self.assertFalse(same_acquisition_manifest(old, new, "sources"))
        new["sources"] = {"Acquisition.py": "unknown revision", "Sensor.py": "same"}
        self.assertFalse(same_acquisition_manifest(old, new, "sources"))

    def test_rejects_invalid_observations(self):
        with self.assertRaisesRegex(ValueError, "Boolean"):
            self.camera.render(np.zeros(self.shape))
        scene = self.empty.copy()
        scene[0, 4, 4] = True
        with self.assertRaisesRegex(ValueError, "camera center"):
            self.camera.render(scene)
        with self.assertRaisesRegex(ValueError, "field of view"):
            DepthCamera.from_acquisition(self.acq, self.shape, hfov_deg=180)


if __name__ == "__main__":
    unittest.main()
