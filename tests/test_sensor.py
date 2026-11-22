import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fdtd.constants import c
from fdtd.conversions import worldE_to_simE
import numpy as np
import torch

from Acquisition import AcquisitionConfig, BEAMS, material_grid_spacing, record_shots, write_scene
from SDF import Sphere
from Sensor import PhasedSensorDevice
from WorldGenerator import grid_init
from WorldObjectType import WorldObject


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.f = 2.4e9
        self.dx = c / self.f / 20

    def sensor(self, angle=(0, 0), **kwargs):
        return PhasedSensorDevice(np.array([18, 30, 30]), kwargs.pop("rotation", np.zeros(3)),
                                  antenna_spacing=10, frequency=self.f, grid_spacing=self.dx,
                                  beamforming_angle=angle, **kwargs)

    def test_constructor_and_resteer_match_in_metres(self):
        sensor = self.sensor((30, 0))
        def get_times():
            return np.array([a.start_time for a in sensor.antennas]).reshape(4, 4)

        azimuth = get_times()
        sensor.recalculate_timings((30, 0))
        np.testing.assert_array_equal(get_times(), azimuth)
        # The high-y edge is 1.5 vacuum wavelengths from the low-y edge.
        delays = np.arange(4) * 0.25 / self.f
        np.testing.assert_allclose(azimuth, np.broadcast_to(delays[:, None], (4, 4)), atol=1e-24)
        sensor.recalculate_timings((0, -30))
        np.testing.assert_allclose(get_times(), np.broadcast_to(delays[::-1], (4, 4)), atol=1e-24)
        rotations = self.sensor((30, 0), rotation=np.array([0., 0., 90.]))
        rotated_delays = np.array([a.start_time for a in rotations.antennas]).reshape(4, 4)
        np.testing.assert_allclose(rotated_delays, azimuth, atol=1e-24)
        sensor.recalculate_timings((0, 0))
        np.testing.assert_array_equal(get_times(), np.zeros((4, 4)))

    def test_receiver_si_line_integral_and_polarization(self):
        sensor = self.sensor(n_antennas=1)
        electric = np.zeros((22, 33, 33, 3))
        # Endpoint fields are 1 and 3 V/m along y. A larger perpendicular
        # component should have no effect on a y-oriented element.
        electric[18, 30, 30] = worldE_to_simE(np.array([42., 1., 0.]))
        electric[18, 31, 30] = worldE_to_simE(np.array([42., 3., 0.]))
        for field in (electric, torch.from_numpy(electric)):
            sensor.calc_voltage_for_antennas(field)
        np.testing.assert_allclose(sensor.antennas[0].E_samples, [2 * self.dx] * 2)
        np.testing.assert_array_equal(sensor.get_bounding_box(), [[18, 30, 30], [18, 31, 30]])
        axial = self.sensor(element_polarization=(1, 0, 0), n_antennas=1)
        axial.calc_voltage_for_antennas(electric)
        self.assertAlmostEqual(axial.antennas[0].E_samples[0], 21. * self.dx)

    def test_even_array_with_one_cell_pitch_has_distinct_emitters(self):
        sensor = PhasedSensorDevice(np.array([4, 8, 8]), np.zeros(3))
        self.assertEqual(len({tuple(a.position) for a in sensor.antennas}), 16)

    def test_field_probes_follow_the_commanded_steering_axis(self):
        dx = c / self.f / 10
        # Probe the radiated field on opposite sides of broadside in empty air.
        for angle, positive, negative in [
                ((30, 0), (18, 24, 16), (18, 10, 16)),
                ((0, 30), (18, 17, 23), (18, 17, 9))]:
            with self.subTest(angle=angle):
                grid = grid_init((29, 33, 33), dx, 1., 1., pml_thickness=3)
                sensor = PhasedSensorDevice(np.array([5, 16, 16]), np.zeros(3), n_antennas=2,
                                            antenna_spacing=5, frequency=self.f, grid_spacing=dx,
                                            beamforming_angle=angle)
                grid[0, 0, 0] = sensor
                exposure = np.zeros(2)
                for _ in range(63):
                    grid.step()
                    exposure += np.array([float(grid.E[p][1]) for p in (positive, negative)]) ** 2
                self.assertGreater(exposure[0], 2 * exposure[1])


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AcquisitionConfig(grid_shape=(24, 24, 24), pml_thickness=3,
                                       grid_spacing_m=0.00625, pitch_cells=3, samples_per_shot=100)

    def test_repeated_broadside_shots_have_identical_traces(self):
        data = record_shots(self.config, angles=((0, 0), (30, 0), (0, 0)))
        np.testing.assert_array_equal(data[:, 0], data[:, 2])
        self.assertGreater(np.abs(data[:, 0] - data[:, 1]).max(), 0.)

    def test_five_shot_scene_and_matched_empty_reference(self):
        obj = WorldObject(Sphere(np.array([15, 12, 12]), 2), 54.16, 11.29)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube"
            write_scene(path, self.config, [obj], n_jobs=1)
            raw = np.load(path / "sensor_samples.npy")
            empty = np.load(path / "empty_sensor_samples.npy")
            labels = np.load(path / "grid.npy")
            metadata = json.loads((path / "acquisition.json").read_text())
            self.assertEqual(raw.shape, (16, 5, 100))
            self.assertEqual(empty.shape, raw.shape)
            self.assertTrue(np.isfinite(raw).all() and np.isfinite(empty).all())
            self.assertGreater(np.abs(raw - empty).max(), 0.)
            self.assertEqual(labels.shape, (15, 18, 18, 3))
            self.assertTrue((labels[..., 2] < 0).any())
            self.assertEqual(metadata["beamforming_angles_deg"], [list(a) for a in BEAMS])
            self.assertEqual(metadata["sensor_layout"], "antenna,direction,timestep")
            self.assertEqual(metadata["voltage_units"], "V")
            self.assertEqual(metadata["acquisition_version"], 2)
            self.assertEqual(metadata["fdtd_version"], "0.2.7")
            self.assertRegex(metadata["fdtd_source_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(metadata["sdf_target"], "voxel_union_boundary_cells")
            # The radius-two sphere occupies a 3x3x3 block of voxel centers.
            # Measure from those centers to the exposed faces, in grid cells.
            x, y, z = np.array([15, 12, 12]) - metadata["grid_origin_cells"]
            np.testing.assert_allclose(labels[x:x + 4, y, z, 2], [-1.5, -0.5, 0.5, 1.5])
            with self.assertRaises(FileExistsError):
                write_scene(path, self.config, [obj])

    def test_material_resolution(self):
        from fdtd.constants import eps0

        f = self.config.frequency_hz
        dx = material_grid_spacing(f)
        for er in (24.16, 83.16):
            for sigma in (6.29, 15.29):
                k = 2 * np.pi * f / c * np.sqrt(er - 1j * sigma / (2 * np.pi * f * eps0))
                self.assertGreaterEqual(2 * np.pi / k.real / dx, 10. - 1e-10)
                self.assertGreaterEqual(1 / abs(k.imag) / dx, 2.)

    def test_standoff_and_recording_window_follow_the_mesh(self):
        nominal = AcquisitionConfig()
        self.assertEqual(nominal.standoff, 16)
        self.assertEqual(self.config.standoff, 3)
        self.assertEqual(replace(self.config, standoff_cells=5).standoff, 5)
        # A stride of one would record all source pulses in this grid, but
        # would terminate before the latest single-scatter round trip.
        with self.assertRaisesRegex(ValueError, "round trip"):
            record_shots(replace(self.config, sample_stride=1), angles=((0, 0),))


if __name__ == "__main__":
    unittest.main()
