import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fdtd
import numpy as np
from fdtd.constants import eps0
from fdtd.grid import curl_H

import helpers
from SDF import Box, Sphere, voxel_signed_distance
from WorldGenerator import (
    add_objects_to_grid,
    bounding_box_overlap,
    generate_objects,
    grid_init,
    install_materials,
    voxelize_world,
)
from WorldObjectType import WorldObject
from inner_functions import MaterialRun, voxelize_slab


class GeometryTests(unittest.TestCase):
    def setUp(self):
        fdtd.set_backend("numpy")

    def test_box_distance_and_bounds_use_the_same_rotation(self):
        center = np.array([3., 7., 11.])
        box = Box(center, np.array([2., 6., 14.]), np.array([35., 50., 25.]))
        for local, expected in [([4, 0, 0], 3), ([4, 7, 0], 5), ([0, 2.8, 6.8], -0.2)]:
            world = helpers.rotate_vec(np.array(local), *box.rotation) + center
            self.assertAlmostEqual(box.sdf(world), expected, places=12)

        corners = np.array([helpers.rotate_vec(np.array(signs) * box.size / 2, *box.rotation)
                            for signs in itertools.product([-1, 1], repeat=3)]) + center
        np.testing.assert_allclose(box.get_bounding_box(), [corners.min(axis=0), corners.max(axis=0)])

    def test_surface_and_last_cell_close_only_their_own_z_run(self):
        sphere = WorldObject(Sphere(np.array([8, 8, 8]), 5), np.array([54.16]), np.array([11.29]))
        for stop, expected in [(17, [(4, 13), (4, 13)]), (14, [(4, 13), (4, 13)]),
                               (12, [(4, 12), (4, 12)])]:
            labels, runs, _ = voxelize_slab([sphere], (8, 9), (8, 10), (0, stop))
            self.assertEqual([(r.y, r.z_start, r.z_stop) for r in runs],
                             [(8 + i, start, end) for i, (start, end) in enumerate(expected)])
            np.testing.assert_array_equal(labels[0, 0, 3], [1, 0, 0])
            if stop > 13:
                np.testing.assert_array_equal(labels[0, 0, 13], [1, 0, 0])

    def test_ghost_sphere_reaches_solver_in_remainder_planes(self):
        grid = grid_init((160, 44, 44), 0.00625, 1., 1., pml_thickness=15)
        sphere = WorldObject(Sphere(np.array([139., 22., 22.]), 5), 54.16, 11.29)
        _, labels = add_objects_to_grid([sphere], grid, pml_thickness=15, n_jobs=20)
        center = (139, 22, 22)
        self.assertAlmostEqual(labels[center][2], -np.sqrt(18.5))
        np.testing.assert_array_equal(labels[134, 22, 22], [1, 0, 0.5])

        actual = np.zeros(grid.shape, dtype=bool)
        for obj in grid.objects:
            actual[obj.x, obj.y, obj.z] = True
            np.testing.assert_allclose(labels[obj.x, obj.y, obj.z, 0], obj.permittivity)
            np.testing.assert_allclose(labels[obj.x, obj.y, obj.z, 1], 11.29)
        self.assertTrue(actual[center])
        np.testing.assert_array_equal(actual, labels[..., 2] < 0)
        grid.step()

    def test_solver_coefficients_and_update_match_labels_in_all_polarizations(self):
        for backend in ("numpy", "torch"):
            with self.subTest(backend=backend):
                fdtd.set_backend(backend)
                grid = grid_init((15, 16, 17), 0.003, 2., 1., pml_thickness=0)
                sphere = WorldObject(Sphere(np.array([7, 8, 8]), 4.), 54.16, 11.29)
                _, labels = add_objects_to_grid([sphere], grid, pml_thickness=0, n_jobs=2)
                inverse = fdtd.backend.numpy(grid.inverse_permittivity).copy()
                absorption = np.zeros_like(inverse)
                for obj in grid.objects:
                    region = (obj.x, obj.y, obj.z)
                    inverse[region] = fdtd.backend.numpy(obj.inverse_permittivity)
                    absorption[region] = fdtd.backend.numpy(obj.absorption_factor)

                epsilon = labels[..., 0, None]
                sigma = labels[..., 1, None]
                expected_inverse = np.broadcast_to(1. / epsilon, inverse.shape)
                expected_loss = np.broadcast_to(grid.time_step * sigma / (2 * eps0 * epsilon), absorption.shape)
                np.testing.assert_allclose(inverse, expected_inverse, rtol=1e-12)
                np.testing.assert_allclose(absorption, expected_loss, rtol=1e-12)

                rng = np.random.default_rng(42)
                electric = rng.normal(size=inverse.shape)
                grid.E[...] = fdtd.backend.array(electric)
                grid.H[...] = fdtd.backend.array(rng.normal(size=inverse.shape))
                curl = fdtd.backend.numpy(curl_H(grid.H))
                drive = grid.courant_number * expected_inverse * curl
                expected = ((1 - expected_loss) * electric + drive) / (1 + expected_loss)
                grid.update_E()
                np.testing.assert_allclose(fdtd.backend.numpy(grid.E), expected, rtol=1e-11, atol=1e-12)

    def test_worker_count_and_overlapping_materials_preserve_every_interior_label(self):
        objects = [WorldObject(Sphere(np.array([3., 5., 3.]), 2.5), 4., 0.6),
                   WorldObject(Sphere(np.array([3., 5., 6.]), 2.5), 8., 2.)]
        grid = grid_init((9, 11, 9), 0.00625, 2., 1., pml_thickness=1)
        serial, serial_runs = voxelize_world(objects, grid, pml_thickness=1, n_jobs=1)
        for workers in (3, 20):
            parallel, runs = voxelize_world(objects, grid, pml_thickness=1, n_jobs=workers)
            np.testing.assert_array_equal(serial, parallel)
            self.assertEqual(serial_runs, runs)

        for local in np.ndindex(7, 9, 7):
            xyz = tuple(np.array(local) + 1)
            distances = [obj.sdf.sdf(np.array(xyz)) for obj in objects]
            nearest = objects[np.argmin(distances)]
            distance = min(distances)
            material = (nearest.permittivity, nearest.conductivity) if distance < 0 else (2., 0.)
            np.testing.assert_allclose(serial[xyz][:2], material)
            self.assertEqual(serial[xyz][2] < 0, distance < 0)
        mask = np.zeros(grid.shape, dtype=bool)
        for run in serial_runs:
            cells = (run.x, run.y, slice(run.z_start, run.z_stop))
            self.assertFalse(mask[cells].any())
            mask[cells] = True
            np.testing.assert_array_equal(serial[cells + (0,)], run.permittivity)
            np.testing.assert_array_equal(serial[cells + (1,)], run.conductivity)
        np.testing.assert_array_equal(mask, serial[..., 2] < 0)

    def test_merged_and_overlapping_cuboids_have_identical_all_channel_targets(self):
        grid = grid_init((25, 25, 25), 0.00625, 1., 1., pml_thickness=0)

        def box(x, length):
            shape = Box(np.array([x, 12., 12.]), np.array([length, 12., 12.]), np.zeros(3))
            return WorldObject(shape, 8., 0.7)

        full = box(12., 8.)
        first, second = box(10.5, 5.), box(13.5, 5.)
        points = np.moveaxis(np.indices(grid.shape), 0, -1)
        split_distance = np.minimum(first.sdf.distances(points), second.sdf.distances(points))
        np.testing.assert_array_equal(full.sdf.distances(points) < 0, split_distance < 0)
        self.assertEqual(full.sdf.sdf(np.array([12., 12., 12.])), -4.)
        self.assertEqual(split_distance[12, 12, 12], -1.)
        one, one_runs = voxelize_world([full], grid, pml_thickness=0)
        split, split_runs = voxelize_world([first, second], grid, pml_thickness=0)
        np.testing.assert_array_equal(one, split)
        np.testing.assert_array_equal(one[..., 2] < 0, full.sdf.distances(points) < 0)
        self.assertEqual(one_runs, split_runs)
        self.assertEqual(one[12, 12, 12, 2], -3.5)

    def test_exact_voxel_distance_to_faces_edges_corners_and_cavity(self):
        one = np.zeros((7, 6, 5), dtype=bool)
        center = (3, 2, 2)
        one[center] = True
        points = np.moveaxis(np.indices(one.shape), 0, -1)
        expected = np.linalg.norm(np.maximum(np.abs(points - center) - 0.5, 0), axis=-1)
        expected[center] = -0.5
        np.testing.assert_array_equal(voxel_signed_distance(one), expected)

        full = np.ones_like(one)
        from_surface = np.minimum(points + 0.5, np.array(one.shape) - points - 0.5)
        np.testing.assert_array_equal(voxel_signed_distance(full), -from_surface.min(axis=-1))
        full[center] = False
        self.assertEqual(voxel_signed_distance(full)[center], 0.5)
        self.assertTrue(np.isposinf(voxel_signed_distance(np.zeros_like(one))).all())

    def test_quarter_turn_has_the_same_surface_and_material_runs_as_axis_aligned_box(self):
        grid = grid_init((25, 25, 25), 0.00625, 2., 1., pml_thickness=0)
        center = np.array([12., 12., 12.])
        rotated = Box(center, np.array([4., 8., 12.]), np.array([0., 0., 90.]))
        aligned = Box(center, np.array([8., 4., 12.]), np.zeros(3))
        labels, runs = voxelize_world([WorldObject(rotated, 8., 0.7)], grid, 0)
        reference, reference_runs = voxelize_world([WorldObject(aligned, 8., 0.7)], grid, 0)
        np.testing.assert_array_equal(labels, reference)
        points = np.moveaxis(np.indices(grid.shape), 0, -1)
        surface = aligned.distances(points) == 0
        self.assertTrue(np.all(labels[surface, 0] == 2.))
        self.assertTrue(np.all(labels[surface, 1] == 0.))
        self.assertTrue(np.all(labels[surface, 2] >= 0.5))
        self.assertEqual(labels[16, 12, 12, 2], 0.5)
        self.assertEqual(runs, reference_runs)

        expanded = Box(center, np.array([8. + 2e-9, 4., 12.]), np.zeros(3))
        labels, _, _ = voxelize_slab([WorldObject(expanded, 8., 0.7)], (16, 17), (12, 13), (12, 13))
        self.assertAlmostEqual(labels[0, 0, 0, 2], -1e-9, places=14)
        self.assertEqual(labels[0, 0, 0, 0], 8.)

    def test_empty_scene_and_zero_thickness_export(self):
        grid = grid_init((11, 13, 17), 0.00625, 2., 1., pml_thickness=0)
        _, labels = add_objects_to_grid([], grid, pml_thickness=0)
        np.testing.assert_array_equal(labels[..., 0], 2)
        np.testing.assert_array_equal(labels[..., 1], 0)
        self.assertTrue(np.isposinf(labels[..., 2]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            helpers.save_grid(path, labels, 0, 3)
            np.testing.assert_array_equal(np.load(path / "grid.npy"), labels[3:])
            helpers.save_grid(path, labels, 2, 3)
            np.testing.assert_array_equal(np.load(path / "grid.npy"), labels[3:-2, 2:-2, 2:-2])

    def test_unresolved_object_cannot_supply_labels_for_an_empty_field(self):
        grid = grid_init((12, 12, 12), 0.00625, 1., 1., pml_thickness=1)
        for center in [[20, 20, 20], [5.5, 5.5, 5.5]]:
            obj = WorldObject(Sphere(np.array(center), 0.2), 4., 0.)
            with self.assertRaisesRegex(ValueError, "no occupied voxel"):
                add_objects_to_grid([obj], grid, pml_thickness=1)
            self.assertEqual(grid.objects, [])

    def test_material_identical_to_grid_background_cannot_supply_a_surface(self):
        # The background value recovered from its FDTD inverse differs from
        # nominal 7.7 by one ulp. Both still have identical update coefficients.
        invisible = WorldObject(Sphere(np.array([5., 5., 5.]), 2.), 7.7, 0.)
        visible = WorldObject(Sphere(np.array([8., 6., 6.]), 2.), 4., 1.)
        for backend in ("numpy", "numpy.float32", "torch.float32"):
            with self.subTest(backend=backend):
                fdtd.set_backend(backend)
                grid = grid_init((12, 12, 12), 0.00625, 7.7, 1., pml_thickness=1)
                for objects in ([invisible], [visible, invisible]):
                    with self.assertRaisesRegex(ValueError, "no occupied voxel centers with material contrast"):
                        add_objects_to_grid(objects, grid, pml_thickness=1)
                    self.assertEqual(grid.objects, [])
                for eps, sigma in ((7.7, 0.25), (3., 0.)):
                    material = WorldObject(invisible.sdf, eps, sigma)
                    labels, runs = voxelize_world([material], grid, pml_thickness=1)
                    self.assertTrue(runs)
                    np.testing.assert_allclose(labels[5, 5, 5, :2], [eps, sigma], rtol=1e-7)
                    self.assertLess(labels[5, 5, 5, 2], 0)

    def test_random_objects_fit_sensor_crop_and_avoid_array(self):
        grid = grid_init((72, 64, 64), 0.00625, 1., 1., pml_thickness=8)
        sensor_box = np.array([[12., 14., 14.], [13., 50., 50.]])
        sensor = SimpleNamespace(position=np.array([12., 32., 32.]),
                                 get_bounding_box=lambda: sensor_box)
        state = np.random.get_state()
        try:
            np.random.seed(11)
            objects = generate_objects(grid, 40, 8, sensor)
        finally:
            np.random.set_state(state)
        self.assertEqual(len(objects), 40)
        for obj in objects:
            bounds = obj.sdf.get_bounding_box()
            self.assertTrue((bounds[0] >= [12, 8, 8]).all())
            self.assertTrue((bounds[1] <= [63, 55, 55]).all())
            self.assertFalse(bounding_box_overlap(bounds, sensor_box))

    def test_generator_accepts_intersecting_objects(self):
        grid = grid_init((60, 60, 60), 0.00625, 1., 1., pml_thickness=0)
        sensor = SimpleNamespace(position=np.array([6., 30., 30.]), get_bounding_box=lambda:
                                 np.array([[6., 15., 15.], [7., 45., 45.]]))
        proposed_x = [24, 33]  # radius-five spheres overlap by one cell
        draws = [value for x in proposed_x for value in (0, np.array([x, 24, 24]))]
        with patch("WorldGenerator.get_random_scalar", return_value=5.), patch(
                "WorldGenerator.np.random.randint", side_effect=draws):
            objects = generate_objects(grid, 2, 0, sensor)
        self.assertEqual([obj.sdf.position[0] for obj in objects], proposed_x)

    def test_physical_generator_units_preserve_dimensions_material_and_rotation_on_refinement(self):
        unit = 0.00625
        sensor = SimpleNamespace(position=np.zeros(3), get_bounding_box=lambda: np.array([[0, 0, 0], [1, 1, 1]]))
        state = np.random.get_state()
        try:
            for seed, shape_type in [(0, Sphere), (1, Box)]:
                objects = []
                for cells, spacing, size_unit in [(120, unit, None), (240, unit / 2, unit),
                                                   (240, unit / 2, None)]:
                    grid = SimpleNamespace(grid_spacing=spacing, E=SimpleNamespace(shape=(cells, cells, cells, 3)))
                    np.random.seed(seed)
                    objects.extend(generate_objects(grid, 1, 3, sensor, size_unit_m=size_unit))
                coarse, refined, default_refined = objects
                self.assertTrue(all(isinstance(obj.sdf, shape_type) for obj in objects))
                if shape_type is Sphere:
                    sizes = [obj.sdf.radius for obj in objects]
                else:
                    sizes = [obj.sdf.size for obj in objects]
                    np.testing.assert_array_equal(coarse.sdf.rotation, refined.sdf.rotation)
                np.testing.assert_allclose(sizes[0] * unit, sizes[1] * unit / 2)
                np.testing.assert_array_equal(sizes[2], sizes[0])
                self.assertEqual(coarse.permittivity, refined.permittivity)
                self.assertEqual(coarse.conductivity, refined.conductivity)
        finally:
            np.random.set_state(state)

    def test_failed_registration_leaves_a_reusable_field_grid(self):
        grid = grid_init((12, 12, 12), 0.00625, 1., 1., pml_thickness=0)
        initial = grid.inverse_permittivity.copy()
        first = MaterialRun(4, 4, 4, 7, "duplicate", 4., 0.3)
        second = MaterialRun(6, 6, 6, 9, "duplicate", 8., 1.1)
        with self.assertRaises(ValueError):
            install_materials(grid, [first, second])
        self.assertEqual(grid.objects, [])
        self.assertFalse(hasattr(grid, "duplicate"))
        np.testing.assert_array_equal(grid.inverse_permittivity, initial)
        install_materials(grid, [first])
        self.assertEqual(len(grid.objects), 1)
        grid.step()


if __name__ == "__main__":
    unittest.main()
