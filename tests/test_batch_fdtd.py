"""Precision tests and a repeatable parity/timing report.

From the simulator root run
PYTHONPATH=. python tests/test_batch_fdtd.py --compare --device cuda --fixture benchmark
to print per-beam relative errors, peak waveform errors and elapsed times as JSON.
"""
import argparse
import json
import os
import sys
from time import perf_counter
import unittest

import fdtd
import numpy as np
import torch

from Acquisition import AcquisitionConfig, BEAMS, _layout, _record_shots
from BatchFDTD import batch_metadata, record_batch
from SDF import Sphere
from WorldObjectType import WorldObject


def reference_pair(config, epsilon, sigma, center=(13., 10., 10.), radius=2.25, timings=None):
    fdtd.set_backend("numpy")
    sphere = WorldObject(Sphere(np.array(center), radius), epsilon, sigma)
    grid, runs = _layout(config, [sphere], n_objects=1, n_jobs=1)
    occupied = grid[..., 2] < 0
    eps = np.ones((2, *config.grid_shape))
    sig = np.zeros_like(eps)
    eps[1, occupied] = grid[..., 0][occupied]
    sig[1, occupied] = grid[..., 1][occupied]
    start = perf_counter()
    vacuum, metadata = _record_shots(config, (), BEAMS)
    split = perf_counter()
    material, _ = _record_shots(config, runs, BEAMS)
    if timings is not None:
        timings.update(vacuum_s=split - start, material_s=perf_counter() - split)
    return eps, sig, np.stack([vacuum, material]), metadata


def relative_per_beam(test, reference):
    axes = (1, 3)  # independent errors for each scene and steering direction
    return np.sqrt(np.sum((test - reference) ** 2, axis=axes) / np.sum(reference ** 2, axis=axes))


class BatchFDTDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AcquisitionConfig(grid_shape=(20, 20, 20), pml_thickness=3, grid_spacing_m=.00625,
                                       pitch_cells=2, samples_per_shot=96)
        cls.eps, cls.sig, cls.expected, cls.metadata = reference_pair(cls.config, 4.5, .06)

    def check_pair(self, device, dtype, raw_tolerance, echo_tolerance):
        actual = record_batch(self.config, self.eps, self.sig, device=device, dtype=dtype).astype(np.float64)
        self.assertEqual(actual.shape, (2, 16, 5, 96))
        echo = actual[1:] - actual[:1]
        reference_echo = self.expected[1:] - self.expected[:1]
        self.assertTrue((np.sum(reference_echo ** 2, axis=(1, 3)) > 0).all())
        self.assertLess(relative_per_beam(actual, self.expected).max(), raw_tolerance)
        self.assertLess(relative_per_beam(echo, reference_echo).max(), echo_tolerance)

    def test_cpu_fields_and_differential_echo_match_all_reference_shots(self):
        self.assertEqual(batch_metadata(self.config), self.metadata)
        self.check_pair("cpu", "float64", 5e-12, 5e-10)
        self.check_pair("cpu", "float32", 3e-6, 3e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fields_and_differential_echo_match_all_reference_shots(self):
        self.check_pair("cuda", "float64", 5e-12, 5e-10)
        self.check_pair("cuda", "float32", 3e-6, 3e-4)

    def test_refined_conductive_material_matches_five_reference_echoes(self):
        config = AcquisitionConfig(grid_shape=(20, 20, 20), pml_thickness=3, standoff_cells=3,
                                   grid_spacing_m=.0011785, pitch_cells=2, samples_per_shot=128)
        eps, sig, expected, _ = reference_pair(config, 54.16, 11.29)
        actual = record_batch(config, eps, sig, device="cpu", dtype="float64")
        self.assertLess(relative_per_beam(actual, expected).max(), 5e-12)
        self.assertLess(relative_per_beam(actual[1:] - actual[:1], expected[1:] - expected[:1]).max(), 5e-9)

    def test_without_pml_and_duplicate_shots(self):
        config = AcquisitionConfig(grid_shape=(16, 18, 19), pml_thickness=0, standoff_cells=2,
                                   pitch_cells=2, elements_per_side=2, grid_spacing_m=.00625, samples_per_shot=90)
        eps = np.ones((1, *config.grid_shape))
        angles = ((0., 0.), (0., 30.), (0., 0.))
        expected, _ = _record_shots(config, (), angles)
        actual = record_batch(config, eps, np.zeros_like(eps), device="cpu", dtype="float64", angles=angles)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        np.testing.assert_array_equal(actual[:, 0], actual[:, 2])

    def test_reject_invalid_dense_materials(self):
        eps = self.eps.copy()
        eps[0, 0, 10, 10] = 2.
        with self.assertRaisesRegex(ValueError, "PML"):
            record_batch(self.config, eps, self.sig, device="cpu")
        with self.assertRaisesRegex(ValueError, "shape"):
            record_batch(self.config, eps[0], self.sig[0], device="cpu")
        with self.assertRaisesRegex(ValueError, "positive"):
            record_batch(self.config, self.eps, -self.sig, device="cpu")


def compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fixture", choices=("small", "benchmark", "thesis"), default="benchmark")
    args = parser.parse_args()
    if args.fixture == "small":
        config = AcquisitionConfig(grid_shape=(20, 20, 20), pml_thickness=3, standoff_cells=3,
                                   grid_spacing_m=.00625, pitch_cells=2, samples_per_shot=96)
        center, radius = (13., 10., 10.), 2.25
    else:
        config = AcquisitionConfig(grid_shape=(40, 40, 40), pml_thickness=4, standoff_cells=3,
                                   grid_spacing_m=.0045 if args.fixture == "benchmark" else None,
                                   pitch_cells=8, samples_per_shot=128)
        center, radius = (23., 21., 18.), 4.25
    epsilon, sigma = (54.16, 11.29) if args.fixture == "thesis" else (4.5, .06)
    timings = {}
    eps, sig, expected, metadata = reference_pair(config, epsilon, sigma, center, radius, timings)
    report = dict(fixture=args.fixture, device=args.device, grid_shape=config.grid_shape,
                  grid_spacing_m=config.spacing, pml_cells=config.pml_thickness,
                  center_cells=center, radius_cells=radius, permittivity=epsilon, conductivity_s_per_m=sigma,
                  antennas=expected.shape[1], beam_angles_deg=BEAMS, samples_per_shot=config.samples_per_shot,
                  stride=metadata["sample_stride"], reference_numpy_wall_s=timings,
                  omp_num_threads=os.environ.get("OMP_NUM_THREADS", "default"), native={})
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    # Exclude first-use JIT and layout creation from steady-state timings.
    # Only one vacuum and one material scene enter each measured two-scene batch.
    record_batch(config, eps[:1], sig[:1], device=args.device, angles=BEAMS[:1])
    batch_metadata(config)
    for dtype in ("float32", "float64"):
        start = perf_counter()
        raw = record_batch(config, eps, sig, device=args.device, dtype=dtype)
        elapsed = perf_counter() - start
        raw = raw.astype(np.float64)
        truth = dict(vacuum=expected[:1], material=expected[1:], echo=expected[1:] - expected[:1])
        results = dict(vacuum=raw[:1], material=raw[1:], echo=raw[1:] - raw[:1])
        errors = {key: dict(relative_l2_per_beam=relative_per_beam(results[key], value)[0].tolist(),
                            peak_abs_error_v=float(np.max(np.abs(results[key] - value))),
                            peak_reference_v=float(np.max(np.abs(value)))) for key, value in truth.items()}
        report["native"][dtype] = dict(both_scenes_wall_s=elapsed, errors=errors)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    else:
        unittest.main()
