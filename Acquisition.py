"""Generate paired five-shot RF scenes and empty-reference recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fdtd
import numpy as np
from fdtd.constants import c, eps0

from Sensor import PhasedSensorDevice
from WorldGenerator import (
    SDF_TARGET, bounding_box_overlap, generate_objects, grid_init, install_materials, voxelize_world,
)
from helpers import save_grid


BEAMS = ((0., 0.), (-30., 0.), (30., 0.), (0., -30.), (0., 30.))
ACQUISITION_VERSION = 2
# The fork and upstream releases can report the same package version. Identify
# the solver code actually in use for both source checkouts and pip installs.
FDTD_SOURCE_SHA256 = sha256(b"".join(
    (Path(fdtd.__file__).parent / name).read_bytes()
    for name in ("backend.py", "boundaries.py", "constants.py", "conversions.py", "grid.py", "objects.py")
)).hexdigest()


def material_grid_spacing(frequency_hz, permittivity=83.16, conductivity=15.29):
    """Resolve the shortest generator-material wavelength at the carrier.

    The largest relative permittivity and conductivity give the shortest phase
    wavelength. The attenuation length is shortest at low permittivity, so
    use the high-conductivity, vacuum-permittivity limit for its bound.
    """
    omega = 2 * np.pi * frequency_hz
    k0 = omega / c
    loss = conductivity / (omega * eps0)
    beta = (k0 * np.sqrt(complex(permittivity, -loss))).real
    alpha = abs((k0 * np.sqrt(complex(1., -loss))).imag)
    wavelength_spacing = 2 * np.pi / (10 * beta)
    return min(wavelength_spacing, 1. / (2 * alpha)) if alpha else wavelength_spacing


@dataclass(frozen=True)
class AcquisitionConfig:
    grid_shape: tuple = (128, 216, 216)  # grid cells, including PML
    pml_thickness: int = 15
    frequency_hz: float = 2.4e9
    grid_spacing_m: float | None = None
    elements_per_side: int = 4
    pitch_cells: float | None = None
    samples_per_shot: int = 750
    sample_stride: int | None = None
    source_amplitude_sim: float = 0.1
    object_unit_m: float | None = None
    standoff_cells: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if not isinstance(self.pml_thickness, (int, np.integer)) or self.pml_thickness < 0:
            raise ValueError("pml_thickness must be a nonnegative integer")
        if self.standoff_cells is not None and (not isinstance(self.standoff_cells, (int, np.integer))
                                                or self.standoff_cells < 0):
            raise ValueError("standoff_cells must be a nonnegative integer")
        if not isinstance(self.elements_per_side, (int, np.integer)) or self.elements_per_side < 2:
            raise ValueError("A steerable square array needs at least two elements per side")
        if (len(self.grid_shape) != 3 or any(not isinstance(n, (int, np.integer)) for n in self.grid_shape)
                or min(self.grid_shape) <= 2 * self.pml_thickness):
            raise ValueError("grid_shape must give three integer cell counts with space for the PML")
        if not isinstance(self.samples_per_shot, (int, np.integer)) or self.samples_per_shot < 2:
            raise ValueError("samples_per_shot must be an integer of at least two")
        if self.sample_stride is not None and (not isinstance(self.sample_stride, (int, np.integer))
                                               or self.sample_stride < 1):
            raise ValueError("sample_stride must be a positive integer")
        if self.grid_spacing_m is not None and (not np.isfinite(self.grid_spacing_m) or self.grid_spacing_m <= 0):
            raise ValueError("grid_spacing_m must be finite and positive")
        if self.object_unit_m is not None and (not np.isfinite(self.object_unit_m) or self.object_unit_m <= 0):
            raise ValueError("object_unit_m must be finite and positive")
        if not np.isfinite(self.source_amplitude_sim) or self.source_amplitude_sim == 0:
            raise ValueError("Scene acquisition needs a nonzero, finite source amplitude")

    @property
    def spacing(self):
        return self.grid_spacing_m if self.grid_spacing_m is not None else material_grid_spacing(self.frequency_hz)

    @property
    def standoff(self):
        if self.standoff_cells is not None:
            return self.standoff_cells
        # The thesis uses three cells of stand-off on a lambda/20 mesh.
        return int(np.ceil(0.15 * c / (self.frequency_hz * self.spacing)))


def _grid_and_sensor(config):
    grid = grid_init(config.grid_shape, config.spacing, 1., 1., config.pml_thickness)
    sensor = _make_sensor(config, grid)
    grid[0, 0, 0] = sensor
    return grid, sensor


def _make_sensor(config, grid):
    wavelength = c / config.frequency_hz
    pitch = config.pitch_cells if config.pitch_cells is not None else wavelength / (2 * config.spacing)
    position, rotation = PhasedSensorDevice.get_position_and_rotation(grid, config.pml_thickness, config.standoff)
    sensor = PhasedSensorDevice(position, rotation, n_antennas=config.elements_per_side,
                                antenna_spacing=pitch, frequency=config.frequency_hz,
                                wavelength=wavelength, amplitude=config.source_amplitude_sim,
                                grid_spacing=config.spacing)
    lo, hi = sensor.get_bounding_box()
    if np.any(lo < config.pml_thickness) or np.any(hi >= np.array(grid.shape) - config.pml_thickness):
        raise ValueError("Choose grid_shape, pitch_cells and standoff_cells to fit the aperture in the interior")
    return sensor


def _sampling_stride(config, grid, sensor, angles):
    first = np.array([int(sensor.position[0]), config.pml_thickness, config.pml_thickness])
    last = np.array(grid.shape) - config.pml_thickness - 1
    corners = np.array(list(product(*zip(first, last))))
    tip_positions = sensor.transmit_positions_cells
    max_distance = np.linalg.norm(corners[:, None, :] - tip_positions, axis=-1).max() * config.spacing
    latest_start = 0.
    for angle in angles:
        sensor.recalculate_timings(angle)
        latest_start = max(latest_start, max(a.start_time for a in sensor.antennas))
    end_of_pulse = latest_start + 1 / config.frequency_hz
    round_trip = 2 * max_distance / c
    base_span = (config.samples_per_shot - 1) * grid.time_step
    stride = config.sample_stride or max(1, int(np.ceil((end_of_pulse + round_trip) / base_span)))
    if stride * base_span < end_of_pulse + round_trip:
        raise ValueError("The recording must cover the transmitted pulse and the round trip to the furthest cell")
    if stride * grid.time_step >= 0.5 / config.frequency_hz:
        raise ValueError("Too few samples to resolve the RF carrier over this recording window")
    return stride


def _record_one(config, material_runs, angle, angles):
    # Grid.reset() in fdtd 0.2.7 leaves the PML history in place.
    grid, sensor = _grid_and_sensor(config)
    install_materials(grid, material_runs)
    stride = _sampling_stride(config, grid, sensor, angles)
    sensor.recalculate_timings(angle)
    for step in range(1 + (config.samples_per_shot - 1) * stride):
        grid.step()
        if step % stride == 0:
            sensor.calc_voltage_for_antennas(grid.E)

    metadata = {
        "acquisition_version": ACQUISITION_VERSION,
        "fdtd_version": fdtd.__version__,
        "fdtd_source_sha256": FDTD_SOURCE_SHA256,
        "sdf_target": SDF_TARGET,
        "sensor_layout": "antenna,direction,timestep",
        "beamforming_angles_deg": [list(pair) for pair in angles],
        "grid_shape": [int(n) for n in grid.shape],
        "grid_spacing_m": float(grid.grid_spacing),
        "pml_thickness": int(config.pml_thickness),
        "grid_origin_cells": [int(sensor.position[0]), int(config.pml_thickness), int(config.pml_thickness)],
        "frequency_hz": float(config.frequency_hz),
        "time_step_s": grid.time_step,
        "sample_stride": int(stride),
        "samples_per_shot": int(config.samples_per_shot),
        "antenna_positions_cells": [a.position.tolist() for a in sensor.antennas],
        "transmit_positions_cells": sensor.transmit_positions_cells.tolist(),
        "element_polarization_local": sensor.element_polarization.tolist(),
        "source_amplitude_sim": float(config.source_amplitude_sim),
        "voltage_units": "V",
    }
    recording = np.asarray([a.E_samples for a in sensor.antennas], dtype=np.float64)
    if not np.isfinite(recording).all():
        raise FloatingPointError("The solver produced non-finite receiver samples")
    return recording, metadata


def _record_shots(config, material_runs, angles):
    recordings = []
    for angle in angles:
        # FDTD attaches boundaries, sources and objects to the grid by circular
        # references. Release the previous shot before allocating the next one.
        gc.collect()
        recording, metadata = _record_one(config, material_runs, angle, angles)
        recordings.append(recording)
    return np.stack(recordings, axis=1), metadata


def record_shots(config, material_runs=(), angles=BEAMS):
    """Record independent shots with axes (antenna, direction, timestep)."""
    angles = tuple(angles)
    if not angles:
        raise ValueError("At least one steering angle is required")
    recordings, _ = _record_shots(config, tuple(material_runs), angles)
    return recordings


def _layout(config, objects, n_objects, n_jobs, *, geometry=None, material_dtype=None):
    gc.collect()
    grid, sensor = geometry if geometry is not None else _grid_and_sensor(config)
    if objects is None:
        objects = generate_objects(grid, n_objects, config.pml_thickness, sensor, size_unit_m=config.object_unit_m)
    if not objects:
        raise ValueError("A supervised scene requires at least one object with a finite signed distance")
    lower = np.array([int(sensor.position[0]), config.pml_thickness, config.pml_thickness])
    upper = np.array(grid.shape) - config.pml_thickness - 1
    for obj in objects:
        box = obj.sdf.get_bounding_box()
        if np.any(box[0] < lower) or np.any(box[1] > upper) or bounding_box_overlap(box, sensor.get_bounding_box()):
            raise ValueError("Scene objects must fit in the saved volume and clear the sensor")
    labels, runs = voxelize_world(objects, grid, config.pml_thickness, n_jobs, material_dtype=material_dtype)
    if not runs:
        raise ValueError("Scene contains no occupied cells at this grid spacing")
    return labels, runs


def _batch_scene(config, objects, n_objects, n_jobs, reference, device, dtype, max_parallel_beams):
    from BatchFDTD import _grid_and_sensor as batch_geometry, batch_metadata, record_batch

    if dtype not in ("float32", "float64"):
        raise ValueError("Batch field dtype must be float32 or float64")
    geometry = batch_geometry(config)
    labels, runs = _layout(config, objects, n_objects, n_jobs, geometry=geometry, material_dtype=dtype)
    # The native solver reads dense coefficients directly from the voxel labels.
    del runs
    metadata = batch_metadata(config)
    source_dir = Path(__file__).resolve().parent
    metadata.update(engine="BatchFDTD", dtype=dtype, solver_sha256=sha256(b"".join(
        (source_dir / name).read_bytes() for name in ("BatchFDTD.cpp", "BatchFDTD.py")
    )).hexdigest())
    options = dict(device=device, dtype=dtype, max_parallel_beams=max_parallel_beams)
    eps = np.ones((1, *config.grid_shape), dtype=dtype)
    sigma = np.zeros_like(eps)
    if reference is None:
        reference = record_batch(config, eps, sigma, **options)[0], metadata
    pml = config.pml_thickness
    interior = (slice(pml, -pml or None),) * 3
    eps[(0, *interior)] = labels[interior + (0,)]
    sigma[(0, *interior)] = labels[interior + (1,)]
    sensor_samples = record_batch(config, eps, sigma, **options)[0]
    return labels, sensor_samples, metadata, reference


def _write_scene(destination, config, objects, n_objects, n_jobs, reference=None, *,
                 batch_device=None, dtype=None, max_parallel_beams=None):
    if destination.exists():
        raise FileExistsError(destination)
    if batch_device is None:
        if dtype is not None or max_parallel_beams is not None:
            raise ValueError("dtype and max_parallel_beams require a batch backend")
        labels, runs = _layout(config, objects, n_objects, n_jobs)
        if reference is None:
            reference = _record_shots(config, (), BEAMS)
        sensor_samples, metadata = _record_shots(config, runs, BEAMS)
    else:
        labels, sensor_samples, metadata, reference = _batch_scene(
            config, objects, n_objects, n_jobs, reference, batch_device,
            "float32" if dtype is None else dtype, max_parallel_beams)
    empty_samples, empty_metadata = reference
    if empty_metadata != metadata:
        raise ValueError("Scene and empty-reference acquisition settings differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".acquiring-", dir=destination.parent) as scratch:
        scene = Path(scratch) / "scene"
        scene.mkdir()
        save_grid(scene, labels, config.pml_thickness, metadata["grid_origin_cells"][0])
        np.save(scene / "sensor_samples.npy", sensor_samples)
        np.save(scene / "empty_sensor_samples.npy", empty_samples)
        (scene / "acquisition.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        scene.rename(destination)
    return reference


def write_scene(destination, config, objects=None, *, n_objects=1, n_jobs=20,
                batch_device=None, dtype=None, max_parallel_beams=None):
    """Simulate one scene and its matched empty reference in a new directory."""
    _, metadata = _write_scene(Path(destination), config, objects, n_objects, n_jobs,
                               batch_device=batch_device, dtype=dtype, max_parallel_beams=max_parallel_beams)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=20, help="CPU workers for voxelization")
    parser.add_argument("--backend", choices=("numpy", "torch", "torch.cuda", "batch.cpu", "batch.cuda"),
                        default="torch")
    parser.add_argument("--dtype", choices=("float32", "float64"), help="Batch solver precision (default: float32)")
    parser.add_argument("--max-parallel-beams", type=int,
                        help="Number of simultaneous beams in the batch solver (default: all five)")
    parser.add_argument("--shape", type=int, nargs=3, default=AcquisitionConfig.grid_shape, metavar=("X", "Y", "Z"))
    parser.add_argument("--pml", type=int, default=AcquisitionConfig.pml_thickness)
    parser.add_argument("--grid-spacing-m", type=float, help="Override the material-resolved cell size in metres")
    parser.add_argument("--object-unit-m", type=float,
                        help="Physical length of one random shape-size unit (default: one grid cell)")
    parser.add_argument("--pitch-cells", type=float, help="Override the half-wavelength element pitch, in grid cells")
    parser.add_argument("--standoff-cells", type=int, help="Override the 0.15-wavelength sensor-to-PML distance")
    parser.add_argument("--samples", type=int, default=AcquisitionConfig.samples_per_shot)
    parser.add_argument("--sample-stride", type=int, help="Solver steps per sample, e.g. for longer recordings")
    parser.add_argument("--depth-output", type=Path, help="Also write pinhole depth frames for these scenes")
    args = parser.parse_args()
    if args.scenes < 1 or args.objects < 1 or args.start_index < 0:
        parser.error("scenes and objects must be positive and start-index must be nonnegative")
    batch_device = args.backend.removeprefix("batch.") if args.backend.startswith("batch.") else None
    if batch_device is None and (args.dtype is not None or args.max_parallel_beams is not None):
        parser.error("--dtype and --max-parallel-beams require --backend batch.cpu or batch.cuda")
    if args.max_parallel_beams is not None and args.max_parallel_beams < 1:
        parser.error("--max-parallel-beams must be positive")
    config = AcquisitionConfig(grid_shape=tuple(args.shape), pml_thickness=args.pml, grid_spacing_m=args.grid_spacing_m,
                               pitch_cells=args.pitch_cells, samples_per_shot=args.samples,
                               standoff_cells=args.standoff_cells, sample_stride=args.sample_stride,
                               object_unit_m=args.object_unit_m)
    paths = [args.dataset / f"scene_{i:06d}" for i in range(args.start_index, args.start_index + args.scenes)]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    fdtd.set_backend("numpy" if batch_device is not None else args.backend)
    reference = None
    for i, path in enumerate(paths, start=args.start_index):
        np.random.seed(args.seed + i)
        reference = _write_scene(path, config, None, args.objects, args.jobs, reference, batch_device=batch_device,
                                 dtype=args.dtype, max_parallel_beams=args.max_parallel_beams)
        print(path, flush=True)
    if args.depth_output is not None:
        from DepthCamera import generate_frames
        generate_frames(paths, args.depth_output)


if __name__ == "__main__":
    main()
