"""Batched isotropic Yee solver for the acquisition geometry and its six PMLs.

record_batch returns [scene, antenna, beam, time] voltages. The fused CPU/CUDA
source builds locally into XDG_CACHE_HOME (or BATCH_FDTD_CACHE). CPU needs a
C++17/OpenMP compiler. CUDA needs nvcc alongside a CUDA-enabled PyTorch build.
"""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import fdtd
from fdtd.constants import eta0
import numpy as np
import torch

from Acquisition import ACQUISITION_VERSION, BEAMS, FDTD_SOURCE_SHA256, _make_sensor, _sampling_stride
from WorldGenerator import SDF_TARGET


BATCH_FDTD_SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_suffix(".cpp").read_bytes()).hexdigest()


class _Layout(ctypes.Structure):
    _fields_ = [(key, ctypes.c_int) for key in (
        "nx", "ny", "nz", "scenes", "beams", "antennas", "path_length", "samples", "stride",
        "pml", "emitter_x0", "emitter_x1"
    )] + [("slab_offsets", ctypes.c_size_t * 4), ("courant", ctypes.c_double), ("direction", ctypes.c_double * 3)]


class _Buffers(ctypes.Structure):
    _fields_ = [(key, ctypes.c_void_p) for key in (
        "electric", "magnetic", "electric_history", "magnetic_history", "decay", "drive",
        "profiles", "source", "weights", "emitters", "paths", "output"
    )]


@lru_cache(maxsize=8)
def _kernel(cuda_capability):
    source = Path(__file__).with_suffix(".cpp")
    if cuda_capability is None:
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        flags = ["-x", "c++", "-std=c++17", "-O3", "-ffp-contract=off", "-fopenmp", "-fPIC", "-shared"]
    else:
        compiler = shutil.which("nvcc")
        if compiler is None:
            from torch.utils.cpp_extension import CUDA_HOME
            candidate = Path(CUDA_HOME or "/usr/local/cuda") / "bin/nvcc"
            compiler = str(candidate) if candidate.is_file() else None
        flags = ["-x", "cu", "-std=c++17", "-O3", "--fmad=false", "-Xcompiler=-fPIC", "--shared"]
    if compiler is None:
        raise RuntimeError("BatchFDTD needs nvcc for CUDA or a C++17/OpenMP compiler for CPU")
    if cuda_capability is not None:
        available = subprocess.check_output([compiler, "--list-gpu-arch"], text=True).split()
        caps = [int(arch.removeprefix("compute_")) for arch in available
                if arch.removeprefix("compute_").isdigit()]
        compatible = [cap for cap in caps if cap <= cuda_capability]
        if not compatible:
            raise RuntimeError(f"{compiler} supports no architecture compatible with compute_{cuda_capability}")
        cap = max(compatible)
        target = f"sm_{cap}" if cap == cuda_capability else f"compute_{cap}"
        flags.append(f"--generate-code=arch=compute_{cap},code={target}")
    version = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256(source.read_bytes() + version + repr(flags).encode()).hexdigest()[:24]
    cache = Path(os.environ.get("BATCH_FDTD_CACHE") or
                 Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rf-batch-fdtd")
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"solver-{key}.so"
    if not library.exists():
        with TemporaryDirectory(dir=cache) as temp:
            output = Path(temp) / library.name
            result = subprocess.run([compiler, *flags, str(source), "-o", str(output)],
                                    text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"BatchFDTD compilation failed with {compiler}\n{result.stderr}")
            os.replace(output, library)
    dll = ctypes.CDLL(str(library))
    dll.rf_simulate.argtypes = [ctypes.POINTER(_Layout), ctypes.POINTER(_Buffers), ctypes.c_int, ctypes.c_void_p]
    dll.rf_simulate.restype = ctypes.c_int
    return dll


@dataclass
class _Plan:
    metadata: dict
    courant: float
    direction: np.ndarray
    profiles: np.ndarray
    source: np.ndarray
    weights: np.ndarray
    emitter_x0: int
    emitters: np.ndarray
    paths: np.ndarray


def _grid_and_sensor(config):
    # Time step and Courant number depend on dimensionality and spacing, not extent.
    probe = fdtd.Grid(tuple(min(int(n), 2) for n in config.grid_shape), grid_spacing=config.spacing)
    geometry = SimpleNamespace(shape=config.grid_shape, grid_spacing=probe.grid_spacing,
                               time_step=probe.time_step, courant_number=probe.courant_number)
    return geometry, _make_sensor(config, geometry)


def _pml_profiles(shape, pml, spacing, courant):
    profiles = np.zeros((2, 3, 2, max(shape)), dtype=np.float64)
    if pml == 0:
        return profiles
    # The pinned isotropic PML depends only on distance normal to a face. Obtain
    # its two 1D sequences without allocating a full reference solver or slabs.
    strip = fdtd.Grid((2 * pml + 1, 2, 2), grid_spacing=spacing, courant_number=courant)
    for high in (False, True):
        boundary = fdtd.PML(a=1e-8)  # same stability factor as WorldGenerator.grid_init
        strip[slice(-pml, None) if high else slice(pml), :, :] = boundary
        for phase, name in enumerate(("E", "H")):
            for index, coefficient in enumerate(("b", "c")):
                values = fdtd.backend.numpy(getattr(boundary, coefficient + name)[:, 0, 0, 0])
                for axis, extent in enumerate(shape):
                    start = extent - pml if high else 0
                    profiles[phase, axis, index, start:start + pml] = values
                    if index == 1:
                        # The reference convolution takes differences within each slab.
                        profiles[phase, axis, index, start if phase == 0 else start + pml - 1] = 0
    return profiles


@lru_cache(maxsize=8)
def _make_plan(config, angles):
    grid, sensor = _grid_and_sensor(config)
    stride = _sampling_stride(config, grid, sensor, angles)
    shape = tuple(grid.shape)
    pml = config.pml_thickness
    profiles = _pml_profiles(shape, pml, grid.grid_spacing, grid.courant_number)
    steps = 1 + (config.samples_per_shot - 1) * stride
    times = np.arange(steps, dtype=np.float64) * grid.time_step
    source = np.empty((steps, len(angles), len(sensor.antennas)), dtype=np.float64)
    for beam, angle in enumerate(angles):
        sensor.recalculate_timings(angle)
        for i, antenna in enumerate(sensor.antennas):
            t = times - antenna.start_time
            pulse = antenna.amplitude * np.sin(2 * np.pi * t / antenna.width_pulse)
            active = antenna.transmitter & (t >= 0) & (t < sensor.n_pulses * antenna.width_pulse)
            source[:, beam, i] = np.where(active, pulse, 0.0)
    tx, ty, tz = sensor.transmit_positions_cells.T
    emitter_x0, emitter_x1 = int(tx.min()), int(tx.max()) + 1
    emitters = np.full((emitter_x1 - emitter_x0, *shape[1:]), -1, dtype=np.int32)
    emitters[tx - emitter_x0, ty, tz] = np.arange(len(sensor.antennas))
    paths = np.ravel_multi_index(sensor._path_indices, shape).astype(np.int32)
    metadata = {
        "acquisition_version": ACQUISITION_VERSION, "fdtd_version": fdtd.__version__,
        "fdtd_source_sha256": FDTD_SOURCE_SHA256, "sdf_target": SDF_TARGET,
        "sensor_layout": "antenna,direction,timestep",
        "beamforming_angles_deg": [list(a) for a in angles], "grid_shape": list(shape),
        "grid_spacing_m": float(grid.grid_spacing), "pml_thickness": int(pml),
        "grid_origin_cells": [int(sensor.position[0]), int(pml), int(pml)],
        "frequency_hz": float(config.frequency_hz), "time_step_s": grid.time_step,
        "sample_stride": int(stride), "samples_per_shot": int(config.samples_per_shot),
        "antenna_positions_cells": [a.position.tolist() for a in sensor.antennas],
        "transmit_positions_cells": sensor.transmit_positions_cells.tolist(),
        "element_polarization_local": sensor.element_polarization.tolist(),
        "source_amplitude_sim": float(config.source_amplitude_sim), "voltage_units": "V",
    }
    plan = _Plan(metadata, grid.courant_number, sensor._direction.copy(), profiles, source,
                 sensor._voltage_weights.copy(), emitter_x0, emitters, paths)
    del grid, sensor
    gc.collect()
    return plan


def _plan(config, angles):
    angles = tuple(tuple(angle) for angle in angles)
    if not angles:
        raise ValueError("At least one steering angle is required")
    return _make_plan(replace(config, grid_shape=tuple(config.grid_shape)), angles)


def batch_metadata(config, angles=BEAMS):
    """Acquisition geometry/time metadata shared with Acquisition.record_shots."""
    return deepcopy(_plan(config, angles).metadata)


@torch.no_grad()
def record_batch(config, permittivity, conductivity, *, device="cuda", dtype="float32", angles=BEAMS,
                 max_parallel_beams=None):
    """Return SI receiver voltages [scene, antenna, beam, time] for full material arrays.

    Inputs have shape [scene, X, Y, Z] including vacuum PML cells. Permittivity
    is relative and conductivity is in S/m. All scenes share the acquisition
    configuration. max_parallel_beams limits the number of independent field
    and PML states resident per scene. All beams use the same acquisition clock.
    """
    if dtype in (torch.float32, "float32", np.float32, np.dtype("float32")):
        dtype = torch.float32
    elif dtype in (torch.float64, "float64", np.float64, np.dtype("float64")):
        dtype = torch.float64
    else:
        raise ValueError("Use float32 or float64 for the FDTD fields")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("BatchFDTD supports CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BatchFDTD needs a CUDA-enabled PyTorch build and GPU for device='cuda'")
    if max_parallel_beams is not None and (
        not isinstance(max_parallel_beams, (int, np.integer)) or isinstance(max_parallel_beams, bool)
        or max_parallel_beams < 1
    ):
        raise ValueError("max_parallel_beams must be a positive integer")
    shape = tuple(config.grid_shape)
    volume = int(np.prod(shape))
    if 3 * volume >= 2**31:
        raise ValueError("Reduce the spatial dimensions for this solver")
    eps = torch.as_tensor(permittivity, dtype=dtype, device=device)
    sig = torch.as_tensor(conductivity, dtype=dtype, device=device)
    if eps.ndim != 4 or tuple(eps.shape[1:]) != shape or sig.shape != eps.shape or eps.shape[0] < 1:
        raise ValueError("Material arrays must both have shape [scene, X, Y, Z] including the PML")
    invalid = (~torch.isfinite(eps) | ~torch.isfinite(sig) | (eps <= 0) | (sig < 0)).any()
    if bool(invalid):
        raise ValueError("Expected finite positive relative permittivity and finite nonnegative conductivity")
    pml = config.pml_thickness
    if pml:
        contrast = (eps != 1) | (sig != 0)
        bad = torch.zeros((), dtype=torch.bool, device=device)
        for axis, size in enumerate(shape, start=1):
            bad |= contrast.narrow(axis, 0, pml).any() | contrast.narrow(axis, size - pml, pml).any()
        if bool(bad):
            raise ValueError("PML cells must contain vacuum (relative permittivity 1, conductivity 0)")
        del contrast
    plan = _plan(config, angles)
    inverse = 1 / eps
    loss = 0.5 * plan.courant * inverse * sig * config.spacing * eta0
    decay = ((1 - loss) / (1 + loss)).contiguous()
    drive = (plan.courant * inverse / (1 + loss)).contiguous()
    scenes = eps.shape[0]
    del eps, sig, inverse, loss
    beams, antennas = plan.source.shape[1:]
    concurrent = min(max_parallel_beams or beams, beams)
    if scenes * concurrent > 65535:
        raise ValueError("Reduce the scene batch size or max_parallel_beams for this solver")
    if concurrent == beams:
        return _record_beams(plan, config, plan.source, decay, drive)
    traces = np.empty((scenes, antennas, beams, config.samples_per_shot),
                      dtype=np.float64 if dtype == torch.float64 else np.float32)
    for start in range(0, beams, concurrent):
        group = slice(start, start + concurrent)
        traces[:, :, group, :] = _record_beams(plan, config, plan.source[:, group], decay, drive)
    return traces


def _record_beams(plan, config, source, decay, drive):
    shape = tuple(config.grid_shape)
    scenes = decay.shape[0]
    beams, antennas = source.shape[1:]
    shots = scenes * beams
    volume = int(np.prod(shape))
    pml = config.pml_thickness
    device = decay.device
    settings = {"dtype": decay.dtype, "device": device}
    fields = torch.zeros((2, shots, 3, volume), **settings)
    # Each derivative axis needs history only on its two boundary slabs.
    # Offsets count cells before storing the two tangential components.
    nx, ny, nz = shape
    slab_offsets = [0]
    for face_area in (ny * nz, nx * nz, nx * ny):
        slab_offsets.append(slab_offsets[-1] + 2 * pml * face_area)
    histories = torch.zeros((2, shots, 2 * slab_offsets[-1]), **settings)
    profiles, source, weights = (torch.as_tensor(a, **settings).contiguous()
                                 for a in (plan.profiles, source, plan.weights))
    emitters, paths = (torch.as_tensor(a, device=device).contiguous() for a in (plan.emitters, plan.paths))
    output = torch.empty((scenes, beams, antennas, config.samples_per_shot), **settings)
    layout = _Layout(*shape, scenes, beams, antennas, plan.paths.shape[1], config.samples_per_shot,
                     plan.metadata["sample_stride"], pml, plan.emitter_x0, plan.emitter_x0 + plan.emitters.shape[0],
                     (ctypes.c_size_t * 4)(*slab_offsets),
                     plan.courant, (ctypes.c_double * 3)(*plan.direction))
    arrays = (fields[0], fields[1], histories[0], histories[1], decay, drive, profiles,
              source, weights, emitters, paths, output)
    buffers = _Buffers(*(a.data_ptr() for a in arrays))
    cuda_capability, stream = None, None
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        cuda_capability = major * 10 + minor
        stream = torch.cuda.current_stream(device).cuda_stream
    kernel = _kernel(cuda_capability)
    with torch.cuda.device(device) if cuda_capability else nullcontext():
        status = kernel.rf_simulate(ctypes.byref(layout), ctypes.byref(buffers), output.element_size(), stream)
        if status:
            raise RuntimeError(f"BatchFDTD CUDA kernel launch failed with error code {status}")
        traces = output.cpu().numpy().transpose(0, 2, 1, 3).copy()
    if not np.isfinite(traces).all():
        raise FloatingPointError("BatchFDTD produced non-finite receiver samples")
    return traces
