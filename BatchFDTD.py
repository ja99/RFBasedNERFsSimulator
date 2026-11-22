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

import fdtd
from fdtd.constants import eta0
import numpy as np
import torch

from Acquisition import ACQUISITION_VERSION, BEAMS, FDTD_SOURCE_SHA256, _grid_and_sensor, _sampling_stride
from WorldGenerator import SDF_TARGET


BATCH_FDTD_SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_suffix(".cpp").read_bytes()).hexdigest()


class _Layout(ctypes.Structure):
    _fields_ = [(key, ctypes.c_int) for key in (
        "nx", "ny", "nz", "scenes", "beams", "antennas", "path_length", "samples", "stride"
    )] + [("courant", ctypes.c_double), ("direction", ctypes.c_double * 3)]


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
    emitters: np.ndarray
    paths: np.ndarray


@lru_cache(maxsize=8)
def _make_plan(config, angles):
    grid, sensor = _grid_and_sensor(config)
    stride = _sampling_stride(config, grid, sensor, angles)
    shape = tuple(grid.shape)
    profiles = np.zeros((2, 3, 2, max(shape)), dtype=np.float64)
    pml = config.pml_thickness
    if pml:
        for axis, letter in enumerate("xyz"):
            for side in ("low", "high"):
                boundary = getattr(grid, f"_{letter}{side}_boundary")
                select = [0, 0, 0, axis]
                select[axis] = slice(None)
                lo = 0 if side == "low" else shape[axis] - pml
                for phase, name in enumerate(("E", "H")):
                    for index, coefficient in enumerate(("b", "c")):
                        values = getattr(boundary, coefficient + name)[tuple(select)]
                        profiles[phase, axis, index, lo:lo + pml] = fdtd.backend.numpy(values)
                    # fdtd's convolution uses differences strictly within each
                    # slab, including the first cell of its upper-side E layer.
                    skip = lo if phase == 0 else lo + pml - 1
                    profiles[phase, axis, 1, skip] = 0
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
    emitters = np.full(shape, -1, dtype=np.int32)
    emitters[tuple(sensor.transmit_positions_cells.T)] = np.arange(len(sensor.antennas))
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
                 sensor._voltage_weights.copy(), emitters, paths)
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
def record_batch(config, permittivity, conductivity, *, device="cuda", dtype="float32", angles=BEAMS):
    """Return SI receiver voltages [scene, antenna, beam, time] for full material arrays.

    Inputs have shape [scene, X, Y, Z] including vacuum PML cells. Permittivity
    is relative and conductivity is in S/m. All scenes share the acquisition
    configuration. Each scene and beam has independent fields and PML history.
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
    shape = tuple(config.grid_shape)
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
    beams, antennas = plan.source.shape[1:]
    scenes = eps.shape[0]
    del eps, sig, inverse, loss
    shots = scenes * beams
    volume = int(np.prod(shape))
    if shots > 65535 or 3 * volume >= 2**31:
        raise ValueError("Split the scene batch or reduce the spatial dimensions for this solver")
    settings = {"dtype": dtype, "device": device}
    fields = torch.zeros((2, shots, 3, volume), **settings)
    histories = torch.zeros((2, shots, 6, volume), **settings)
    profiles, source, weights = (torch.as_tensor(a, **settings).contiguous()
                                 for a in (plan.profiles, plan.source, plan.weights))
    emitters, paths = (torch.as_tensor(a, device=device).contiguous() for a in (plan.emitters, plan.paths))
    output = torch.empty((scenes, beams, antennas, config.samples_per_shot), **settings)
    layout = _Layout(*shape, scenes, beams, antennas, plan.paths.shape[1], config.samples_per_shot,
                     plan.metadata["sample_stride"], plan.courant, (ctypes.c_double * 3)(*plan.direction))
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
