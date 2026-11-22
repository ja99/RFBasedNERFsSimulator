import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fdtd
import numpy as np
import torch
from fdtd.constants import c
from fdtd.conversions import simE_to_worldE

import helpers


def sin_pulse(t: float, width: float) -> float:
    """Sine source at time t, with period width, both in seconds."""
    return np.sin(2.0 * np.pi * t / width)


def _grid_indices(points):
    # Ties must be rounded the same way on both sides of the array. Banker's
    # rounding can collapse adjacent, half-integer element positions.
    return np.floor(np.asarray(points) + 0.5).astype(np.intp)


@dataclass
class Antenna:
    position: np.ndarray  # base, in grid cells
    E_samples: List[float]  # received line integrals, in volts
    index_slice: Tuple[slice, slice, slice]
    antenna_length: float  # cells
    transmitter: bool
    start_time: Optional[float]  # seconds
    width_pulse: Optional[float]  # seconds per cycle
    amplitude: Optional[float]  # electric field increment in fdtd simulation units


class SensorDevice:
    def __init__(self):
        self.antennas = []
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.element_polarization = np.array([0., 1., 0.])  # array-local coordinates
        self.grid_spacing = 0.01
        self.n_pulses = 1

    def save_samples(self, path: pathlib.Path):
        """Write one recording with axes (antenna, time)."""
        np.save(path / "sensor_samples.npy", np.asarray([a.E_samples for a in self.antennas]))

    def get_bounding_box(self) -> np.ndarray:
        points = np.vstack(([a.position for a in self.antennas], self.transmit_positions_cells))
        return np.stack([points.min(axis=0), points.max(axis=0)])

    @staticmethod
    def get_position_and_rotation(grid: fdtd.Grid, pml_thickness: int, pml_distance: int = 3):
        _, ny, nz = grid.shape
        return np.array([pml_thickness + pml_distance, ny // 2, nz // 2]), np.zeros(3)

    @staticmethod
    def get_slice_indices(v1: np.ndarray, v2: np.ndarray) -> Tuple[slice, slice, slice]:
        low = np.minimum(v1, v2).astype(int)
        high = np.maximum(v1, v2).astype(int)
        return tuple(slice(lo, hi + 1) for lo, hi in zip(low, high))

    def create_antenna_slice(self):
        """Cache the on-grid segments and trapezoidal line-integral weights."""
        self._direction = helpers.rotate_vec(self.element_polarization, *self.rotation)
        bases = np.array([a.position for a in self.antennas])
        lengths = np.array([a.antenna_length for a in self.antennas])
        segments = max(1, int(np.ceil(lengths.max())))
        path = bases[:, None, :] + (lengths[:, None, None] * self._direction
                                    * np.linspace(0., 1., segments + 1)[None, :, None])
        indices = _grid_indices(path)
        self.transmit_positions_cells = indices[:, -1]
        self._path_indices = tuple(indices[:, :, axis] for axis in range(3))
        weights = np.ones(segments + 1) / segments
        weights[[0, -1]] *= 0.5
        self._voltage_weights = simE_to_worldE(lengths[:, None] * self.grid_spacing * weights)
        self._torch_geometry = None
        for antenna, base, tip in zip(self.antennas, indices[:, 0], self.transmit_positions_cells):
            antenna.index_slice = self.get_slice_indices(base, tip)

    def calc_voltage_for_antennas(self, E):
        """Append the base-to-tip integral of the projected SI electric field."""
        if torch.is_tensor(E):
            key = (E.device, E.dtype)
            if self._torch_geometry is None or self._torch_geometry[0] != key:
                indices = tuple(torch.as_tensor(i, device=E.device) for i in self._path_indices)
                direction = E.new_tensor(self._direction)
                weights = E.new_tensor(self._voltage_weights)
                self._torch_geometry = key, indices, direction, weights
            _, indices, direction, weights = self._torch_geometry
            voltages = ((E[indices] @ direction) * weights).sum(dim=1).detach().cpu().tolist()
        else:
            projected = E[self._path_indices] @ self._direction
            voltages = (projected * self._voltage_weights).sum(axis=1).tolist()
        for antenna, voltage in zip(self.antennas, voltages):
            antenna.E_samples.append(voltage)

    def _register_grid(self, grid: fdtd.Grid, x, y, z):
        if not np.isclose(grid.grid_spacing, self.grid_spacing, rtol=1e-9, atol=0.):
            raise ValueError("Sensor grid_spacing must match the field solver grid_spacing (metres)")
        lo, hi = self.get_bounding_box()
        if np.any(lo < 0) or np.any(hi >= grid.shape):
            raise ValueError("Antenna base or tip lies outside the field grid")
        self.grid = grid
        grid.sources.append(self)

    def update_E(self):
        self.add_transmissions_in_grid(self.grid, self.grid.time_steps_passed)

    def update_H(self):
        pass

    def add_transmissions_in_grid(self, grid: fdtd.Grid, t: int):
        current_t = t * grid.time_step
        indices = []
        amplitudes = []
        for i, antenna in enumerate(self.antennas):
            if antenna.transmitter:
                pulse_t = current_t - antenna.start_time
                if 0. <= pulse_t < self.n_pulses * antenna.width_pulse:
                    indices.append(i)
                    amplitudes.append(antenna.amplitude * sin_pulse(pulse_t, antenna.width_pulse))
        if indices:
            tip_indices = tuple(self.transmit_positions_cells[indices].T)
            fields = np.asarray(amplitudes)[:, None] * self._direction
            if torch.is_tensor(grid.E):
                fields = grid.E.new_tensor(fields)
            grid.E[tip_indices] += fields


class PhasedSensorDevice(SensorDevice):
    def __init__(self, position: np.ndarray, rotation: np.ndarray, n_antennas: int = 4,
                 antenna_spacing: float = 1., frequency: float = 2.4e9,
                 wavelength: Optional[float] = None, amplitude: float = 1.,
                 beamforming_angle: Tuple[float, float] = (0., 0.), grid_spacing: float = 0.01,
                 n_pulses: int = 1, element_polarization=(0., 1., 0.)):
        """Square y–z aperture with local +x broadside.

        Positions and pitch are in grid cells, wavelength and grid_spacing in
        metres. Element polarization is specified in the local array frame.
        `(1, 0, 0)` reproduces the original longitudinal element orientation.
        """
        super().__init__()
        if int(n_antennas) != n_antennas or n_antennas < 1:
            raise ValueError("n_antennas must be a positive integer per array side")
        if int(n_pulses) != n_pulses or n_pulses < 1:
            raise ValueError("n_pulses must be a positive integer")
        if not all(np.isfinite(x) and x > 0 for x in (frequency, grid_spacing, antenna_spacing)):
            raise ValueError("frequency, grid_spacing and antenna_spacing must be finite and positive")
        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite")
        self.frequency = float(frequency)
        self.wavelength = c / frequency if wavelength is None else float(wavelength)
        if not np.isfinite(self.wavelength) or self.wavelength <= 0.:
            raise ValueError("wavelength must be finite and positive")
        self.n_pulses = int(n_pulses)
        self.grid_spacing = float(grid_spacing)
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        polarization = np.asarray(element_polarization, dtype=float)
        if any(v.shape != (3,) or not np.isfinite(v).all()
               for v in (self.position, self.rotation, polarization)):
            raise ValueError("position, rotation and element_polarization must be finite 3-vectors")
        norm = np.linalg.norm(polarization)
        if not np.isfinite(norm) or norm == 0.:
            raise ValueError("element_polarization must have a finite nonzero norm")
        self.element_polarization = polarization / norm

        offsets = (np.arange(int(n_antennas)) - (n_antennas - 1) / 2) * antenna_spacing
        for y in offsets:
            for z in offsets:
                base = self.position + helpers.rotate_vec(np.array([0., y, z]), *self.rotation)
                self.antennas.append(Antenna(position=_grid_indices(base), E_samples=[],
                                             index_slice=None, antenna_length=1., transmitter=True,
                                             start_time=0., width_pulse=1. / frequency,
                                             amplitude=float(amplitude)))
        self.create_antenna_slice()
        if len(np.unique(self.transmit_positions_cells, axis=0)) != len(self.antennas):
            raise ValueError("Multiple transmitters occupy the same cell. Increase antenna_spacing.")
        self.recalculate_timings(beamforming_angle)

    def recalculate_timings(self, beamforming_angle: Tuple[float, float] = (0., 0.)):
        """Steer towards local azimuth (+y) and elevation (+z), both in degrees."""
        angles = np.asarray(beamforming_angle, dtype=float)
        if angles.shape != (2,) or not np.isfinite(angles).all() or np.any(np.abs(angles) > 90):
            raise ValueError("Expected (azimuth, elevation) in degrees between -90 and 90")
        az, el = np.deg2rad(angles)
        unit = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        direction = helpers.rotate_vec(unit, *self.rotation)
        # Equal arrival times require downstream sources to fire later. Use the
        # actual, discretized transmit locations even when the array is rotated.
        offsets_m = (self.transmit_positions_cells - self.position) * self.grid_spacing
        delays = (offsets_m @ direction) / (self.wavelength * self.frequency)
        for antenna, delay in zip(self.antennas, delays - delays.min()):
            antenna.start_time = float(delay)
        self.beamforming_angle = tuple(angles)
