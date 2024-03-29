import time
from dataclasses import dataclass
from math import ceil
from typing import List, Tuple, Optional

import torch
from icecream import ic
from torch import multiprocessing as mp
import fdtd
from tqdm import tqdm

import helpers
import numpy as np
import pathlib

import inner_functions


def sin_pulse(
        t: float,  # current time in seconds
        width: float  # the width of the pulse, in seconds
) -> float:
    x = t / width * 2.0 * np.pi
    return np.sin(x)



@dataclass
class Antenna:
    position: np.ndarray
    E_samples: List[float]
    index_slice: Tuple[slice, slice, slice]
    antenna_length: float  # cells
    transmitter: bool
    start_time: Optional[float]  # in seconds
    width_pulse: Optional[float]  # in seconds
    amplitude: Optional[float]  # in volts


class SensorDevice:
    antennas: List[Antenna]
    position: np.ndarray
    rotation: np.ndarray
    grid: fdtd.Grid


    def __init__(self):
        self.source_slices = []
        self.antennas = []
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])

    def save_samples(self, path: pathlib.Path):
        ar = np.zeros((len(self.antennas), len(self.antennas[0].E_samples)))
        for a, antenna in enumerate(self.antennas):
            ar[a, :] = antenna.E_samples

        np.save(path / "sensor_samples.npy", ar)


    def get_bounding_box(self) -> np.ndarray:
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])

        for antenna in self.antennas:
            min_coords = np.minimum(min_coords, antenna.position)
            max_coords = np.maximum(max_coords, antenna.position)

        return np.stack([min_coords, max_coords])

    @staticmethod
    def get_position_and_rotation(grid: fdtd.Grid, pml_thickness: int, pml_distance: int = 3) -> (
    np.ndarray, np.ndarray):
        x, y, z = grid.shape

        # Middle indices along each axis
        my, mz = y // 2, z // 2

        position = np.array([pml_thickness + pml_distance, my, mz])
        rotation = np.array([0.0, 0.0, 0.0])

        return position, rotation

    @staticmethod
    def get_slice_indices(v1: np.ndarray, v2: np.ndarray) -> Tuple[slice, slice, slice]:
        # Assuming v1 and v2 both have shape [3]

        # Stack the arrays and get the minimum and maximum values for x, y, z
        min_vals = np.min(np.stack([v1, v2]), axis=0)
        max_vals = np.max(np.stack([v1, v2]), axis=0)

        # Return slices
        return (slice(int(min_vals[0]), int(max_vals[0]) + 1),
                slice(int(min_vals[1]), int(max_vals[1]) + 1),
                slice(int(min_vals[2]), int(max_vals[2]) + 1))

    # ToDo: make it return the slice and and length and add them in the init
    def create_antenna_slice(self):
        for antenna in self.antennas:
            antenna_base = antenna.position
            antenna_tip_relative = helpers.rotate_vec(np.array([antenna.antenna_length, 0., 0.]),
                                                      *self.rotation)
            antenna_tip = antenna_base + antenna_tip_relative
            antenna_tip = np.round(antenna_tip)

            antenna.index_slice = self.get_slice_indices(antenna_base, antenna_tip)

    def calc_voltage_for_antennas(self, E: np.ndarray):
        #ToDo add direction vector to antenna
        antenna_dir_vec = helpers.rotate_vec(np.array([1., 0., 0.]), *self.rotation)

        for antenna in self.antennas:
            # Get the E field vectors for the antenna slice in [n, 3] shape
            e_vecs = E[antenna.index_slice].squeeze()
            # Sum the E field vectors along the slice
            e_vec = np.sum(e_vecs, axis=0)
            # Project the E field vector onto the antenna direction vector
            e_projected = np.dot(e_vec, antenna_dir_vec)
            # Calculate the voltage
            voltage = e_projected * antenna.antenna_length

            antenna.E_samples.append(voltage.item())



    # Overwrite for this to work with fdtd grid
    def _register_grid(self, grid: fdtd.Grid, x: int, y: int, z: int):
        self.grid = grid
        self.grid.sources.append(self)


    # Overwrite for this to work with fdtd grid
    def update_E(self):
        self.add_transmissions_in_grid(self.grid, self.grid.time_steps_passed)

    # Overwrite for this to work with fdtd grid
    def update_H(self):
        pass


    def add_transmissions_in_grid(self,
                                  grid: fdtd.Grid,
                                  t: int,  # current timestep
                                  ):
        t = float(t)
        dt = grid.time_step  # in seconds
        current_t = t * dt  # in seconds

        dir_vec = helpers.rotate_vec(np.array([1., 0., 0.]), *self.rotation)
        for antenna in self.antennas:
            if antenna.transmitter:
                # determine if the current timestep is within the pulse
                if antenna.start_time <= current_t <= antenna.start_time + antenna.width_pulse:
                    pulse_t = current_t - antenna.start_time
                    amplitude = antenna.amplitude * sin_pulse(pulse_t, antenna.width_pulse)
                    # ToDo: get rid of antenna slice and add tip index to antenna
                    tip = antenna.position + dir_vec * antenna.antenna_length
                    vec_tensor = torch.tensor(dir_vec * amplitude).to(grid.E.device)
                    grid.E[round(tip[0].item()), round(tip[1].item()), round(tip[2].item())] += vec_tensor


class PhasedSensorDevice(SensorDevice):
    def __init__(self, position: np.array,
                 rotation: np.ndarray,
                 n_antennas: int = 4.0,
                 antenna_spacing: int = 1.0,
                 frequency: float = 2.4e9,
                 wavelength: float = 1.0,
                 amplitude: float = 1.0,
                 beamforming_angle: Tuple[float, float] = (0.0, 0.0),  # (azimuth, elevation) [-90°:+90°]
                 grid_spacing: float = 0.01,
                 n_pulses: float = 1.0
                 ):

        super().__init__()
        sensor_size = (n_antennas - 1) * antenna_spacing

        phi, theta = beamforming_angle
        phi, theta = np.deg2rad(phi), np.deg2rad(theta)

        time_delays = []

        for y in np.linspace(-0.5 * sensor_size, 0.5 * sensor_size, n_antennas):
            for z in np.linspace(-0.5 * sensor_size, 0.5 * sensor_size, n_antennas):
                relative_pos = np.array([0.0, y, z])

                # safe for phase shift calculation later
                xi, yi, zi = (relative_pos * grid_spacing)

                relative_pos = helpers.rotate_vec(relative_pos, *rotation)
                undiscretized_pos = position + relative_pos
                discretized_pos = np.round(undiscretized_pos)

                # Calculate the phase shift for each antenna
                phase_shift = (2 * np.pi / wavelength) * (
                        yi * np.sin(theta) * np.cos(phi) + zi * np.sin(theta) * np.sin(phi))

                # Convert the phase shift to a time delay
                time_delay = phase_shift / (2 * np.pi * frequency)
                time_delays.append(time_delay)

                antenna = Antenna(
                    position=discretized_pos,
                    E_samples=[],
                    index_slice=None,
                    transmitter=True,
                    antenna_length=1.0,
                    start_time=None,
                    width_pulse=1.0 / frequency,
                    amplitude=float(amplitude)
                )

                self.antennas.append(antenna)

        # recalculate the time delays so that the first antenna has a time delay of zero
        time_delays = np.array(time_delays)
        time_delays += np.abs(np.min(time_delays))

        print(f"{time_delays=}")

        for antenna, time_delay in zip(self.antennas, time_delays):
            antenna.start_time = time_delay

        self.wavelength = wavelength
        self.frequency = frequency

        self.position = np.round(position)
        self.rotation = rotation

        #ToDo: make this a stateless function and add it to the init
        self.create_antenna_slice()

    def recalculate_timings(self,
                            beamforming_angle: Tuple[float, float] = (0.0, 0.0),  # (azimuth, elevation) [-90°:+90°]
                            ):
        phi, theta = beamforming_angle
        phi, theta = np.deg2rad(phi), np.deg2rad(theta)

        time_delays = []

        for antenna in self.antennas:
            relative_pos = antenna.position - self.position
            xi, yi, zi = relative_pos
            phase_shift = (2 * np.pi / self.wavelength) * (
                    yi * np.sin(theta) * np.cos(phi) + zi * np.sin(theta) * np.sin(phi))
            time_delay = phase_shift / (2 * np.pi * self.frequency)
            time_delays.append(time_delay)

        # recalculate the time delays so that the first antenna has a time delay of zero
        time_delays = np.array(time_delays)
        time_delays += np.abs(np.min(time_delays))

        print(f"{time_delays=}")

        for antenna, time_delay in zip(self.antennas, time_delays):
            antenna.start_time = time_delay

