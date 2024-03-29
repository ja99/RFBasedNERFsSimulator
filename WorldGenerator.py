import random
import time
from typing import Tuple, List

import fdtd
import numpy as np
from enum import Enum

import torch
from tqdm import tqdm
from torch import multiprocessing as mp
import SDF
from Sensor import SensorDevice
from WorldObjectType import WorldObject
from inner_functions import add_objects_to_grid_inner
from icecream import ic


class ObjectType(Enum):
    SPHERE = 0
    BOX = 1


def grid_init(
        GRID_SIZE: Tuple[float, float, float],
        grid_spacing: float,
        permittivity: float,
        permeability: float,
        pml_thickness: int,
        pml_stability_factor: float = 1e-8,
) -> fdtd.Grid:
    grid = fdtd.Grid(
        GRID_SIZE,
        grid_spacing=grid_spacing,
        permittivity=permittivity,
        permeability=permeability,
    )

    grid[:, :, -pml_thickness:] = fdtd.PML(name="pml_z0", a=pml_stability_factor)
    grid[:, :, :pml_thickness] = fdtd.PML(name="pml_z1", a=pml_stability_factor)
    grid[:, -pml_thickness:, :] = fdtd.PML(name="pml_y0", a=pml_stability_factor)
    grid[:, :pml_thickness, :] = fdtd.PML(name="pml_y1", a=pml_stability_factor)
    grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_x0", a=pml_stability_factor)
    grid[:pml_thickness, :, :] = fdtd.PML(name="pml_x1", a=pml_stability_factor)

    return grid


def get_grid_center(grid: fdtd.Grid) -> np.ndarray:
    return np.array([
        len(grid.E) * 0.5,
        len(grid.E[0]) * 0.5,
        len(grid.E[0][0]) * 0.5,
    ])


def get_random_position(grid: fdtd.Grid, pml_thickness: int) -> np.ndarray:
    return np.array([
        float(np.random.randint(pml_thickness, len(grid.E) - pml_thickness)),
        float(np.random.randint(pml_thickness, len(grid.E[0]) - pml_thickness)),
        float(np.random.randint(pml_thickness, len(grid.E[0][0]) - pml_thickness)),
    ])


def get_random_rotation() -> np.ndarray:
    return np.array([
        np.random.randint(0, 360),
        np.random.randint(0, 360),
        np.random.randint(0, 360),
    ])


def get_random_box_size(range: Tuple[float, float]) -> np.ndarray:
    return np.array([
        float(np.random.randint(*range)),
        float(np.random.randint(*range)),
        float(np.random.randint(*range)),
    ])


def get_random_scalar(range: Tuple[float, float]) -> float:
    return float(np.random.randint(*range))


def bounding_box_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    # Check if box1's max is less than box2's min or box1's min is greater than box2's max along any axis
    for i in range(3):
        if box1[1, i] < box2[0, i] or box1[0, i] > box2[1, i]:
            return False

    # If none of the above conditions were met, the boxes overlap
    return True


def fill_world(grid: fdtd.Grid, n_obj: int, pml_thickness: int, sensor: SensorDevice) -> (fdtd.Grid, np.ndarray):
    sensor_bounding_box = sensor.get_bounding_box()

    objs = []

    for i in range(n_obj):
        # Values for human tissue +- random value
        permittivity = np.array([54.16 + get_random_scalar((-30, 30))])
        conductivity = np.array([11.29 + get_random_scalar((-5, 5))])

        while True:
            obj_type = ObjectType(np.random.randint(0, 2))
            if obj_type == ObjectType.SPHERE:
                obj = WorldObject(
                    SDF.Sphere(
                        position=get_random_position(grid, pml_thickness),
                        radius=get_random_scalar((5, 25)),
                    ),
                    permittivity=permittivity,
                    conductivity=conductivity,
                )

                if not bounding_box_overlap(obj.sdf.get_bounding_box(), sensor_bounding_box):
                    objs.append(obj)
                    break

            elif obj_type == ObjectType.BOX:
                obj = WorldObject(
                    SDF.Box(
                        position=get_random_position(grid, pml_thickness),
                        size=get_random_box_size((5, 50)),
                        rotation=get_random_rotation(),
                    ),
                    permittivity=permittivity,
                    conductivity=conductivity,
                )

                if not bounding_box_overlap(obj.sdf.get_bounding_box(), sensor_bounding_box):
                    objs.append(obj)
                    break

    return add_objects_to_grid(objs, grid, pml_thickness)


def add_objects_to_grid(objects: List[WorldObject], grid: fdtd.Grid, pml_thickness: int, n_jobs: int = 20) -> (
        fdtd.Grid, np.ndarray):
    # ToDo: add objects in 2d slices for performance
    export_grid = np.zeros((*grid.E.shape[:3], 3))

    x_min = pml_thickness
    x_max = grid.E.shape[0] - pml_thickness
    y_min = pml_thickness
    y_max = grid.E.shape[1] - pml_thickness
    z_min = pml_thickness
    z_max = grid.E.shape[2] - pml_thickness

    x_ranges: List[Tuple[int, int]] = []

    # subdivide the grid into n_jobs parts
    for i in range(n_jobs):
        x_ranges.append((x_min + (x_max - x_min) // n_jobs * i, x_min + (x_max - x_min) // n_jobs * (i + 1)))

    y_range = (y_min, y_max)
    z_range = (z_min, z_max)

    print(x_ranges)
    ctx = mp.get_context('spawn')
    objects_queue = ctx.Queue()
    subgrid_queue = ctx.Queue()
    kill_me_queue = ctx.Queue()
    subgrid_q_lock = ctx.Lock()
    kill_me_q_lock = ctx.Lock()
    processes = []

    for i in range(n_jobs):
        p = ctx.Process(target=add_objects_to_grid_inner,
                        args=(objects, objects_queue, subgrid_queue, kill_me_queue, subgrid_q_lock,
                              kill_me_q_lock, i, x_ranges[i], y_range, z_range))
        p.start()
        processes.append(p)

    while True:
        try:
            while objects_queue.empty() is False:
                x, y, z_start, z_end, name, permittivity, conductivity = objects_queue.get()
                grid[x, y, z_start:z_end] = fdtd.AbsorbingObject(
                    permittivity=torch.tensor(permittivity).to(grid.E.device),
                    conductivity=torch.tensor(conductivity).to(grid.E.device),
                    name=name)
            while subgrid_queue.empty() is False:
                x_range, y_range, subgrid = subgrid_queue.get()
                export_grid[x_range[0]:x_range[1], y_range[0]:y_range[1], z_min:z_max] = subgrid

            if kill_me_queue.qsize() == n_jobs and subgrid_queue.qsize() == 0 and objects_queue.qsize() == 0:
                break
        except Exception as e:
            print(e)

    ic()

    for p in processes:
        p.join()

    ic()

    return grid, export_grid
