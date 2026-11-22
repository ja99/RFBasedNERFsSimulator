from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import fdtd
import numpy as np
from enum import Enum

import SDF
from Sensor import SensorDevice
from WorldObjectType import WorldObject
from inner_functions import MaterialRun, voxelize_slab


SDF_TARGET = "voxel_union_boundary_cells"


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

    if pml_thickness < 0 or min(grid.shape) <= 2 * pml_thickness:
        raise ValueError("PML thickness must leave a nonempty interior on each axis")
    if pml_thickness == 0:
        return grid

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
    tolerance = max(SDF.coordinate_roundoff(box1), SDF.coordinate_roundoff(box2))
    gap = np.maximum(box1[0], box2[0]) - np.minimum(box1[1], box2[1])
    return bool(np.all(gap <= tolerance))


def generate_objects(grid: fdtd.Grid, n_obj: int, pml_thickness: int,
                     sensor: SensorDevice, *, size_unit_m: Optional[float] = None) -> List[WorldObject]:
    """Generate whole objects inside the retained volume and clear of the sensor.

    Sampled radii and edge lengths use grid cells by default. A physical size
    unit in metres preserves the dimension distribution when refining the mesh.
    """
    if size_unit_m is not None and (not np.isfinite(size_unit_m) or size_unit_m <= 0):
        raise ValueError("The object size unit in metres must be finite and positive")
    size_scale = 1.0 if size_unit_m is None else size_unit_m / grid.grid_spacing
    sensor_bounds = sensor.get_bounding_box()
    lower = np.array([max(pml_thickness, int(sensor.position[0])), pml_thickness, pml_thickness])
    upper = np.array(grid.E.shape[:3]) - pml_thickness - 1
    objects = []
    for _ in range(n_obj):
        permittivity = 54.16 + get_random_scalar((-30, 30))
        conductivity = 11.29 + get_random_scalar((-5, 5))
        for _ in range(1000):
            if np.random.randint(2) == ObjectType.SPHERE.value:
                shape = SDF.Sphere(np.zeros(3), get_random_scalar((5, 25)) * size_scale)
            else:
                shape = SDF.Box(np.zeros(3), get_random_box_size((5, 50)) * size_scale, get_random_rotation())

            lo, hi = shape.get_bounding_box()
            min_center = np.ceil(lower - lo).astype(int)
            max_center = np.floor(upper - hi).astype(int)
            if np.any(min_center > max_center):
                continue
            shape.position = np.random.randint(min_center, max_center + 1).astype(float)
            bounds = shape.get_bounding_box()
            if not bounding_box_overlap(bounds, sensor_bounds):
                objects.append(WorldObject(shape, permittivity, conductivity))
                break
        else:
            raise ValueError("Could not fit an object inside the sensor crop and clear of the array")
    return objects


def fill_world(grid: fdtd.Grid, n_obj: int, pml_thickness: int, sensor: SensorDevice, *,
               size_unit_m: Optional[float] = None):
    objects = generate_objects(grid, n_obj, pml_thickness, sensor, size_unit_m=size_unit_m)
    return add_objects_to_grid(objects, grid, pml_thickness)


def voxelize_world(objects: List[WorldObject], grid: fdtd.Grid, pml_thickness: int, n_jobs: int = 20):
    """Return the full label array and reusable z-runs for the physical interior.

    The most negative primitive SDF assigns the material in overlapping shapes.
    The exported SDF measures the boundary of the resulting occupied voxel union
    in grid cells. Occupied centers are negative. Empty reference scenes use +inf.
    """
    shape = grid.E.shape[:3]
    if pml_thickness < 0 or min(shape) <= 2 * pml_thickness:
        raise ValueError("PML thickness must leave a nonempty interior on each axis")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")

    x_range, y_range, z_range = [(pml_thickness, n - pml_thickness) for n in shape]
    xs = range(*x_range)
    ys, zs = slice(*y_range), slice(*z_range)
    interior = (slice(*x_range), ys, zs)
    export_grid = np.zeros((*shape, 3), dtype=np.float64)
    # The generator's scalar label represents the isotropic background set on the solver.
    inverse = getattr(grid, "inverse_permittivity", None)
    if inverse is None:
        inverse_background = np.broadcast_to(1.0, export_grid[interior + (0,)].shape)
    else:
        inverse_background = fdtd.backend.numpy(inverse[interior + (0,)])
        if np.any(inverse_background == 0):
            raise ValueError("Voxelization requires a grid before material objects are registered")
    np.divide(1.0, inverse_background, out=export_grid[interior + (0,)])
    # fdtd casts background material parameters to the requested backend
    # precision even when its allocated field or coefficient arrays are float64.
    material_dtype = fdtd.backend.numpy(fdtd.backend.array(0.0, dtype=fdtd.backend.float)).dtype

    def sample_plane(x):
        return voxelize_slab(objects, (x, x + 1), y_range, z_range, export_grid[x, ys, zs, 0],
                             inverse_background[x - x_range[0]], material_dtype)

    material_runs = []
    occupied = np.zeros(shape, dtype=bool)
    present = np.zeros(len(objects), dtype=bool)
    # One work item per plane covers both short interiors and non-divisible worker counts.
    with ThreadPoolExecutor(max_workers=min(len(xs), n_jobs)) as pool:
        for x, (plane, runs, present_in_plane) in zip(xs, pool.map(sample_plane, xs)):
            export_grid[x, ys, zs] = plane[0]
            material_runs.extend(runs)
            for run in runs:
                occupied[run.x, run.y, run.z_start:run.z_stop] = True
            present |= present_in_plane
    if not present.all():
        missing = np.flatnonzero(~present).tolist()
        raise ValueError(f"Objects {missing} contain no occupied voxel centers with material contrast")
    export_grid[interior + (2,)] = SDF.voxel_signed_distance(occupied[interior])
    return export_grid, material_runs


class _IsotropicZRun(fdtd.AbsorbingObject):
    def _register_grid(self, grid, x, y, z):
        super()._register_grid(grid, x, y, z)
        # fdtd replaces the final Ez coefficient of a multi-cell z slice with
        # one from the far grid boundary, then computes a loss from it. A run
        # is 1 x 1 x N with scalar epsilon and sigma. Its Ex coefficients are
        # the actual material coefficients for all three polarizations.
        self.inverse_permittivity[..., -1, 2] = self.inverse_permittivity[..., -1, 0]
        self.absorption_factor[..., -1, 2] = self.absorption_factor[..., -1, 0]


def install_materials(grid: fdtd.Grid, material_runs: List[MaterialRun]) -> fdtd.Grid:
    """Register precomputed material runs on a fresh field grid for each shot."""
    if not material_runs:
        return grid

    # fdtd mutates the grid before validating each object. Restore that state if a
    # registration fails so a retry cannot leave a partly populated simulation.
    inverse = getattr(grid, "inverse_permittivity", None)
    registered = getattr(grid, "objects", [])
    original_count = len(registered)
    if inverse is not None:
        bounds = (
            slice(min(r.x for r in material_runs), max(r.x for r in material_runs) + 1),
            slice(min(r.y for r in material_runs), max(r.y for r in material_runs) + 1),
            slice(min(r.z_start for r in material_runs), max(r.z_stop for r in material_runs)),
        )
        view = inverse[bounds]
        backup = view.copy() if isinstance(view, np.ndarray) else view.clone()
    try:
        for run in material_runs:
            grid[run.x, run.y, run.z_start:run.z_stop] = _IsotropicZRun(
                permittivity=run.permittivity,
                conductivity=fdtd.backend.array(run.conductivity),
                name=run.name,
            )
    except BaseException:
        if inverse is not None:
            inverse[bounds] = backup
        for obj in registered[original_count:]:
            if getattr(grid, obj.name, None) is obj:
                delattr(grid, obj.name)
        del registered[original_count:]
        raise
    return grid


def add_objects_to_grid(objects: List[WorldObject], grid: fdtd.Grid, pml_thickness: int, n_jobs: int = 20):
    labels, runs = voxelize_world(objects, grid, pml_thickness, n_jobs)
    install_materials(grid, runs)
    return grid, labels
