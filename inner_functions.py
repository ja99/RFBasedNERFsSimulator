from typing import List, NamedTuple, Tuple

import numpy as np

from SDF import coordinate_roundoff
from WorldObjectType import WorldObject


class MaterialRun(NamedTuple):
    x: int
    y: int
    z_start: int
    z_stop: int
    name: str
    permittivity: float
    conductivity: float


def _sampled_distances(shape, points):
    distance = shape.distances(points)
    if not np.isfinite(distance).all():
        raise ValueError("An object returned a non-finite signed distance")
    tolerance = coordinate_roundoff(shape.get_bounding_box())
    return np.where(np.abs(distance) <= tolerance, 0.0, distance)


def min_dist_to_obj(objects: List[WorldObject], point: np.ndarray) -> Tuple[float, int]:
    if not objects:
        return np.inf, -1
    distances = [float(_sampled_distances(obj.sdf, point)) for obj in objects]
    nearest = int(np.argmin(distances))
    return distances[nearest], nearest


def voxelize_slab(objects, x_range, y_range, z_range, background_permittivity=1.0,
                  background_inverse=None, material_dtype=np.float64):
    """Sample cell centers and derive FDTD material runs from their ownership."""
    if background_inverse is None:
        background_inverse = np.reciprocal(np.asarray(background_permittivity, dtype=float))
    axes = [np.arange(start, stop) for start, stop in (x_range, y_range, z_range)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = points.shape[:-1]
    labels = np.empty((*shape, 3), dtype=np.float64)
    labels[..., 0] = background_permittivity
    labels[..., 1] = 0.0
    distance = labels[..., 2]
    distance.fill(np.inf)
    owners = np.full(shape, -1, dtype=np.int32)
    present = np.zeros(len(objects), dtype=bool)

    for index, obj in enumerate(objects):
        candidate = _sampled_distances(obj.sdf, points)
        closer = candidate < distance
        distance[closer] = candidate[closer]
        inside = candidate < 0
        occupied = closer & inside
        # Legacy scenes may wrap scalars in one-element NumPy arrays or Torch tensors.
        values = [v.item() if hasattr(v, "item") else v for v in (obj.permittivity, obj.conductivity)]
        eps, sigma = np.asarray(values, dtype=material_dtype)
        if not np.isfinite([eps, sigma]).all():
            raise ValueError("Material permittivity and conductivity must be finite")
        # Compare the inverse coefficient used by FDTD. Taking its reciprocal
        # to label the background can shift nominal epsilon by one ulp.
        inverse_epsilon = np.reciprocal(np.asarray(eps, dtype=background_inverse.dtype))
        has_contrast = (inverse_epsilon != background_inverse) | (sigma != 0)
        present[index] = (inside & has_contrast).any()
        np.copyto(owners, np.where(has_contrast, index, -1), where=occupied)
        np.copyto(labels[..., 0], np.where(has_contrast, eps, background_permittivity), where=occupied)
        labels[..., 1][occupied] = sigma

    runs = []
    for local_x, local_y in np.argwhere((owners >= 0).any(axis=-1)):
        x = int(local_x) + x_range[0]
        y = int(local_y) + y_range[0]
        row = owners[local_x, local_y]
        # Air sentinels close each row, including an object touching its last cell.
        edges = np.flatnonzero(np.diff(row, prepend=-1, append=-1))
        for start, stop in zip(edges[:-1], edges[1:]):
            if row[start] >= 0:
                z_start = z_range[0] + int(start)
                z_stop = z_range[0] + int(stop)
                eps, sigma = labels[local_x, local_y, start, :2]
                runs.append(MaterialRun(x, y, z_start, z_stop, f"{x},{y},{z_start}:{z_stop},",
                                        float(eps), float(sigma)))
    return labels, runs, present


def add_objects_to_grid_inner(objects, objects_queue, subgrid_queue, kill_me_queue,
                              subgrid_q_lock, kill_me_q_lock, process_id,
                              x_range, y_range, z_range):
    """Queue-based entry point for archived simulator drivers."""
    labels, runs, _ = voxelize_slab(objects, x_range, y_range, z_range)
    for run in runs:
        objects_queue.put(run)
    with subgrid_q_lock:
        subgrid_queue.put((x_range, y_range, labels))
    with kill_me_q_lock:
        kill_me_queue.put(process_id)
