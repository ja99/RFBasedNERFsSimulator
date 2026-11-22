from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import helpers


def voxel_signed_distance(occupied: np.ndarray) -> np.ndarray:
    """Exact cell-center distance to the boundary of the occupied unit voxel cubes.

    The world beyond the array is air. A grid with no occupied voxels has distance +inf.
    A voxel face is a square of side one. The orthogonal projection of an integer
    cell center onto that square, clamped to its edges, lands at its center, an
    edge midpoint, or a corner. Searching those nine points on each exposed face
    therefore gives the exact Euclidean distance, including at edges and corners.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 3 or min(occupied.shape) < 1:
        raise ValueError("Occupancy must be a nonempty three-dimensional grid")
    if not occupied.any():
        return np.full(occupied.shape, np.inf)

    # Coordinates on this doubled lattice put centers at odd indices and
    # voxel faces at even indices along the face normal. Shared points coalesce.
    surface = np.zeros(tuple(2 * n + 1 for n in occupied.shape), dtype=bool)
    for normal in range(3):
        faces = np.diff(occupied, axis=normal, prepend=False, append=False)
        tangent1, tangent2 = [axis for axis in range(3) if axis != normal]
        indices = [slice(None, None, 2)] * 3
        for i in range(3):
            indices[tangent1] = slice(i, i + 2 * occupied.shape[tangent1], 2)
            for j in range(3):
                indices[tangent2] = slice(j, j + 2 * occupied.shape[tangent2], 2)
                surface[tuple(indices)] |= faces
    tree = cKDTree(np.argwhere(surface))
    del surface
    plane = np.empty((*occupied.shape[1:], 3), dtype=np.float64)
    plane[..., 1], plane[..., 2] = 2 * np.indices(occupied.shape[1:]) + 1
    distance = np.empty(occupied.shape, dtype=np.float64)
    for x in range(occupied.shape[0]):
        plane[..., 0] = 2 * x + 1
        distance[x] = tree.query(plane)[0] * 0.5
    distance[occupied] *= -1
    return distance


def coordinate_roundoff(bounds: np.ndarray) -> float:
    """Allow for floating-point coordinate subtraction and rotation, in cells."""
    scale = max(1.0, np.abs(bounds).max(), np.ptp(bounds, axis=0).max())
    return 16 * np.finfo(np.float64).eps * scale


class SDFObject(ABC):
    @abstractmethod
    def sdf(self, point: np.ndarray) -> float:
        pass

    def distances(self, points: np.ndarray) -> np.ndarray:
        """Evaluate a grid of points, including shapes with a point-only SDF."""
        return np.apply_along_axis(self.sdf, -1, points)

    @abstractmethod
    def get_bounding_box(self) -> np.ndarray:
        pass


@dataclass
class Sphere(SDFObject):
    position: np.ndarray
    radius: float

    def sdf(self, point: np.ndarray) -> float:
        return float(self.distances(point))

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.position, axis=-1) - self.radius

    def get_bounding_box(self) -> np.ndarray:
        return np.array([self.position - self.radius, self.position + self.radius])


@dataclass
class Box(SDFObject):
    position: np.ndarray
    size: np.ndarray  # full extents along local x, y, z
    rotation: np.ndarray  # roll, pitch, yaw in degrees

    def sdf(self, point: np.ndarray) -> float:
        return float(self.distances(point))

    def distances(self, points: np.ndarray) -> np.ndarray:
        # For row vectors, multiplying on the right by R applies the inverse rotation.
        local = (points - self.position) @ helpers.rotation_matrix(*self.rotation)
        q = np.abs(local) - self.size / 2
        exterior = np.linalg.norm(np.maximum(q, 0), axis=-1)
        interior = np.minimum(q.max(axis=-1), 0)
        return exterior + interior

    def get_bounding_box(self) -> np.ndarray:
        half_extent = np.abs(helpers.rotation_matrix(*self.rotation)) @ (self.size / 2)
        return np.array([self.position - half_extent, self.position + half_extent])
