from abc import abstractmethod, ABC
from dataclasses import dataclass
import helpers
import numpy as np

class SDFObject(ABC):
    @abstractmethod
    def sdf(self, point: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_bounding_box(self) -> np.ndarray:
        pass


@dataclass
class Sphere(SDFObject):
    position: np.ndarray
    radius: float

    def sdf(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.position) - self.radius

    def get_bounding_box(self) -> np.ndarray:
        lower_corner = self.position - self.radius
        upper_corner = self.position + self.radius
        return np.array([lower_corner, upper_corner])


@dataclass
class Box(SDFObject):
    position: np.ndarray  # [x, y, z]
    size: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [roll, pitch, yaw]

    def sdf(self, point: np.ndarray) -> float:
        local_point = helpers.rotate_vec(point - self.position, *-self.rotation)

        # Absolute distance from the local_point to the box edges
        d = np.abs(local_point) - self.size / 2

        # Calculate the outside distance (this will be positive if the point is outside the box, zero if on the surface)
        outside_distance = np.linalg.norm(np.clip(d, 0.0, None), ord=2)

        # Calculate the inside distance (this will be negative if the point is inside the box, zero if on the surface)
        inside_distance = np.min(np.maximum(d[0], np.maximum(d[1], d[2])), 0)

        # The SDF is outside_distance + inside_distance (this will be positive outside the box, negative inside)
        return outside_distance + inside_distance

    def get_bounding_box(self) -> np.ndarray:
        # Step 1: Compute the 8 corner points of the rotated box in local coordinates
        half_size = self.size / 2
        corners_local = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [ half_size[0],  half_size[1],  half_size[2]],
        ])

        # Step 2: Transform the corner points to world coordinates
        # For this we need to implement a rotation function, which is not provided here.
        corners_world = [helpers.rotate_vec(corner, *self.rotation) + self.position for corner in corners_local]

        # Step 3: Find min and max coordinates
        min_corner = np.min(corners_world, axis=0)
        max_corner = np.max(corners_world, axis=0)

        # Step 4: Return AABB
        return np.array([min_corner, max_corner])