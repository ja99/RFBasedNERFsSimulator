import pathlib
from datetime import datetime
import numpy as np


def rotate_vec(vec: np.ndarray,  # [x, y, z]
               roll: float,  # degrees
               pitch: float,  # degrees
               yaw: float  # degrees
               ) -> np.ndarray:  # [x, y, z]
    # Convert angles from degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Define rotation matrices
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll), -np.sin(roll)],
                   [0.0, np.sin(roll), np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    # Compute total rotation matrix
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    # Rotate vector
    rotated_vec = np.matmul(R, vec)

    return rotated_vec


# rescale a value from one range to another
def rescale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min






def create_save_path() -> pathlib.Path:
    SAVE_PATH = pathlib.Path("beamforming_data") / str(datetime.now())
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    return SAVE_PATH



def save_grid(SAVE_PATH: pathlib.Path, export_grid: np.ndarray, pml_thickness: int, sensor_x: int):
    np.save(SAVE_PATH / "grid.npy", export_grid[sensor_x:-pml_thickness, pml_thickness:-pml_thickness, pml_thickness:-pml_thickness])


