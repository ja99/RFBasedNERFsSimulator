import pathlib
from datetime import datetime
import numpy as np


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Local-to-world rotation for roll, pitch and yaw in degrees."""
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
    return Rz @ Ry @ Rx


def rotate_vec(vec: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotation_matrix(roll, pitch, yaw) @ vec


# rescale a value from one range to another
def rescale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min






def create_save_path() -> pathlib.Path:
    SAVE_PATH = pathlib.Path("beamforming_data") / str(datetime.now())
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    return SAVE_PATH



def save_grid(SAVE_PATH: pathlib.Path, export_grid: np.ndarray, pml_thickness: int, sensor_x: int):
    stop = -pml_thickness if pml_thickness else None
    lateral = slice(pml_thickness, stop)
    np.save(SAVE_PATH / "grid.npy", export_grid[sensor_x:stop, lateral, lateral])


