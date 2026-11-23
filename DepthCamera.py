"""Calibrated pinhole depth observations of the opaque voxel surfaces in a saved RF scene."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def _calibration(acquisition, shape, width, height, hfov_deg):
    shape = tuple(shape)
    if (len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
            or any(not isinstance(n, (int, np.integer)) or n < 1 for n in (width, height))):
        raise ValueError("Volume and image dimensions must be positive integers")
    if not 0 < hfov_deg < 180:
        raise ValueError("Horizontal field of view must be between 0 and 180 degrees")
    spacing = float(acquisition["grid_spacing_m"])
    origin = np.asarray(acquisition["grid_origin_cells"])
    antennas = np.asarray(acquisition["antenna_positions_cells"], dtype=float)
    if (not np.isfinite(spacing) or spacing <= 0 or origin.shape != (3,) or not np.isfinite(origin).all()
            or antennas.ndim != 2 or antennas.shape[1] != 3 or len(antennas) == 0
            or not np.isfinite(antennas).all() or np.ptp(antennas[:, 0]) > 1e-8):
        raise ValueError("Expected a metric grid and a receive array in a constant-X plane")
    if "grid_shape" in acquisition and "pml_thickness" in acquisition:
        upper = np.asarray(acquisition["grid_shape"]) - acquisition["pml_thickness"]
        if not np.array_equal(origin + shape, upper):
            raise ValueError("Occupancy dimensions differ from the acquisition crop")
    focal = float(width / (2 * np.tan(np.deg2rad(hfov_deg) / 2)))
    return {
        "depth_camera_version": 1,
        "intrinsics": dict(width=int(width), height=int(height), hfov_deg=float(hfov_deg),
                           fx=focal, fy=focal, cx=(width - 1) / 2, cy=(height - 1) / 2),
        "position_m": (antennas.mean(axis=0) * spacing).tolist(),
        "camera_to_world_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        "grid_origin_cells": origin.tolist(), "grid_spacing_m": spacing,
        "crop_shape_cells": list(map(int, shape)),
        "depth_quantity": "optical_axis_distance_m", "no_return_value": 0.0,
        "scene_model": "opaque_unit_voxels_centered_on_integer_world_grid_coordinates",
    }


class DepthCamera:
    """Fixed rays and cell-entry distances, reusable across scenes sharing a calibration."""

    @classmethod
    def from_acquisition(cls, acquisition, shape, width=48, height=48, hfov_deg=90.0):
        return cls(_calibration(acquisition, shape, width, height, hfov_deg))

    def __init__(self, calibration):
        self.calibration = calibration
        self.shape = tuple(calibration["crop_shape_cells"])
        intrinsics = calibration["intrinsics"]
        self.image_shape = (intrinsics["height"], intrinsics["width"])
        rows, columns = np.indices(self.image_shape, dtype=float)
        pixels = np.column_stack(((columns.ravel() - intrinsics["cx"]) / intrinsics["fx"],
                                  (rows.ravel() - intrinsics["cy"]) / intrinsics["fy"],
                                  np.ones(rows.size)))
        directions = pixels @ np.asarray(calibration["camera_to_world_rotation"]).T
        spacing = calibration["grid_spacing_m"]
        # Local voxel (0,0,0) starts at crop_origin - 1/2 in global cell coordinates.
        origin = np.asarray(calibration["position_m"]) / spacing - calibration["grid_origin_cells"] + 0.5
        self._paths = [self._trace(origin, directions[start:start + 512], spacing)
                       for start in range(0, len(directions), 512)]

    def _trace(self, origin, directions, spacing):
        # Split each ray at the exact voxel faces. A segment's midpoint identifies the
        # crossed cell and its first face gives depth even when entering from the side.
        rays = len(directions)
        crossings = [np.zeros((rays, 1))]
        for axis, size in enumerate(self.shape):
            slope = directions[:, axis, None]
            times = np.full((rays, size + 1), np.inf)
            np.divide(np.arange(size + 1) - origin[axis], slope, out=times, where=slope != 0)
            crossings.append(np.maximum(times, 0))
        times = np.sort(np.concatenate(crossings, axis=1), axis=1)
        enter, leave = times[:, :-1], times[:, 1:]
        valid = np.isfinite(leave) & (leave > enter + 1e-12)
        mids = np.zeros_like(enter)
        np.add(enter, leave, out=mids, where=valid)
        cells = np.floor(origin + directions[:, None, :] * (mids[..., None] * 0.5)).astype(np.int64)
        valid &= ((cells >= 0) & (cells < self.shape)).all(axis=-1)

        # Pack only in-volume segments, preserving their order along each ray.
        counts = np.cumsum(valid, axis=1)
        rows, segments = np.nonzero(valid)
        slots = counts[rows, segments] - 1
        shape = (rays, max(1, int(counts[:, -1].max())))
        indices = np.full(shape, -1, dtype=np.intp)
        depths = np.zeros(shape, dtype=np.float32)
        indices[rows, slots] = np.ravel_multi_index(cells[rows, segments].T, self.shape)
        depths[rows, slots] = enter[rows, segments] * spacing
        return indices, depths

    def render(self, occupied):
        """Return first-surface axial depth in metres; zero denotes an empty pixel."""
        occupied = np.asarray(occupied)
        if occupied.shape != self.shape or occupied.dtype != np.bool_:
            raise ValueError("Expected a Boolean occupancy mask on the calibrated crop")
        # Index -1 is the empty sentinel in the packed ray paths.
        scene = np.append(occupied.ravel(), False)
        output = []
        for indices, depths in self._paths:
            solid = scene[indices]
            row = np.arange(len(indices))
            col = solid.argmax(axis=1)
            valid = solid[row, col]
            depth = np.where(valid, depths[row, col], np.float32(0))
            if np.any(valid & (depth <= 0)):
                raise ValueError("The camera center must lie outside the observed material")
            output.append(depth)
        return np.concatenate(output).reshape(self.image_shape)


def _write_calibration(path, calibration):
    # Concurrent writers may share identical calibration, e.g. separate scene batches.
    with NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as handle:
        tmp = Path(handle.name)
        json.dump(calibration, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != calibration:
                raise ValueError(f"Incompatible depth camera settings in {path}") from None
    finally:
        tmp.unlink(missing_ok=True)


def generate(scenes, output, width=48, height=48, hfov_deg=90.0):
    """Write a shared camera.json and one metric depth.npy image per scene."""
    scenes, output = Path(scenes), Path(output)
    roots = [scenes] if (scenes / "grid.npy").is_file() else sorted(scenes.glob("scene_*"))
    if not roots:
        raise ValueError(f"No saved scenes in {scenes}")
    return generate_frames(roots, output, width, height, hfov_deg)


def generate_frames(roots, output, width=48, height=48, hfov_deg=90.0):
    """Render a specified set of published scenes, sharing their camera calibration."""
    roots = list(roots)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    camera = None
    for scene in roots:
        labels = np.load(scene / "grid.npy", mmap_mode="r", allow_pickle=False)
        if labels.ndim != 4 or labels.shape[-1] != 3:
            raise ValueError(f"Expected a saved 3-D field with three channels in {scene}")
        acq = json.loads((scene / "acquisition.json").read_text())
        spec = _calibration(acq, labels.shape[:3], width, height, hfov_deg)
        if camera is None:
            _write_calibration(output / "camera.json", spec)
            camera = DepthCamera(spec)
        elif camera.calibration != spec:
            raise ValueError(f"Camera calibration changed at {scene}")
        # The renderer sees only solid/empty geometry; the RF material channels
        # and object descriptions are never used to color, identify, or select a hit.
        sdf = labels[..., 2]
        if not (np.isfinite(sdf).all() or np.isposinf(sdf).all()):
            raise ValueError(f"Invalid signed-distance field in {scene}")
        depth = camera.render(sdf < 0)
        dest = output / scene.name
        dest.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=dest, delete=False) as handle:
            tmp = Path(handle.name)
        try:
            with tmp.open("wb") as handle:
                np.save(handle, depth, allow_pickle=False)
            tmp.replace(dest / "depth.npy")
        finally:
            tmp.unlink(missing_ok=True)
    return len(roots)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True, help="One scene directory or parent of scene_*")
    p.add_argument("--output", type=Path, required=True, help="Destination for the depth camera frames")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--hfov-deg", type=float, default=90.0)
    args = p.parse_args()
    print(f"Wrote {generate(**vars(args))} depth frames to {args.output}")
