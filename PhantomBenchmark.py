"""Generate balanced synthetic RF phantoms using the batched acquisition solver."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from Acquisition import AcquisitionConfig
from BenchmarkProvenance import same_acquisition_manifest
from BatchFDTD import batch_metadata, record_batch
from SDF import Box, Sphere, voxel_signed_distance
from WorldObjectType import WorldObject
from inner_functions import voxelize_slab


VERSION = 1
MATERIALS = {
    "dielectric": ((2.5, 6.0), (0.01, 0.12), 0.0045),
    "thesis": ((24.16, 83.16), (6.29, 15.29), None),
}


def configuration(material):
    return AcquisitionConfig(grid_shape=(40, 40, 40), pml_thickness=4, standoff_cells=3,
                             pitch_cells=8, samples_per_shot=128, grid_spacing_m=MATERIALS[material][2])


def draw_phantom(index, seed, material, count=1):
    """Balance each adjacent index pair while keeping its type independent of position and material."""
    if min(index, seed) < 0 or count not in (1, 2):
        raise ValueError("Use nonnegative indices and seeds, and one or two objects")
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
    pair_rng = np.random.default_rng(np.random.SeedSequence([seed, index // 2, 1]))
    first_is_box = bool((index % 2) ^ pair_rng.integers(2))
    eps_range, sigma_range, _ = MATERIALS[material]
    config = configuration(material)
    lower = np.array([config.pml_thickness + config.standoff, config.pml_thickness, config.pml_thickness])
    upper = np.array(config.grid_shape) - config.pml_thickness - 1
    objects, truth = [], []
    for j in range(count):
        box = first_is_box if j == 0 else bool(rng.integers(2))
        for _ in range(1024):
            radius = rng.uniform(3.25, 6.25) if count == 1 else rng.uniform(2.8, 4.7)
            long, mid = rng.uniform(1.15, 1.65), rng.uniform(0.85, 1.1)
            size = radius * np.cbrt(4 * np.pi / 3) * np.array([long, mid, 1 / (long * mid)])
            angles = Rotation.random(random_state=rng).as_euler("xyz", degrees=True)
            # Use the same admissible center distribution for an equal-volume sphere and box.
            bounds = np.maximum(radius, Box(np.zeros(3), size, angles).get_bounding_box()[1])
            low = lower + bounds + [1.05, 0.05, 0.05]
            high = upper - bounds - 0.05
            if (low >= high).any():
                continue
            center = rng.uniform(low, high)
            shape = Box(center, size, angles) if box else Sphere(center, radius)
            lo, hi = shape.get_bounding_box()
            if all(np.any((lo > obj.sdf.get_bounding_box()[1] + 1) |
                          (hi < obj.sdf.get_bounding_box()[0] - 1)) for obj in objects):
                break
        else:
            raise RuntimeError("Could not place disjoint phantoms in the acquisition volume")
        eps, sigma = float(rng.uniform(*eps_range)), float(rng.uniform(*sigma_range))
        objects.append(WorldObject(shape, eps, sigma))
        description = dict(shape="box" if box else "sphere", center_cells=center.tolist(),
                           permittivity=eps, conductivity_s_per_m=sigma)
        if box:
            description.update(sides_cells=size.tolist(), euler_xyz_deg=angles.tolist())
        else:
            description.update(radius_cells=radius)
        truth.append(description)
    return objects, truth


def rasterize(config, origin, objects):
    """Use the same voxel ownership and signed-distance convention as the original acquisition."""
    upper = np.array(config.grid_shape) - config.pml_thickness
    ranges = tuple(zip(origin, upper))
    labels, _, present = voxelize_slab(objects, *ranges)
    if not present.all():
        raise ValueError("Phantom has no material contrast on this grid")
    labels[..., 2] = voxel_signed_distance((labels[..., 0] != 1) | (labels[..., 1] != 0))
    return labels


def generate(root, *, scenes, seed, start_index, material, objects, batch_size, device, dtype):
    if min(scenes, batch_size) <= 0 or min(start_index, seed) < 0:
        raise ValueError("Scene and batch counts must be positive and seeds and indices nonnegative")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = configuration(material)
    metadata = batch_metadata(config)
    origin = metadata["grid_origin_cells"]
    region = tuple(slice(start, n - config.pml_thickness) for start, n in zip(origin, config.grid_shape))
    source_dir = Path(__file__).resolve().parent
    geometry_sources = ("PhantomBenchmark.py", "Acquisition.py", "Sensor.py", "WorldGenerator.py",
                        "SDF.py", "inner_functions.py", "helpers.py")
    geometry_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                       for name in geometry_sources}
    manifest = dict(phantom_benchmark_version=VERSION, source_sha256=geometry_hashes, seed=seed,
                    objects_per_scene=objects, material=material,
                    engine="BatchFDTD", dtype=dtype, acquisition=metadata,
                    solver_sha256=hashlib.sha256(b"".join((source_dir / f).read_bytes()
                                                 for f in ("BatchFDTD.cpp", "BatchFDTD.py"))).hexdigest())
    manifest_file = root / "benchmark.json"
    if manifest_file.exists():
        if not same_acquisition_manifest(json.loads(manifest_file.read_text()), manifest, "source_sha256"):
            raise ValueError(f"Refusing to mix a new generator or solver with existing data in {root}")
    else:
        if any(root.glob("scene_*")):
            raise ValueError("Cannot resume a dataset without its benchmark manifest")
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = []
    required = ("grid.npy", "sensor_samples.npy", "empty_sensor_samples.npy", "acquisition.json", "scene.json")
    for i in range(start_index, start_index + scenes):
        scene = root / f"scene_{i:06d}"
        if scene.exists():
            if not all((scene / key).is_file() for key in required):
                raise ValueError(f"Incomplete published scene {scene}")
        else:
            pending.append(i)
    if not pending:
        return
    empty_eps = np.ones((1, *config.grid_shape), dtype=dtype)
    empty = record_batch(config, empty_eps, np.zeros_like(empty_eps), device=device, dtype=dtype)[0]
    acquisition_json = json.dumps(metadata, indent=2) + "\n"
    started = monotonic()
    for offset in range(0, len(pending), batch_size):
        indices = pending[offset:offset + batch_size]
        eps = np.ones((len(indices), *config.grid_shape), dtype=dtype)
        sigma = np.zeros_like(eps)
        batch = []
        for b, i in enumerate(indices):
            world, truth = draw_phantom(i, seed, material, objects)
            labels = rasterize(config, origin, world)
            eps[(b, *region)] = labels[..., 0]
            sigma[(b, *region)] = labels[..., 1]
            batch.append((labels, truth))
        shots = record_batch(config, eps, sigma, device=device, dtype=dtype)
        for i, shot, (labels, truth) in zip(indices, shots, batch):
            with TemporaryDirectory(prefix=".acquiring-", dir=root) as scratch:
                folder = Path(scratch) / "scene"
                folder.mkdir()
                np.save(folder / "grid.npy", labels.astype(np.float32))
                np.save(folder / "sensor_samples.npy", shot)
                np.save(folder / "empty_sensor_samples.npy", empty)
                (folder / "acquisition.json").write_text(acquisition_json)
                (folder / "scene.json").write_text(json.dumps(dict(index=i, objects=truth), indent=2) + "\n")
                folder.rename(root / f"scene_{i:06d}")
        print(f"{offset + len(indices)}/{len(pending)} scenes in {monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--material", choices=MATERIALS, default="dielectric")
    parser.add_argument("--objects", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--depth-output", type=Path, help="Also write pinhole depth frames for the requested scenes")
    args = vars(parser.parse_args())
    depth_output = args.pop("depth_output")
    generate(**args)
    if depth_output is not None:
        from DepthCamera import generate_frames
        paths = [args["root"] / f"scene_{i:06d}" for i in range(args["start_index"], args["start_index"] + args["scenes"])]
        generate_frames(paths, depth_output)
