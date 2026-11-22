"""A metre-scale, single-excitation laboratory benchmark with compact on-disk geometry labels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from Acquisition import AcquisitionConfig
from BatchFDTD import batch_metadata, record_batch
from SDF import Box, Sphere, coordinate_roundoff


CONFIG = AcquisitionConfig(grid_shape=(146,146,146), pml_thickness=8, grid_spacing_m=.008,
                           standoff_cells=5, pitch_cells=8, samples_per_shot=256)
ANGLES = ((0., 0.),)


def phantom(seed, i):
    rng = np.random.default_rng(np.random.SeedSequence([seed,i,0]))
    coin = np.random.default_rng(np.random.SeedSequence([seed,i//2,1])).integers(2)
    kind = "box" if (i % 2) ^ coin else "sphere"
    center = np.array([13 + rng.uniform(25,114), 73 + rng.uniform(-46,46), 73 + rng.uniform(-46,46)])
    radius = rng.uniform(4.5,8.)
    long, mid = rng.uniform(1.15,1.60), rng.uniform(.85,1.1)
    sides = np.cbrt(4*np.pi/3)*radius*np.array([long,mid,1/(long*mid)])
    angles = Rotation.random(random_state=rng).as_euler("xyz", degrees=True)
    obj = Box(center,sides,angles) if kind == "box" else Sphere(center,radius)
    if obj.get_bounding_box()[1,0] > CONFIG.grid_shape[0] - CONFIG.pml_thickness - 1:
        # A very long cuboid at the back can exceed the ROI. Reflect its center
        # about the limiting plane while retaining its orientation and volume.
        center[0] -= 2*max(0., obj.get_bounding_box()[1,0] - (CONFIG.grid_shape[0]-CONFIG.pml_thickness-1))
    bounds = obj.get_bounding_box()
    if (bounds[0] < [13,8,8]).any() or (bounds[1] > 137).any():
        raise ValueError(f"Phantom outside the usable volume: {bounds}")
    return obj, dict(centers_cells=center, kind=kind, radius_cells=radius, sizes_cells=sides, euler_deg=angles,
                     permittivity=rng.uniform(1.5,2.3), conductivity_s_per_m=rng.uniform(.001,.012))


def insert(eps, sig, obj, label):
    lo, hi = obj.get_bounding_box()
    region = tuple(slice(int(np.floor(a)), int(np.ceil(b))+1) for a,b in zip(lo,hi))
    axes = [np.arange(s.start,s.stop) for s in region]
    distances = obj.distances(np.stack(np.meshgrid(*axes,indexing="ij"), axis=-1))
    mask = distances < -coordinate_roundoff(obj.get_bounding_box())
    eps[region][mask] = label["permittivity"]
    sig[region][mask] = label["conductivity_s_per_m"]


def generate(root, scenes, seed, start, sealed, batch_size, dtype):
    if scenes < 2 or batch_size < 1 or min(seed,start)<0:
        raise ValueError("Use at least two scenes, a positive batch size and nonnegative seed and start")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "shots"
    cache.mkdir(exist_ok=True)
    acq = batch_metadata(CONFIG, ANGLES)
    manifest = dict(acquisition=acq, dtype=dtype, seed=seed, start=start, scenes=scenes, sealed=sealed,
                    sources={k:sha256((Path(__file__).parent/k).read_bytes()).hexdigest()
                             for k in ("MeterBenchmark.py","BatchFDTD.py","BatchFDTD.cpp","SDF.py","Sensor.py","Acquisition.py","helpers.py")})
    config_file = root/"benchmark.json"
    if config_file.exists() and json.loads(config_file.read_text()) != manifest:
        raise ValueError(f"Different generator, acquisition or requested cohort already at {root}")
    config_file.write_text(json.dumps(manifest,indent=2)+"\n")
    air = np.ones((1,*CONFIG.grid_shape), dtype=dtype)
    empty = record_batch(CONFIG,air,np.zeros_like(air),dtype=dtype,angles=ANGLES)[0]
    data = []
    start_time = monotonic()
    for s in range(start,start+scenes,batch_size):
        ids = list(range(s, min(s+batch_size,start+scenes)))
        entries = [phantom(seed,i) for i in ids]
        path = cache / f"shot_{s:06d}_{ids[-1]:06d}.npy"
        if not path.is_file():
            eps=np.ones((len(ids),*CONFIG.grid_shape),dtype=dtype)
            sig=np.zeros_like(eps)
            for j,(obj, label) in enumerate(entries):
                insert(eps[j],sig[j],obj,label)
            shots=record_batch(CONFIG,eps,sig,angles=ANGLES,dtype=dtype)
            with NamedTemporaryFile(dir=cache,delete=False) as tmp:
                np.save(tmp, shots)
            Path(tmp.name).replace(path)
        shots = np.load(path)
        if shots.shape != (len(ids),*empty.shape) or not np.isfinite(shots).all():
            raise ValueError(f"Invalid receiver file at {path}")
        data.extend((shot,label) for shot,(_,label) in zip(shots,entries))
        if (s-start)//batch_size % 4 == 0 or ids[-1]==start+scenes-1:
            print(f"{ids[-1]-start+1}/{scenes} in {monotonic()-start_time:.1f}s",flush=True)
    split=np.full(scenes,"test" if sealed else "train", dtype="<U10")
    if not sealed:
        val=np.random.default_rng(1741).permutation(scenes)[:max(1,scenes//10)]
        split[val]="validation"
    labels={key:np.asarray([row[1][key] for row in data]) for key in data[0][1]}
    with NamedTemporaryFile(dir=root,delete=False) as handle:
        np.savez_compressed(handle, sensor_samples=np.stack([row[0] for row in data]), empty_sensor_samples=empty,
                            acquisition_json=json.dumps(acq), split=split,
                            scene_ids=[f"meter_{i:06d}" for i in range(start,start+scenes)], **labels)
    Path(handle.name).replace(root/"signals.npz")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--scenes",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--sealed",action="store_true")
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--dtype",choices=["float32","float64"],default="float32")
    generate(**vars(p.parse_args()))
