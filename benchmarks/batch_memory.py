"""Measure two-scene, five-shot recording time, Linux peak RSS and CUDA allocation.

From a simulator checkout:
    PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python benchmarks/batch_memory.py --side 64
Use the same Python environment and CPU affinity for both revisions. Run once to
populate each revision's compiled kernel cache before collecting timings.
"""
import argparse
import hashlib
import json
import os
import resource
import time

import numpy as np
import torch

from Acquisition import AcquisitionConfig, BEAMS
from BatchFDTD import batch_metadata, record_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--side", type=int, default=64, help="Side length of the cubic benchmark in cells")
    geometry.add_argument("--default-acquisition", action="store_true",
                          help="Use the material-resolved default grid and a 14-cell conductive cube")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-parallel-beams", type=int, help="Maximum beams per solver launch")
    parser.add_argument("--save", help="Optional path for the raw output voltage array")
    args = parser.parse_args()
    if args.side < 40 or args.repeat < 1:
        parser.error("use a side of at least 40 and at least one recording")
    solver_options = {}
    if args.max_parallel_beams is not None:
        if args.max_parallel_beams < 1:
            parser.error("--max-parallel-beams must be positive")
        solver_options["max_parallel_beams"] = args.max_parallel_beams
    if args.default_acquisition:
        config = AcquisitionConfig()
        target = (1, slice(80, 94), slice(101, 115), slice(101, 115))
        material = (54.16, 11.29)
    else:
        side = args.side
        config = AcquisitionConfig(grid_shape=(side,) * 3, grid_spacing_m=.0045, pml_thickness=4,
                                   pitch_cells=8, standoff_cells=3, samples_per_shot=128)
        mid = side // 2
        target = (1, slice(19, 27), slice(mid - 5, mid + 5), slice(mid - 4, mid + 4))
        material = (4.5, .035)
    eps = np.ones((2, *config.grid_shape), dtype=np.float32)
    sigma = np.zeros_like(eps)
    eps[target], sigma[target] = material
    metadata = batch_metadata(config)
    steps = 1 + (config.samples_per_shot - 1) * metadata["sample_stride"]
    is_cuda = args.device == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        voltages = record_batch(config, eps, sigma, device=args.device, dtype="float32", **solver_options)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    if args.save:
        np.save(args.save, voltages)
    result = {
        "grid_shape": config.grid_shape, "grid_spacing_m": config.spacing, "pml": config.pml_thickness,
        "scenes": len(eps), "beams": len(BEAMS), "steps": steps, "samples": config.samples_per_shot,
        "device": args.device, "max_parallel_beams": args.max_parallel_beams,
        "elapsed_seconds": elapsed, "omp_threads": os.getenv("OMP_NUM_THREADS"),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "voltage_sha256": hashlib.sha256(voltages.tobytes()).hexdigest(),
        "echo_rms_v": float(np.sqrt(np.mean((voltages[1] - voltages[0]) ** 2))),
    }
    if is_cuda:
        result["gpu"] = torch.cuda.get_device_name()
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
