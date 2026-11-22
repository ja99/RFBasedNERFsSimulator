"""Fit the one or two inclusions selected by EchoSet to all measured RF voltages.

The input NPZ comes from EchoSet.py predict --instances. Each two-inclusion
trial solves Maxwell's equations with both materials on the same grid.
"""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from BatchFDTD import record_batch
from PhantomBenchmark import MATERIALS
from RefinePhantom import EchoFit, Shape, _acquisition, _best, _coarse_beams, _gauss_newton, refine_scene
from SDF import voxel_signed_distance


@dataclass
class Pair:
    shapes: tuple

    @property
    def slices(self):
        split = 6 if self.shapes[0].kind == "sphere" else 11
        return slice(0, split), slice(split, split + (6 if self.shapes[1].kind == "sphere" else 11))

    def constrain(self, p):
        return np.concatenate([model.constrain(p[part]) for model, part in zip(self.shapes, self.slices)])


class JointFit(EchoFit):
    def parts(self, model, p):
        first, second = model.shapes
        a, b = [p[part] for part in model.slices]
        masks = [self.mask(first, a), self.mask(second, b)]
        intersection = masks[0] & masks[1]
        if intersection.any():
            # Intermediate estimates can overlap. Give each such voxel to the
            # inclusion in which its center lies deeper, so it has one material.
            points = self.points[intersection]
            use_first = first.distance(a, points) <= second.distance(b, points)
            masks[0][intersection] = use_first
            masks[1][intersection] = ~use_first
        return masks, (self.materials(a), self.materials(b))

    def evaluate(self, model, candidates, beams):
        keys, missing = [], {}
        for p in candidates:
            masks, materials = self.parts(model, p)
            values = np.asarray(materials, dtype=self.dtype)
            key = (beams, values.tobytes(), *(np.packbits(mask).tobytes() for mask in masks))
            keys.append(key)
            if key not in self.cache:
                missing[key] = (masks, values)
        jobs = list(missing)
        for start in range(0, len(jobs), self.batch_size):
            group = jobs[start:start + self.batch_size]
            eps = np.ones((len(group), *self.config.grid_shape), dtype=self.dtype)
            sigma = np.zeros_like(eps)
            for row, key in enumerate(group):
                masks, values = missing[key]
                for mask, (er, conductivity) in zip(masks, values):
                    eps[(row, *self.region)][mask] = er
                    sigma[(row, *self.region)][mask] = conductivity
            raw = record_batch(self.config, eps, sigma, angles=tuple(self.angles[i] for i in beams),
                               device=self.device, dtype=self.dtype).astype(float)
            echoes = (raw - self.empty[:, beams, :]) * self.gain[:, beams, :]
            self.cache.update((key, wave.ravel()) for key, wave in zip(group, echoes))
            self.candidate_shots += len(group) * len(beams)
            self.forward_batches += 1
        return np.stack([self.cache[key] for key in keys])


def _slots(path, shape, region):
    prediction = np.load(path, allow_pickle=False)
    if not isinstance(prediction, np.lib.npyio.NpzFile):
        raise ValueError("Expected the NPZ from EchoSet --instances")
    with prediction:
        if "masks" not in prediction:
            raise ValueError("Expected predicted masks in the NPZ from EchoSet --instances")
        slots = prediction["masks"]
        centers = prediction["centers_cells"] if "centers_cells" in prediction else None
        probability = float(prediction["probability_two"]) if "probability_two" in prediction else None
    crop = tuple(s.stop - s.start for s in region)
    if slots.ndim != 4 or len(slots) not in (1, 2) or not np.all((slots == 0) | (slots == 1)):
        raise ValueError("Expected one or two binary masks from EchoSet --instances")
    if slots.shape[1:] == shape:
        slots = slots[(slice(None), *region)]
        if centers is not None:
            centers = centers - np.array([s.start for s in region])
    if slots.shape[1:] != crop or (centers is not None and centers.shape != (len(slots), 3)):
        raise ValueError(f"EchoSet masks must use the acquisition crop {crop}")
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("Invalid two-instance probability")
    slots = slots.astype(bool)
    for i in range(len(slots)):
        if not slots[i].any():
            if centers is None or not np.isfinite(centers[i]).all():
                raise ValueError("An empty predicted slot needs its RF-predicted center")
            center = np.clip(centers[i], 0, np.array(crop) - 1)
            coords = np.moveaxis(np.indices(crop), 0, -1)
            slots[i] = np.linalg.norm(coords - center, axis=-1) <= 1.75
    return slots, probability


def _material_steps(model, p, step):
    delta = np.zeros_like(p)
    for part in model.slices:
        delta[part][-2:] = step
    return delta


def _verify(fit, model, p, trials):
    return _best(fit, model, [p, *trials], tuple(range(len(fit.angles))))


def _paired_proposals(fit, model, p, options, coarse):
    """Screen all combinations of two local change sets, then verify in full Maxwell."""
    observed = fit.observed(coarse)
    norm = np.linalg.norm(observed)
    base = fit.evaluate(model, [p], coarse)[0]
    target = (observed - base) / norm
    deltas = [(fit.evaluate(model, group, coarse) - base) / norm for group in options]
    # Both local perturbations were computed with the other material present.
    # Their first-order sum cheaply ranks thousands of combined updates.
    left, right = deltas[0] - target, deltas[1]
    score = (left**2).sum(1)[:, None] + (right**2).sum(1)[None, :] + 2 * (left @ right.T)
    material_ids = np.flatnonzero(_material_steps(model, p, 1.))
    probes = []
    for index in material_ids:
        low, high = p.copy(), p.copy()
        low[index], high[index] = max(0., p[index]-.1), min(1., p[index]+.1)
        probes.extend((low, high))
    waves = fit.evaluate(model, probes, coarse)
    widths = np.array([probes[2*i+1][j] - probes[2*i][j] for i, j in enumerate(material_ids)])
    jac = (waves[1::2] - waves[::2]).T / (widths * norm)
    gram = jac.T @ jac
    projection = -((left @ jac)[:, None, :] + (right @ jac)[None, :, :])
    regularized = gram + np.diag(.01 * np.maximum(gram.diagonal(), 1e-16))
    corrections = np.linalg.solve(regularized, projection.reshape(-1, 4).T).T.reshape(projection.shape)
    corrections = np.clip(np.clip(corrections, -.30, .30) + p[material_ids], 0., 1.) - p[material_ids]
    adjusted_score = score + np.sum((corrections @ gram - 2 * projection) * corrections, axis=-1)
    shortlist = []
    for scores, adjust, count in ((adjusted_score, True, 12), (score, False, 6)):
        for flat in np.argsort(scores, axis=None)[:count]:
            i, j = np.unravel_index(flat, scores.shape)
            trial = options[0][i].copy()
            trial[model.slices[1]] = options[1][j][model.slices[1]]
            if adjust:
                trial[material_ids] += corrections[i, j]
            shortlist.append(trial)
    return _verify(fit, model, p, shortlist)


def _translations(fit, model, p, coarse):
    offsets = [np.array(v) * scale for scale in (1., 2.5) for v in product((-1, 0, 1), repeat=3)]
    offsets.extend(direction * np.eye(3)[axis] * 5 for direction in (-1, 1) for axis in range(3))
    offsets = np.unique(offsets, axis=0)
    options = []
    for part in model.slices:
        batch = np.tile(p, (len(offsets), 1))
        batch[:, part][:, :3] += offsets
        options.append([model.constrain(x) for x in batch])
    return _paired_proposals(fit, model, p, options, coarse)


def _local_poses(fit, model, p, coarse, rng, scale):
    options = []
    for part, shape in zip(model.slices, model.shapes):
        batch = np.tile(p, (32, 1))
        dimension = len(p[part]) - 2
        size = np.full(dimension, scale)
        if shape.kind == "box":
            size[6:9] *= .32
        batch[1:, part][:, :dimension] += rng.normal(size=(len(batch)-1, dimension)) * size
        options.append([model.constrain(x) for x in batch])
    return _paired_proposals(fit, model, p, options, coarse)


def _fit_pair(fit, slots, iterations):
    full, coarse = tuple(range(len(fit.angles))), _coarse_beams(fit)
    lower, upper = fit.points[0, 0, 0] - .5, fit.points[-1, -1, -1] + .5
    primitives = [Shape(kind, lower, upper) for kind in ("sphere", "box")]
    starts = [[shape.start(mask, fit.points) for shape in primitives] for mask in slots]
    branches = []
    for a, b in product(range(2), repeat=2):
        model = Pair((primitives[a], primitives[b]))
        p = np.r_[starts[0][a], starts[1][b]]
        _, initial = _best(fit, model, [p], full)
        trial = _gauss_newton(fit, model, p, coarse, _material_steps(model, p, .15), .002)
        p, error = _verify(fit, model, p, [trial])
        branches.append((error, initial, model, p))

    # Keep both spherical and cuboidal alternatives for each learned slot.
    # First adjust relative positions, then spend the fine-pose budget on
    # the two best complete predictions.
    positioned = []
    for error, initial, model, p in sorted(branches, key=lambda item: item[0]):
        if error > .002:
            p, error = _translations(fit, model, p, coarse)
        positioned.append((error, initial, model, p))
        if error < 5e-4:
            break
    results = []
    for error, initial, model, p in sorted(positioned, key=lambda item: item[0])[:2]:
        for iteration in range(iterations):
            if error < 5e-4:
                break
            progress = iteration / max(1, iterations - 1)
            for part, shape in zip(model.slices, model.shapes):
                steps = np.zeros_like(p)
                steps[part] = 1.25 - .78 * progress
                steps[part][-2:] = .16 - .07 * progress
                if shape.kind == "box":
                    steps[part][6:9] = np.deg2rad(16. - 9. * progress)
                trial = _gauss_newton(fit, model, p, coarse, steps, .02)
                p, error = _verify(fit, model, p, [trial])
        # Coupled material derivatives address changes in one inclusion's
        # illumination due to the other. Select each update on the full RF.
        for step in (.14, .055, .015):
            if error < 5e-5:
                break
            p = _gauss_newton(fit, model, p, full, _material_steps(model, p, step), .0007)
            p, error = _best(fit, model, [p], full)
        results.append((error, initial, model, p))
        if error < 5e-4:
            break
    error, initial, model, p = min(results, key=lambda item: item[0])
    if error > .035:
        # The threshold and budgets are common to every input. Local covariance
        # proposals allow rotations, sizes and centers to change together.
        rng = np.random.default_rng(34913)
        for scale in (.85, .45, .24):
            p, error = _local_poses(fit, model, p, coarse, rng, scale)
            if error < .015:
                break
        for step in (.08, .025):
            p = _gauss_newton(fit, model, p, full, _material_steps(model, p, step), .001)
        p, error = _best(fit, model, [p], full)
    return (error, initial, model, p), min(x[1] for x in branches)


def refine_instances(scene, prediction, *, material="dielectric", device="cuda", dtype="float32",
                     iterations=3, batch_size=24):
    """Return voxel fields and diagnostics using the RF-predicted instance count."""
    scene, prediction = Path(scene), Path(prediction)
    if min(iterations, batch_size) < 1 or material not in MATERIALS or dtype not in ("float32", "float64"):
        raise ValueError("Invalid material, precision, iteration count or batch size")
    if prediction.resolve() in {scene.resolve() / "grid.npy", scene.resolve() / "scene.json"}:
        raise ValueError("Supply the RF-only instance prediction")
    start = perf_counter()
    acquisition = json.loads((scene / "acquisition.json").read_text())
    if acquisition.get("sensor_layout") != "antenna,direction,timestep":
        raise ValueError("Expected antenna,direction,timestep measured recordings")
    config, _, origin = _acquisition(acquisition)
    region = tuple(slice(int(x), n - config.pml_thickness) for x, n in zip(origin, config.grid_shape))
    slots, probability = _slots(prediction, config.grid_shape, region)
    if len(slots) == 1:
        with TemporaryDirectory() as scratch:
            single = Path(scratch) / "prediction.npy"
            np.save(single, slots[0])
            arrays, report = refine_scene(scene, single, material=material, device=device, dtype=dtype,
                                          iterations=iterations, batch_size=batch_size)
        arrays["instance_masks"] = arrays["occupied"][None]
        component_keys = ("shape", "center_cells", "radius_cells", "sides_cells", "rotation_matrix",
                          "permittivity", "conductivity_s_per_m")
        report["components"] = [{key: report[key] for key in component_keys if key in report}]
        report["components"][0]["occupied_cells"] = int(arrays["occupied"].sum())
        report["initial_best_relative_rf_l2"] = min(
            entry["initial"] for entry in report["candidate_relative_l2"].values())
    else:
        fit = JointFit(scene, material, device, dtype, batch_size)
        (error, chosen_initial, model, p), initial = _fit_pair(fit, slots, iterations)
        masks, values = fit.parts(model, p)
        occupied = masks[0] | masks[1]
        eps, sigma = np.ones(fit.shape, dtype="f4"), np.zeros(fit.shape, dtype="f4")
        components = []
        for mask, (er, conductivity), shape, part in zip(masks, values, model.shapes, model.slices):
            eps[mask], sigma[mask] = er, conductivity
            v = p[part]
            component = dict(shape=shape.kind, center_cells=v[:3].tolist(), permittivity=float(er),
                             conductivity_s_per_m=float(conductivity), occupied_cells=int(mask.sum()))
            if shape.kind == "sphere":
                component["radius_cells"] = float(v[3])
            else:
                component.update(sides_cells=(2 * v[3:6]).tolist(),
                                 rotation_matrix=Rotation.from_rotvec(v[6:9]).as_matrix().tolist())
            components.append(component)
        full = tuple(range(len(fit.angles)))
        echo = fit.evaluate(model, [p], full)[0].reshape(fit.target.shape)
        beam_norm = np.maximum(np.linalg.norm(fit.target, axis=(0, 2)), 1e-30)
        beam_error = np.linalg.norm(echo - fit.target, axis=(0, 2)) / beam_norm
        arrays = dict(sdf_cells=voxel_signed_distance(occupied).astype("f4"), occupied=occupied,
                      permittivity=eps, conductivity_s_per_m=sigma, instance_masks=np.stack(masks),
                      predicted_echo_v=echo, grid_spacing_m=config.spacing, grid_origin_cells=origin)
        report = dict(components=components, relative_rf_l2=error, initial_best_relative_rf_l2=initial,
                      selected_initial_relative_rf_l2=chosen_initial, relative_rf_l2_per_beam=beam_error.tolist(),
                      candidate_shots=fit.candidate_shots, forward_batches=fit.forward_batches, iterations=iterations,
                      sample_stride=config.sample_stride, solver_dtype=dtype, material_protocol=material,
                      mean_receiver_gain=float(fit.gain.mean()), empty_calibration_relative_l2=float(fit.empty_error))
    report.update(selected_instances=len(slots), probability_two=probability, elapsed_s=perf_counter() - start)
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("prediction", type=Path, help="NPZ written by EchoSet.py predict --instances")
    parser.add_argument("output", type=Path)
    parser.add_argument("--material", choices=MATERIALS, default="dielectric")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    protected = {args.prediction.resolve(), *(args.scene.resolve()/name for name in
                 ("grid.npy", "scene.json", "acquisition.json", "empty_sensor_samples.npy", "sensor_samples.npy"))}
    if args.output.resolve() in protected:
        parser.error("Choose a separate output for the refined instances")
    arrays, metadata = refine_instances(args.scene, args.prediction, material=args.material, device=args.device,
                                        dtype=args.dtype, iterations=args.iterations, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        path = Path(scratch) / "prediction.npz"
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, allow_nan=False))
        path.replace(args.output)
    print(json.dumps(metadata, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
