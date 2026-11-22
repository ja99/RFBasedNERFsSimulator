"""Fit one homogeneous sphere or rotated cuboid to calibrated RF echoes.

Run `python RefinePhantom.py SCENE PREDICTED_SDF.npy OUTPUT.npz`. An NPZ with
the reconstructor's sdf_cells or occupied array is also accepted. Inference
uses the two voltage files and acquisition.json in SCENE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.spatial.transform import Rotation

from Acquisition import AcquisitionConfig
from BatchFDTD import batch_metadata, record_batch
from PhantomBenchmark import MATERIALS
from SDF import voxel_signed_distance


def _acquisition(metadata):
    positions = np.asarray(metadata["antenna_positions_cells"])
    n = int(np.sqrt(len(positions)))
    if n < 2 or n * n != len(positions) or metadata["voltage_units"] != "V":
        raise ValueError("The waveform fit needs a calibrated square array with voltages in SI units")
    angles = tuple(map(tuple, metadata["beamforming_angles_deg"]))
    origin = np.asarray(metadata["grid_origin_cells"], dtype=int)
    pml = metadata["pml_thickness"]
    geometry = dict(grid_shape=tuple(metadata["grid_shape"]), pml_thickness=pml,
                    grid_spacing_m=metadata["grid_spacing_m"], frequency_hz=metadata["frequency_hz"],
                    elements_per_side=n, standoff_cells=int(origin[0] - pml),
                    samples_per_shot=metadata["samples_per_shot"], sample_stride=metadata["sample_stride"],
                    source_amplitude_sim=metadata["source_amplitude_sim"])
    # An even array can be rounded to cells from a fractional physical pitch.
    # Try both its inferred discretized pitch and the standard half wavelength.
    for pitch in (float(np.ptp(positions[:, 1]) / (n - 1)), None):
        config = AcquisitionConfig(**geometry, pitch_cells=pitch)
        try:
            modeled = batch_metadata(config, angles)
        except ValueError:
            continue
        if (np.array_equal(modeled["grid_origin_cells"], origin)
                and np.array_equal(modeled["antenna_positions_cells"], positions)
                and np.array_equal(modeled["transmit_positions_cells"], metadata["transmit_positions_cells"])
                and np.array_equal(modeled["element_polarization_local"], metadata["element_polarization_local"])
                and np.isclose(modeled["time_step_s"], metadata["time_step_s"], atol=0, rtol=1e-12)):
            return config, angles, origin
    raise ValueError("The recorded transmitters, receivers or sample clock cannot be reproduced")


def _read_initial(path, region, full_shape):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            if "sdf_cells" in data:
                values = data["sdf_cells"]
            elif "occupied" in data:
                values = data["occupied"].astype(bool)
            else:
                raise ValueError("The predicted NPZ needs sdf_cells or occupied")
    else:
        values = data
    if values.shape == full_shape:
        values = values[region]
    shape = tuple(s.stop - s.start for s in region)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite predicted SDF or binary mask with spatial shape {shape}")
    binary = values.dtype == bool or np.all((values == 0) | (values == 1))
    mask = values.astype(bool) if binary else (values < 0)
    if not mask.any() or mask.all():
        if binary or np.ptp(values) == 0:
            raise ValueError("The initial prediction contains no localized inclusion")
        mask = values <= np.quantile(values, .005)
    components, count = label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    return components == (1 + sizes[1:].argmax())


@dataclass
class Shape:
    kind: str
    lower: np.ndarray
    upper: np.ndarray

    def constrain(self, values):
        p = np.array(values, dtype=float, copy=True)
        span = self.upper - self.lower
        p[-2:] = np.clip(p[-2:], 0., 1.)
        if self.kind == "sphere":
            p[3] = np.clip(p[3], .8, span.min() / 2)
            extent = p[3]
        else:
            p[3:6] = np.maximum(p[3:6], .8)
            extent = np.abs(Rotation.from_rotvec(p[6:9]).as_matrix()) @ p[3:6]
            scale = min(1., np.min(span / (2 * extent)))
            p[3:6] *= scale
            extent *= scale
        p[:3] = np.clip(p[:3], self.lower + extent, self.upper - extent)
        return p

    def distance(self, p, points):
        local = points - p[:3]
        if self.kind == "sphere":
            return np.linalg.norm(local, axis=-1) - p[3]
        delta = np.abs(local @ Rotation.from_rotvec(p[6:9]).as_matrix()) - p[3:6]
        return np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(axis=-1), 0)

    def start(self, mask, points):
        cloud = points[mask]
        center = cloud.mean(axis=0)
        if self.kind == "sphere":
            geometry = np.r_[center, np.cbrt(3 * len(cloud) / (4 * np.pi))]
        else:
            centered = cloud - center
            variance, axes = np.linalg.eigh(centered.T @ centered / len(cloud))
            axes = axes[:, ::-1].copy()
            axes[:, -1] *= np.linalg.det(axes)
            half = np.sqrt(3 * np.maximum(variance[::-1], 0) + .25)
            # Axis permutations/signs describe the same box. A representation
            # close to identity keeps the rotation-vector Jacobian well scaled.
            symmetries = Rotation.create_group("O").as_matrix()
            equivalent = axes @ symmetries
            i = np.trace(equivalent, axis1=1, axis2=2).argmax()
            half = np.abs(symmetries[i]).T @ half
            geometry = np.r_[center, half, Rotation.from_matrix(equivalent[i]).as_rotvec()]
        # Inner and outer surface samples from the learned occupancy should
        # fall on the corresponding sides of the fitted primitive.
        band = binary_dilation(mask) ^ binary_erosion(mask)
        points_near = points[band]
        sign = np.where(mask[band], 1., -1.)
        def violations(g):
            return np.maximum(sign * self.distance(g, points_near) + .025, 0)
        fit = least_squares(violations, geometry, loss="soft_l1", f_scale=.3, max_nfev=45)
        return self.constrain(np.r_[fit.x, .5, .5])


@lru_cache(maxsize=8)
def _vacuum(config, angles, device, dtype):
    eps = np.ones((1, *config.grid_shape), dtype=dtype)
    return record_batch(config, eps, np.zeros_like(eps), angles=angles, device=device, dtype=dtype)[0].astype(float)


class EchoFit:
    def __init__(self, scene, material, device, dtype, batch_size):
        with (scene / "acquisition.json").open() as stream:
            metadata = json.load(stream)
        self.config, self.angles, origin = _acquisition(metadata)
        upper = np.asarray(self.config.grid_shape) - self.config.pml_thickness
        self.region = tuple(slice(int(a), int(b)) for a, b in zip(origin, upper))
        self.shape = tuple(upper - origin)
        self.points = np.moveaxis(np.indices(self.shape, dtype=float), 0, -1) + origin
        raw = np.load(scene / "sensor_samples.npy", allow_pickle=False).astype(float)
        empty = np.load(scene / "empty_sensor_samples.npy", allow_pickle=False).astype(float)
        required = (self.config.elements_per_side ** 2, len(self.angles), self.config.samples_per_shot)
        if raw.shape != required or empty.shape != required or not np.isfinite(raw).all() or not np.isfinite(empty).all():
            raise ValueError(f"Both raw recordings must be finite with (antenna,beam,time) shape {required}")
        if not np.any(empty):
            raise ValueError("An empty reference with zero receive power cannot calibrate the RF forward model")
        self.target = raw - empty
        if not np.any(self.target):
            raise ValueError("An empty differential recording cannot identify a single inclusion")
        self.empty = _vacuum(self.config, self.angles, device, dtype)
        power = np.sum(self.empty ** 2, axis=-1, keepdims=True)
        self.gain = np.divide(np.sum(self.empty * empty, axis=-1, keepdims=True), power,
                              out=np.ones_like(power), where=power > 0)
        self.empty_error = np.linalg.norm(self.empty * self.gain - empty) / max(np.linalg.norm(empty), 1e-30)
        self.bounds = np.array(MATERIALS[material][:2])
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        self.cache = {}
        self.candidate_shots = self.forward_batches = 0

    def materials(self, p):
        return self.bounds[:, 0] + p[-2:] * np.ptp(self.bounds, axis=1)

    def mask(self, model, p):
        local = self.points - p[:3]
        if model.kind == "sphere":
            return np.einsum("...i,...i->...", local, local) < (p[3] - 1e-12) ** 2
        local = local @ Rotation.from_rotvec(p[6:9]).as_matrix()
        return np.all(np.abs(local) < p[3:6] - 1e-12, axis=-1)

    def evaluate(self, model, candidates, beams):
        return self.evaluate_masks(((self.mask(model, p), *self.materials(p)) for p in candidates), beams)

    def evaluate_masks(self, phantoms, beams):
        keys, missing = [], {}
        for mask, eps, sigma in phantoms:
            eps, sigma = np.asarray((eps, sigma), dtype=self.dtype)
            key = (beams, eps, sigma, np.packbits(mask).tobytes())
            keys.append(key)
            if key not in self.cache:
                missing[key] = (mask, eps, sigma)
        jobs = list(missing)
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start:start + self.batch_size]
            eps = np.ones((len(chunk), *self.config.grid_shape), dtype=self.dtype)
            sig = np.zeros_like(eps)
            for j, key in enumerate(chunk):
                mask, er, conductivity = missing[key]
                eps[(j, *self.region)][mask] = er
                sig[(j, *self.region)][mask] = conductivity
            raw = record_batch(self.config, eps, sig, angles=tuple(self.angles[i] for i in beams),
                               device=self.device, dtype=self.dtype).astype(float)
            simulated = (raw - self.empty[:, beams, :]) * self.gain[:, beams, :]
            for key, echo in zip(chunk, simulated):
                self.cache[key] = echo.ravel()
            self.candidate_shots += len(chunk) * len(beams)
            self.forward_batches += 1
        return np.stack([self.cache[key] for key in keys])

    def observed(self, beams):
        return self.target[:, beams, :].ravel()


def _best(fit, model, candidates, beams):
    candidates = [model.constrain(p) for p in candidates]
    observed = fit.observed(beams)
    error = np.linalg.norm(fit.evaluate(model, candidates, beams) - observed, axis=1)
    i = error.argmin()
    return candidates[i], float(error[i] / max(np.linalg.norm(observed), 1e-30))


def _gauss_newton(fit, model, p, beams, steps, damping):
    active = np.flatnonzero(steps)
    perturb = np.eye(len(p))[active] * steps
    probes = [p, *(p + dp for dp in perturb), *(p - dp for dp in perturb)]
    probes = [model.constrain(candidate) for candidate in probes]
    values = fit.evaluate(model, probes, beams)
    jacobian = .5 * (values[1:1 + len(active)] - values[1 + len(active):]).T
    norms = np.linalg.norm(jacobian, axis=0)
    norms = np.maximum(norms, 1e-12 * max(np.linalg.norm(fit.observed(beams)), 1e-30))
    unit = jacobian / norms
    delta = np.linalg.solve(unit.T @ unit + damping * np.eye(len(active)), unit.T @ (fit.observed(beams) - values[0]))
    move = np.zeros_like(p)
    move[active] = steps[active] * np.clip(delta / norms, -2.5, 2.5)
    options = probes + [p + fraction * move for fraction in (1., .55, .25)]
    return _best(fit, model, options, beams)[0]


def _coarse_beams(fit):
    broadside = int(np.argmin(np.sum(np.asarray(fit.angles) ** 2, axis=1)))
    energies = np.sum(fit.target ** 2, axis=(0, 2))
    other = [int(i) for i in np.argsort(-energies) if i != broadside][:1]
    return (broadside, *other)


def _population(fit, model, p, error):
    """Screen subcell pose proposals on two beams, verify finalists on every beam."""
    coarse = _coarse_beams(fit)
    full = tuple(range(len(fit.angles)))
    observed = fit.observed(coarse)
    rng = np.random.default_rng(637901 if model.kind == "sphere" else 637903)
    geom = len(p) - 2
    # A binary voxel boundary can have a flat or misleading two-sided difference.
    # Larger residuals get a second, wider search, including new cuboid orientations.
    wide = error > .12
    rounds = 6 if wide else 4
    population = 56 if wide else 32
    geom_scale = np.full(geom, 1.1 if wide else .55)
    geom_scale[:3] *= 1.35
    if model.kind == "box":
        geom_scale[6:9] = .4 if wide else .23
    floor = np.full(geom, .07)
    if model.kind == "box":
        floor[6:9] = .035
    covariance = np.diag(geom_scale ** 2)
    for generation in range(rounds):
        offset = rng.multivariate_normal(np.zeros(geom), covariance, size=population)
        proposals = np.broadcast_to(p, (population + 1, len(p))).copy()
        proposals[1:, :geom] += offset
        # Give quantized surfaces an opportunity to cross multiple neighbouring
        # voxel centers at once, including in an almost converged fit.
        for i, step in enumerate((1., .5, .25), start=1):
            proposals[i] = p
            proposals[i, :3] = np.round(p[:3] / step) * step
        if wide and generation == 0 and model.kind == "box":
            count = population // 4
            proposals[-count:, 6:9] = Rotation.random(count, random_state=rng).as_rotvec()
        proposals = np.stack([model.constrain(v) for v in proposals])

        # A reusable material Jacobian estimates how much of each coarse
        # residual can be removed by refitting epsilon/sigma. Every shortlisted
        # adjustment is then checked by an actual five-shot forward solve.
        lower, upper = [], []
        for j in (-2, -1):
            low, high = p.copy(), p.copy()
            low[j], high[j] = max(0., p[j] - .10), min(1., p[j] + .10)
            lower.append(low)
            upper.append(high)
        waves = fit.evaluate(model, [*lower, *upper], coarse)
        widths = np.array([upper[i][-2 + i] - lower[i][-2 + i] for i in range(2)])
        jac = (waves[2:] - waves[:2]).T / widths
        norms = np.maximum(np.linalg.norm(jac, axis=0), 1e-12 * np.linalg.norm(observed))
        unit = jac / norms
        echoes = fit.evaluate(model, proposals, coarse)
        residual = observed - echoes
        correction = np.linalg.solve(unit.T @ unit + .01 * np.eye(2), unit.T @ residual.T).T / norms
        adjusted = np.clip(proposals[:, -2:] + np.clip(correction, -.30, .30), 0., 1.)
        predicted_error = np.linalg.norm(residual - (adjusted - proposals[:, -2:]) @ jac.T, axis=1)
        rank = np.argsort(predicted_error)
        shortlist = proposals[rank[:8]].copy()
        shortlist[:, -2:] = adjusted[rank[:8]]
        best_raw = proposals[np.argmin(np.linalg.norm(residual, axis=1))]
        p, error = _best(fit, model, [p, best_raw, *shortlist], full)
        if error < 5e-4:
            break
        # Adapt the principal directions from local elites, with a diagonal
        # floor so an initially weak coordinate can recover on subsequent shots.
        diffs = proposals[rank[:12], :geom] - p[:geom]
        widths = np.maximum(geom_scale * .69 ** (generation + 1), floor)
        scaled = np.clip(diffs / np.maximum(widths, floor), -2., 2.)
        covariance = widths[:, None] * (.5 * np.eye(geom) + .5 * scaled.T @ scaled / len(scaled)) * widths[None, :]
    return p, error


def _surface_step(fit, model, p, seed):
    """Approximate local shape derivatives by flipping individual boundary voxels."""
    beams = _coarse_beams(fit)
    full = tuple(range(len(fit.angles)))
    mask = fit.mask(model, p)
    distance = np.abs(model.distance(p, fit.points))
    seed_distance = np.abs(model.distance(seed, fit.points))
    proximity = np.minimum(distance, seed_distance)
    cells = np.argwhere(proximity < 1.35)
    if len(cells) > 248:
        rng = np.random.default_rng(5561)
        weights = 1 / (.25 + proximity[tuple(cells.T)])
        cells = cells[rng.choice(len(cells), 248, replace=False, p=weights / weights.sum())]
    eps, sigma = fit.materials(p)
    base = fit.evaluate(model, [p], beams)[0]
    derivatives = []
    for start in range(0, len(cells), fit.batch_size):
        indices = cells[start:start + fit.batch_size]
        toggles = np.repeat(mask[None], len(indices), axis=0)
        toggles[(np.arange(len(indices)), *indices.T)] ^= True
        waves = fit.evaluate_masks(((m, eps, sigma) for m in toggles), beams)
        sign = np.where(mask[tuple(indices.T)], -1., 1.)
        derivatives.extend((waves - base) * sign[:, None])
    for j in (-2, -1):
        low, high = p.copy(), p.copy()
        low[j], high[j] = max(0., p[j] - .07), min(1., p[j] + .07)
        first, last = fit.evaluate(model, [low, high], beams)
        derivatives.append((last - first) / (high[j] - low[j]))
    # Compress the waveform-space least-squares problem to the span of these
    # finite-contrast derivatives. The omitted orthogonal residual is constant.
    observed = fit.observed(beams)
    norm = np.linalg.norm(observed)
    basis = np.asarray(derivatives) / norm
    vals, vecs = np.linalg.eigh(basis @ basis.T)
    keep = vals > max(vals.max() * 1e-10, 1e-25)
    if not keep.any():
        return _best(fit, model, [p], full)
    vals, vecs = np.sqrt(vals[keep]), vecs[:, keep].T
    transform = vals[:, None] * vecs
    target = (vecs @ (basis @ ((observed - base) / norm))) / vals
    points, inside = fit.points[tuple(cells.T)], mask[tuple(cells.T)].astype(float)
    options = [p]
    for start in (p, seed):
        trial = start.copy()
        trial[-2:] = p[-2:]
        limits = np.ones_like(trial)
        if model.kind == "box":
            limits[6:9] = .4
        for temperature in (.34, .14, .055):
            def residual(v):
                pose = model.constrain(v)
                occupancy = expit(-model.distance(pose, points) / temperature)
                change = np.r_[occupancy - inside, pose[-2:] - p[-2:]]
                return np.r_[transform @ change - target, .008 * (v - start) / limits]
            trial = model.constrain(least_squares(residual, trial, x_scale=limits, max_nfev=38).x)
            options.append(trial)
    return _best(fit, model, options, full)


def _search(fit, model, p, iterations):
    all_beams = tuple(range(len(fit.angles)))
    coarse = _coarse_beams(fit)
    _, initial = _best(fit, model, [p], all_beams)
    material_only = np.zeros_like(p)
    material_only[-2:] = .16
    p = _gauss_newton(fit, model, p, coarse, material_only, .015)
    offsets = np.eye(len(p))[:3] * 1.5
    options = [p, *(p + delta for delta in offsets), *(p - delta for delta in offsets)]
    geometry = slice(3, 4 if model.kind == "sphere" else 6)
    for scale in (.85, 1.15):
        trial = p.copy()
        trial[geometry] *= scale
        options.append(trial)
    if model.kind == "box":
        trial = p.copy()
        trial[6:9] = 0
        options.append(trial)
    p, _ = _best(fit, model, options, coarse)
    for iteration in range(iterations):
        progress = iteration / max(1, iterations - 1)
        beams = coarse if iteration < iterations - 2 else all_beams
        steps = np.full_like(p, 1.1 - .75 * progress)
        if model.kind == "box":
            steps[6:9] = np.deg2rad(16 - 10 * progress)
        steps[-2:] = .16 - .10 * progress
        p = _gauss_newton(fit, model, p, beams, steps, .03)
    for step in (.16, .06, .02):
        material_only[-2:] = step
        p = _gauss_newton(fit, model, p, all_beams, material_only, .001)
    p, error = _best(fit, model, [p], all_beams)
    return p, initial, error


def refine_scene(scene, prediction, *, material="dielectric", device="cuda", dtype="float32",
                 iterations=5, batch_size=24):
    """Return prediction arrays and JSON-compatible diagnostics from raw paired voltages."""
    if iterations < 1 or batch_size < 1 or material not in MATERIALS or dtype not in ("float32", "float64"):
        raise ValueError("Invalid material protocol, solver precision, iteration count or batch size")
    start = perf_counter()
    scene, prediction = Path(scene), Path(prediction)
    if prediction.resolve() in {scene.resolve() / "grid.npy", scene.resolve() / "scene.json"}:
        raise ValueError("Supply an RF-only predicted field to initialize the refinement")
    fit = EchoFit(scene, material, device, dtype, batch_size)
    seed = _read_initial(prediction, fit.region, fit.config.grid_shape)
    candidates = []
    for kind in ("sphere", "box"):
        model = Shape(kind, fit.points[0, 0, 0] - .5, fit.points[-1, -1, -1] + .5)
        p, initial, error = _search(fit, model, model.start(seed, fit.points), iterations)
        candidates.append((error, initial, model, p))
    candidates.sort(key=lambda candidate: candidate[0])
    full = tuple(range(len(fit.angles)))
    best_verified = candidates[0][0]
    for i, (error, initial, model, p) in enumerate(candidates):
        if error < 5e-4 or (i and best_verified < .015 and error > 2.5 * best_verified):
            continue
        p, error = _population(fit, model, p, error)
        if error > .10:
            # A local RF minimum can move away from an otherwise informative
            # learned seed. Restart the subcell search around the seed pose
            # using the calibrated material obtained on the first branch.
            restart = model.start(seed, fit.points)
            restart[-2:] = p[-2:]
            alternative, alternative_error = _population(fit, model, restart, error)
            if alternative_error < error:
                p, error = alternative, alternative_error
        if error > .035:
            restart = model.start(seed, fit.points)
            for _ in range(2):
                trial, trial_error = _surface_step(fit, model, p, restart)
                improvement = error - trial_error
                p, error = trial, trial_error
                if error < .015 or improvement < .006:
                    break
        material_only = np.zeros_like(p)
        for step in (.10, .035, .012):
            material_only[-2:] = step
            p = _gauss_newton(fit, model, p, full, material_only, .001)
        p, error = _best(fit, model, [p], full)
        best_verified = min(best_verified, error)
        candidates[i] = (error, initial, model, p)
    error, _, model, p = min(candidates, key=lambda item: item[0])
    occupied = fit.mask(model, p)
    eps, sigma = fit.materials(p)
    all_beams = tuple(range(len(fit.angles)))
    echo = fit.evaluate(model, [p], all_beams)[0].reshape(fit.target.shape)
    target_norm = np.linalg.norm(fit.target, axis=(0, 2))
    beam_error = np.linalg.norm(echo - fit.target, axis=(0, 2)) / np.maximum(target_norm, 1e-30)
    derivatives = []
    for j in (-2, -1):
        low, high = p.copy(), p.copy()
        low[j], high[j] = max(0., p[j] - .025), min(1., p[j] + .025)
        lower, upper = fit.evaluate(model, [low, high], all_beams)
        derivatives.append((upper - lower) / (high[j] - low[j]) / np.linalg.norm(fit.target))
    singular = np.linalg.svd(derivatives, compute_uv=False)
    report = dict(shape=model.kind, center_cells=p[:3].tolist(), relative_rf_l2=error,
                  relative_rf_l2_per_beam=beam_error.tolist(), permittivity=float(eps),
                  conductivity_s_per_m=float(sigma), empty_calibration_relative_l2=float(fit.empty_error),
                  mean_receiver_gain=float(fit.gain.mean()), material_protocol=material,
                  candidate_relative_l2={m.kind: {"initial": i, "final": e} for e, i, m, _ in candidates},
                  fixed_geometry_material_jacobian_singular_values=singular.tolist(),
                  candidate_shots=fit.candidate_shots, forward_batches=fit.forward_batches,
                  iterations=iterations, sample_stride=fit.config.sample_stride,
                  solver_dtype=dtype, elapsed_s=perf_counter() - start)
    if model.kind == "sphere":
        report["radius_cells"] = float(p[3])
    else:
        report["sides_cells"] = (2 * p[3:6]).tolist()
        report["rotation_matrix"] = Rotation.from_rotvec(p[6:9]).as_matrix().tolist()
    arrays = dict(sdf_cells=voxel_signed_distance(occupied).astype(np.float32), occupied=occupied,
                  permittivity=np.where(occupied, eps, 1.).astype(np.float32),
                  conductivity_s_per_m=np.where(occupied, sigma, 0.).astype(np.float32), predicted_echo_v=echo,
                  grid_spacing_m=fit.config.spacing, grid_origin_cells=np.array([s.start for s in fit.region]))
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--material", choices=MATERIALS, default="dielectric")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    scene_root = args.scene.resolve()
    protected = {args.prediction.resolve(), *(scene_root / name for name in
                 ("acquisition.json", "sensor_samples.npy", "empty_sensor_samples.npy", "scene.json", "grid.npy"))}
    if args.output.resolve() in protected:
        parser.error("Choose a separate output path for the refined prediction")
    arrays, report = refine_scene(args.scene, args.prediction, material=args.material, device=args.device,
                                  dtype=args.dtype, iterations=args.iterations, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        temporary = Path(scratch) / "prediction.npz"
        np.savez_compressed(temporary, **arrays, metadata=json.dumps(report, allow_nan=False))
        temporary.replace(args.output)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
