# Calibrated RF phantom simulation

A scene consists of receiver voltages in SI units, a matched empty acquisition, array and timing metadata, and a voxel grid with relative permittivity, conductivity in S/m and signed distance in cells. The compact laboratory uses 16 transversely polarized elements and five individually reset steered shots. The metre-scale benchmark uses one broadside excitation of all 16 elements.

## Generate compact phantoms

Install the pinned FDTD fork and Python dependencies with `python -m pip install -r requirements.txt`. The batched solver requires a C++17 compiler with OpenMP for CPU execution or a CUDA-enabled PyTorch installation and `nvcc` for GPU execution. Its compiled kernels are cached in `BATCH_FDTD_CACHE` or the user's cache directory.

```bash
export RF_RUN=/workspace/rf-benchmark
export BATCH_FDTD_CACHE="$RF_RUN/compiled"
python PhantomBenchmark.py "$RF_RUN/train" --seed 81047 --scenes 768 --device cuda
python PhantomBenchmark.py "$RF_RUN/test" --seed 92371 --start-index 10000 --scenes 128 --device cuda
python ../RFBasedNERFsModels/SaveAsHDF5.py "$RF_RUN/train"
python ../RFBasedNERFsModels/SaveAsHDF5.py "$RF_RUN/test" --normalization-from "$RF_RUN/train"
```

Choose `--material thesis` for the highly conductive, finer-cell acquisition or `--objects 2` for separated inclusions. Generate and normalize a separate training archive for each acquisition geometry. The generator saves scene truth separately from measurement metadata and restores completed scenes when the generation manifest matches. Standard raw waveforms remain usable with `Acquisition.py` and the models' converter.

## Simulate larger grids

The batched FDTD solver stores the absorbing boundary's history only on the six faces where it is used. With a cubic grid of side `N` and PML thickness `p`, this takes `24 p N²` scalar values per simultaneous shot. The previous implementation allocated `12 N³` history values. Both allocate another `6 N³` values for the electric and magnetic fields. FP32 uses four bytes per value. Material coefficients are shared across a scene's beams.

For the compact phantom generator, `--batch-size 1 --max-parallel-beams 1` runs one scene and one beam at a time. The output still includes all five beams on the common sampling clock in their original order. In Python, pass `max_parallel_beams=1` to `BatchFDTD.record_batch` with your own `AcquisitionConfig` and material arrays. Increasing the limit allows more beams to run concurrently when memory permits. Use a fresh output directory for a new generator revision because the dataset manifest records the solver source hash.

For a configurable grid, choose `batch.cuda` or `batch.cpu` with `Acquisition.py`. For example, this command records one random scene at the default material-resolved spacing, together with its matched vacuum recording. The batched generator uses FP32 by default and accepts `--dtype float64` for precision studies.

```bash
python Acquisition.py ./rf-larger --backend batch.cuda --shape 128 216 216 --scenes 1 --max-parallel-beams 1
```

The benchmark below uses FP32, a four-cell PML, 16 receivers and 128 voltage samples. One scene contains a lossy dielectric box and the other is the matched vacuum reference. Both receive all five shots. CPU times on the Ryzen 7 5800H use four pinned OpenMP threads. CPU memory is peak process RSS. GPU memory is peak allocated PyTorch device memory on the RTX 3050 Laptop GPU. GPU allocator reserves and the CUDA context need additional device memory.

| Device and grid | Previous peak | Optimized peak | Previous recording time | Optimized recording time |
| --- | ---: | ---: | ---: | ---: |
| CPU, 64³, 382 steps | 527 MiB | 385 MiB | 10.43 s | 6.26 s |
| CPU, 80³, 509 steps | 734 MiB | 424 MiB | 19.20 s | 13.58 s |
| GPU, 64³, 382 steps | 185 MiB | 79 MiB | 0.64–0.67 s | 0.64–0.67 s |
| GPU, 184³, 890 steps | 4,277 MiB minimum¹ | 1,646 MiB | — | 34.22 s |
| GPU, 184³, one beam at a time | — | 407 MiB | — | 34.95 s |

¹ Calculated for the old fields and boundary history alone. This exceeds the installed GPU's memory. The 64³ CPU times are medians of three recordings, the 64³ GPU times are two warm recordings per revision, and each larger-grid configuration was timed once. The two 184³ runs gave bitwise-identical voltages. On each grid where both revisions ran, the saved FP32 voltages matched bit for bit on the same device. At 80³ on CPU, limiting execution to one beam per scene reduced peak process memory to 354 MiB and took 14.79 s.

With the default 128 × 216 × 216 grid (1.179 mm cells and a 15-cell PML), the same GPU recorded vacuum and a 14-cell-wide conductive cube (relative permittivity 54.16, conductivity 11.29 S/m). All five beams and 750 samples took 66.3 s and 460 MiB of peak GPU tensor memory with one beam per scene at a time. The box produced a finite echo on every beam, with peak amplitude ranging from 0.963 to 1.496 mV after vacuum subtraction. On the broadside beam, the FP32 echo differed from an FP64 run at the same 750 sample times by 0.0117% in relative L2 norm. The largest voltage error was 0.357 µV on an echo with a 1.148 mV peak. This timing covers waveform acquisition from material arrays. Scene voxelization and disk writing add time in the generator.

To reproduce a measurement from the simulator root, run once to compile the kernel and then repeat in a fresh process. Increasing the side at fixed spacing also increases the required round-trip recording window and the number of solver steps.

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python benchmarks/batch_memory.py --side 64 --repeat 3
PYTHONPATH=. python benchmarks/batch_memory.py --side 184 --device cuda --max-parallel-beams 1
PYTHONPATH=. python benchmarks/batch_memory.py --default-acquisition --device cuda --max-parallel-beams 1
```

## Pair RF scenes with a depth camera

For a new batch, pass `--depth-output "$RF_RUN/train-depth"` to either `PhantomBenchmark.py` or `Acquisition.py`. This writes a registered frame for each requested scene. For existing compact data, render the same frames on CPU with the standalone command.

```bash
python DepthCamera.py --scenes "$RF_RUN/train" --output "$RF_RUN/train-depth"
python DepthCamera.py --scenes "$RF_RUN/test" --output "$RF_RUN/test-depth"
```

The ideal pinhole camera sits at the mean receive-element position and looks along the positive world X axis. Each pixel returns the optical-axis distance in metres to the first occupied voxel face. All objects are optically opaque in this model. Pixel columns run toward negative world Y, and rows run toward negative world Z. The default image has 48 × 48 pixels, a 90° horizontal field of view and square pixels. Use `--width`, `--height` and `--hfov-deg` to set the optics.

Each scene produces a float32 `scene_*/depth.npy` with zero for a missed ray. The shared `camera.json` records the intrinsics, metric pose, crop and miss convention. An inference pipeline can load the image and calibration beside the RF recordings. The simulator reads voxel occupancy to render the visible surface. Object identities and RF material coefficients play no role in the image.

For ray-based reconstruction, `camera.observations(depth_m, tolerance_m=0)` returns two Boolean voxel grids for measured free space and first surfaces. A missed ray leaves the RF estimate unconstrained. A positive uncertainty half-width preserves ambiguous voxels at the observed surface. This inverse operation uses only the measured frame and camera calibration.

For a static scene, `camera.foreground_depth(measured_m, empty_room_m, tolerance_m=0)` keeps the measured depth of a new first surface. Capture both frames with the same camera pose and intrinsics. A positive tolerance in metres sets the minimum separation when both frames see a surface. Misses and unchanged background return zero.

## Single-shot metre-scale experiment

```bash
python MeterBenchmark.py "$RF_RUN/meter/training" --seed 75079 --scenes 1024
python MeterBenchmark.py "$RF_RUN/meter/sealed" --seed 91631 --start 10000 --scenes 128 --sealed
```

The useful volume is 125 × 130 × 130 cells at 8 mm spacing. Per-scene primitive parameters preserve full-resolution voxel labels without storing a dense material volume for every training scene. `signals.npz` contains signed raw receiver voltages, the common empty recording, calibration and explicit split membership.

## Verify the forward model

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
PYTHONPATH=. python tests/test_batch_fdtd.py --compare --device cuda --fixture benchmark
```

The comparison reports independent error for each of the five raw and empty-subtracted waveforms against the pinned reference solver. The models repository provides pretrained RF-only reconstruction. The corresponding experiments document the lattice spacing, physical field of view, material families, run commands and accuracy. [Models](https://github.com/ja99/RFBasedNERFsModels) · [Research report in the private thesis repository](https://github.com/ja99/MasterThesisLatex/blob/main/diagnostics/benchmark_results.md)
