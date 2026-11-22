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

The comparison reports independent error for each of the five raw and empty-subtracted waveforms against the pinned reference solver. The models repository provides pretrained RF-only reconstruction. The corresponding experiments document the lattice spacing, physical field of view, material families, run commands and accuracy. [Models](https://github.com/ja99/RFBasedNERFsModels) · [Research report](https://github.com/ja99/MasterThesisLatex/blob/main/diagnostics/benchmark_results.md)
