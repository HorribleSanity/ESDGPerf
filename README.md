# ESDGPerf

This repository contains the code associated with the paper
[*GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver with Non-Conservative Terms*](https://arxiv.org/abs/2605.16684).

The code implements an entropy-stable discontinuous Galerkin solver for the
compressible Euler equations with gravity and includes the benchmark drivers
used for the performance results in the paper.

## Requirements

- Julia 1.10
- An NVIDIA GPU and a compatible CUDA driver
- MPI and p4est
- Slurm for the supplied multi-GPU batch scripts
- NVIDIA Nsight Compute or Nsight Systems for profiling runs

The batch scripts are configured for NVIDIA A100 nodes on NCSA Delta. Slurm
directives and system-specific Julia preferences will need to be changed on
other systems.

## Setup

Clone the repository and instantiate the Julia environment:

```sh
git clone https://github.com/HorribleSanity/ESDGPerf.git
cd ESDGPerf
julia +1.10 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

`LocalPreferences.toml` is not tracked because the MPI, CUDA, and p4est
configuration is system-specific.

## Quick test

On NCSA Delta, submit the four-GPU smoke test with:

```sh
sbatch quick-run.sh
```

This runs 100 time steps of the rising-thermal-bubble problem. Slurm output is
written to `quick-run_<jobid>.txt` in the submission directory.

## Reproducing the benchmarks

The principal batch scripts are:

| Result | Script |
| --- | --- |
| Optimization-step timings | `benchmarks/roofline/roofline_opt.sh` |
| Roofline measurements | `roofline.sh` |
| Strong scaling | `strong-point.sh` |
| Coarse-element weak scaling | `weak-coarse.sh` |
| Refinement-based weak scaling | `weak-refinement.sh` |
| Rising-bubble accuracy tests | `rb-visual.sh` |
| Baroclinic-instability tests | `baro-compare.sh`, `baro-compare-dense.sh` |
| Load-imbalance diagnostic | `imbalance-test.sh` |
| Nsight Systems profiling | `profile.sh`, `profile-scaling.sh` |

Large result files are written to
`/scratch/bfit/$USER/ESDGPerf` by default. Set `ESDG_OUTPUT_ROOT` to use a
different location.

## Repository structure

- `src/`: solver kernels and supporting implementation
- `test/run/`: rising-bubble and baroclinic-instability drivers
- `benchmarks/`: kernel benchmarks and load-imbalance diagnostics
- `misc/`: post-processing utilities

## Citation

If you use this code, please cite the associated paper linked above.
