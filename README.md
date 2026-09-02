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
git clone git@github.com:HorribleSanity/ESDGPerf.git
cd ESDGPerf
julia +1.10 --project=. -e 'using Pkg; Pkg.instantiate()'
```

The command `julia +1.10` uses
[Juliaup](https://github.com/JuliaLang/juliaup) notation.

`LocalPreferences.toml` is not tracked because the MPI, CUDA, and p4est
configuration is system-specific.

### NCSA Delta configuration

Configure CUDA.jl to use Delta's system CUDA toolkit:

```sh
julia +1.10 --project=. -e \
  'using CUDA; CUDA.set_runtime_version!(v"13.2", local_toolkit=true)'
```

Configure MPI.jl to use Delta's system Cray MPICH library and the Slurm
launcher:

```sh
julia +1.10 --project=. -e \
  'using MPIPreferences; MPIPreferences.use_system_binary(
      library_names=["libmpi_gnu_112"],
      abi="MPICH",
      mpiexec="srun",
  )'
```

Build [p4est](https://github.com/cburstedde/p4est) against the same system MPI
library, following the
[P4est.jl instructions for custom MPI installations](https://github.com/trixi-framework/P4est.jl#using-a-custom-version-of-mpi-andor-p4est):

```sh
cd p4est-2.8.6
mkdir build
cd build

../configure CC=mpicc CFLAGS="-O2 -Wall -Wno-unused-parameter" \
  --enable-mpi \
  --enable-shared \
  --prefix="$HOME/local/p4est"

make -j4
make install
```

Configure P4est.jl to use the resulting installation:

```sh
julia +1.10 --project=. -e \
  'using P4est
   P4est.set_library_p4est!("/path/to/libp4est.so")'
```

These commands write machine-specific settings to `LocalPreferences.toml`.
Restart Julia after changing any setting. See the
[MPI.jl configuration documentation](https://juliaparallel.org/MPI.jl/stable/configuration/)
for additional MPI options.

After configuring the machine-specific preferences, precompile the environment
in a new Julia process:

```sh
julia +1.10 --project=. -e 'using Pkg; Pkg.precompile()'
```

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
