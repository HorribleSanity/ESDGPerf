#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=imbalance-test
#SBATCH --output=imbalance-test_%j.txt
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive

set -euo pipefail

JULIA="julia +1.10"

srun -n 1 $JULIA --startup-file=no --project=. -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=1
L=6

for P in 1 2 4 8 16 32; do
    echo ""
    echo "=== Load imbalance: N=${N}, L=${L}, ranks=${P} ==="
    srun -n "$P" --ntasks-per-node=4 \
        $JULIA --startup-file=no --project=. \
        benchmarks/imbalance_test.jl -N "$N" -L "$L"
done
