#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=quick-run
#SBATCH --output=quick-run_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive

set -euo pipefail

# Quick 100-step run to verify that the multi-GPU code executes.

echo "=== Quick Run ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "======================================"

export JULIA_CUDA_MEMORY_POOL=none
export MPICH_GPU_SUPPORT_ENABLED=1

nvidia-smi
echo ""

JULIA="julia +1.10"

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=3
L=6
F=Float64
P=4

echo "Running N=${N} L=${L} F=${F} with ${P} GPUs..."

time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
  --enable-single-buffer-comm \
  --number-of-steps 100 \
  --polynomial-order ${N} \
  --refinement-level ${L} \
  --floating-point-type ${F} \
  --elements-in-direction-x 1 \
  --elements-in-direction-y 1 \
  --elements-in-direction-z 1 \
  2>&1

echo ""
echo "Done!"
