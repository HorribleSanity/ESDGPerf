#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=nsys-profile
#SBATCH --output=profile_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

# Collects a single-GPU Nsight Systems profile of the rising-bubble solver.

echo "=== Nsight Systems Profiling Run ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"

export JULIA_CUDA_MEMORY_POOL=none
export CUDA_VISIBLE_DEVICES=0

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

# Instantiate once
srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
L=3
STEPS=500

OUTDIR="${OUTPUT_ROOT}/profiles"
mkdir -p "${OUTDIR}"
PROFILE="${OUTDIR}/risingbubble_N${N}_L${L}_job${SLURM_JOBID}"

echo "Profiling rising bubble N=${N} L=${L} for ${STEPS} steps"
echo "Output: ${PROFILE}.nsys-rep"

srun -n 1 nsys profile \
  --trace=cuda,nvtx \
  --stats=true \
  --force-overwrite=true \
  -o ${PROFILE} \
  ${JULIA} --project=. test/run/risingbubble.jl \
  --polynomial-order ${N} \
  --refinement-level ${L} \
  --number-of-steps ${STEPS}

echo ""
echo "Profile complete!"
echo "View with: nsys stats ${PROFILE}.nsys-rep"
echo "Or open ${PROFILE}.nsys-rep in Nsight Systems GUI"
