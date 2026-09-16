#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=weak-refinement-point
#SBATCH --output=weak-refinement-point_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --exclusive

# Run one refinement-based weak-scaling point of a single coarse element.
# P=1,8,64 selects L=5,6,7 respectively (32,768 elements per GPU).
# Allow 40 minutes for precompilation/startup plus roughly a minute of stepping.
# Submit with P set and --nodes=ceil(P/4), e.g.:
#   P=8 sbatch --nodes=2 --ntasks=8 weak-refinement-point.sh

set -euo pipefail

P="${P:?Submit with P set to the desired number of MPI ranks/GPUs}"
case "$P" in
    1) L=5 ;; 8) L=6 ;; 64) L=7 ;;
    *)
        echo "Unsupported P=${P}; expected one of: 1, 8, 64." >&2
        exit 1
        ;;
esac

NNODES=$(((P + 3) / 4))

if (( SLURM_JOB_NUM_NODES != NNODES )); then
    echo "Expected ${NNODES} allocated nodes for P=${P}, but received ${SLURM_JOB_NUM_NODES}." >&2
    exit 1
fi

echo 'Job is running on nodes:'
echo "$SLURM_JOB_NODELIST"
echo ------------------------------------------------------
echo "SLURM: submission node:        $SLURM_SUBMIT_HOST"
echo "SLURM: partition:              $SLURM_JOB_PARTITION"
echo "SLURM: submission directory:   $SLURM_SUBMIT_DIR"
echo "SLURM: job identifier:         $SLURM_JOBID"
echo "SLURM: job name:               $SLURM_JOB_NAME"
echo "SLURM: current home directory: $HOME"
echo "SLURM: PATH:                   $PATH"
echo ------------------------------------------------------

export JULIA_CUDA_MEMORY_POOL=none
# Enable CUDA-aware MPI with Delta's Cray MPICH.
export MPICH_GPU_SUPPORT_ENABLED=1
JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

echo "=== GPU Check ==="
nvidia-smi
echo ""

echo "=== Slurm Launcher Version ==="
srun --version
echo ""

echo "=== MPI Environment Variables ==="
printenv | grep -i mpi | sort || true
echo ""

echo "=== CUDA Environment ==="
printenv | grep -i cuda || true
echo ""

echo "Using JULIA: $JULIA"
echo "Job ID: $SLURM_JOBID"

# Instantiate and precompile on one node before launching the measured run.
srun --nodes=1 --ntasks=1 $JULIA --project=. -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
F=Float64

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/weak-scaling-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}_P${P}_sfc_packed_gpu-buffer"
mkdir -p "$RUN_DIR"
bash scripts/record_run_metadata.sh "$RUN_DIR" "$0"
prefix="${RUN_DIR}/weak-${SLURM_JOBID}-${F}-N${N}-L${L}-P${P}"

echo "Results will be saved to: ${RUN_DIR}"
echo "Running N=${N} L=${L} F=${F} with ${P} GPUs on ${NNODES} nodes..."
echo "Rank placement:"
srun --nodes="$NNODES" \
    --ntasks="$P" \
    --ntasks-per-node=4 \
    bash -c 'echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} node=$(hostname)"'

time srun --kill-on-bad-exit=1 --nodes="$NNODES" \
    --ntasks="$P" \
    --ntasks-per-node=4 \
    ./launch.sh $JULIA --project=. test/run/risingbubble.jl \
    --enable-single-buffer-comm \
    --number-of-steps 1000 \
    --polynomial-order "$N" \
    --refinement-level "$L" \
    --floating-point-type "$F" \
    --elements-in-direction-x 1 \
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1 \
    --timing-prefix "$prefix" \
    --enable-timing \
    2>&1 | tee -a "${prefix}.log"

echo "Completed ${P} GPU run"
ls -lh "${prefix}.log"
