#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=strong-point
#SBATCH --output=strong-point_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive

# Run one strong-scaling point for the fixed problem used in the paper. Submit
# with P set and with --nodes=ceil(P/4), for example:
#   P=8 sbatch --nodes=2 --ntasks=8 strong-point.sh

set -euo pipefail

P="${P:?Submit with P set to the desired number of MPI ranks/GPUs}"
case "$P" in
    1|4|8|12|16|32) ;;
    *)
        echo "Unsupported P=${P}; expected one of: 1, 4, 8, 12, 16, 32." >&2
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
L=6
F=Float64

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/strong-scaling-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}_P${P}_packed"
mkdir -p "$RUN_DIR"
prefix="${RUN_DIR}/strong-${SLURM_JOBID}-${F}-N${N}-L${L}-P${P}"
VTK_DIR="${RUN_DIR}/vtk"

echo "Results will be saved to: ${RUN_DIR}"
echo "Running N=${N} L=${L} F=${F} with ${P} GPUs on ${NNODES} nodes..."
echo "Rank placement:"
srun --nodes="$NNODES" \
    --ntasks="$P" \
    --ntasks-per-node=4 \
    bash -c 'echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} node=$(hostname)"'

# One power sampler per node; %N keeps each node's CSV separate.
srun --overlap --nodes="$NNODES" --ntasks="$NNODES" \
    --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 \
    --output="${prefix}-power-%N.csv" --error="${prefix}-power-%N.err" \
    nvidia-smi --query-gpu=timestamp,index,uuid,power.draw \
    --format=csv --loop-ms=250 &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true; wait "$SMI_PID" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

time srun --overlap --nodes="$NNODES" \
    --ntasks="$P" \
    --ntasks-per-node=4 \
    ./launch.sh $JULIA --project=. test/run/risingbubble.jl \
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
