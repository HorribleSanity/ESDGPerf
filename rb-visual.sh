#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=rb-visual
#SBATCH --output=rb-visual_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --exclusive

set -euo pipefail

# Runs the smooth and sharp rising thermal bubbles in Float64 and Float32.
# Writes per-step timing TSV files and VTK visualization output.

N=4
L=6
P=8
TIMEEND=1000

echo "=== Rising Bubble Visualization ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "N=${N} L=${L} P=${P} timeend=${TIMEEND}"
echo "======================================"

export JULIA_CUDA_MEMORY_POOL=none
# Enable CUDA-aware MPI with Delta's Cray MPICH.
export MPICH_GPU_SUPPORT_ENABLED=1

nvidia-smi
echo ""

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/rb-visual-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}_gpu-buffer"
mkdir -p ${RUN_DIR}
bash scripts/record_run_metadata.sh "$RUN_DIR" "$0"

echo "Results: ${RUN_DIR}"

for PROFILE in smooth sharp; do
  for F in Float64 Float32; do
    VTK_DIR="${RUN_DIR}/vtk-${PROFILE}-${F}"
    mkdir -p ${VTK_DIR}
    prefix="${RUN_DIR}/timing-${PROFILE}-${F}-N${N}-L${L}-P${P}"

    echo ""
    echo ">>> Running ${PROFILE} profile with ${F}, N=${N}, L=${L}, P=${P}, timeend=${TIMEEND} ..."

    time srun --kill-on-bad-exit=1 -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
      --enable-single-buffer-comm \
      --polynomial-order ${N} \
      --refinement-level ${L} \
      --floating-point-type ${F} \
      --bubble-profile ${PROFILE} \
      --time-end ${TIMEEND} \
      --elements-in-direction-x 1 \
      --elements-in-direction-y 1 \
      --elements-in-direction-z 1 \
      --enable-vtk-output \
      --vtk-output-directory "${VTK_DIR}" \
      --vtk-output-number-of-saves 10 \
      --enable-timing \
      --timing-prefix ${prefix} \
      2>&1 | tee -a ${RUN_DIR}/${PROFILE}-${F}.log

    echo "<<< Completed ${PROFILE} ${F}"
  done
done

echo ""
echo "All runs complete!"
ls -lh ${RUN_DIR}/
