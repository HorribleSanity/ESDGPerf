#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=baro-compare
#SBATCH --output=baro-compare_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=8:00:00
#SBATCH --exclusive

set -euo pipefail


# Reproduces the paper's coarse baroclinic-instability comparison:
# N=4, L=3, K=(12,2,1), 12 simulated days, 8 GPUs,
# with a 2.0-second timestep, Float64 and Float32 timing, and VTK output.

# Same mesh as Feb 23 run (job16159219) that produced the day12 plot
N=4
L=3
P=8
Kx=12
Ky=2
Kz=1
TIMEEND=1036800  # 12 days
TIMESTEP=2.0

echo "=== Baroclinic Instability Float Comparison ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "N=${N} L=${L} P=${P} Kx=${Kx} Ky=${Ky} Kz=${Kz} timeend=${TIMEEND} timestep=${TIMESTEP}"
echo "Resolution: $(( Kx * 2**L )) x $(( Ky * 2**L )) x $(( Kz * 2**L )) elements"
echo "VTK output every day (hardcoded in Julia script)"
echo "================================================"

export JULIA_CUDA_MEMORY_POOL=none
# Enable CUDA-aware MPI with Delta's Cray MPICH.
export MPICH_GPU_SUPPORT_ENABLED=1

nvidia-smi
echo ""

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/baro-compare-results/run_${TIMESTAMP}_job${SLURM_JOBID}_Kx${Kx}_Ky${Ky}_Kz${Kz}_L${L}_gpu-buffer"
mkdir -p ${RUN_DIR}
bash scripts/record_run_metadata.sh "$RUN_DIR" "$0"

echo "Results: ${RUN_DIR}"

for F in Float64 Float32; do
  VTK_DIR="${RUN_DIR}/vtk-${F}"
  mkdir -p ${VTK_DIR}
  prefix="${RUN_DIR}/timing-${F}-N${N}-L${L}-P${P}"

  echo ""
  echo ">>> Running ${F} N=${N} L=${L} Kx=${Kx} Ky=${Ky} Kz=${Kz} P=${P} timeend=${TIMEEND} ..."

  time srun --kill-on-bad-exit=1 -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/baroclinicinstability.jl \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --time-end ${TIMEEND} \
    --time-step ${TIMESTEP} \
    --elements-in-direction-x ${Kx} \
    --elements-in-direction-y ${Ky} \
    --elements-in-direction-z ${Kz} \
    --enable-vtk-output \
    --vtk-output-directory "${VTK_DIR}" \
    --enable-timing \
    --timing-prefix ${prefix} \
    --enable-single-buffer-comm \
    2>&1 | tee -a ${RUN_DIR}/${F}.log

  echo "<<< Completed ${F}"
done

echo ""
echo "All runs complete!"
ls -lh ${RUN_DIR}/
