#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=baro-dense
#SBATCH --output=baro-dense_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=8:00:00
#SBATCH --exclusive


# Reproduces the paper's fine baroclinic timing comparison:
# N=4, L=3, K=(36,6,3), one simulated day, a 0.5-second timestep,
# and 8 GPUs.
# The larger 41.5M-DOF mesh raises GPU utilization.

# Dense baroclinic run: 3x coarse elements in each direction vs original
# Matches rising bubble DOF ballpark (~41M vs 32M)
N=4
L=3
P=8
Kx=36
Ky=6
Kz=3
TIMEEND=86400  # 1 day (enough for timing comparison)
TIMESTEP=0.5

echo "=== Baroclinic Instability Dense Float Comparison ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "N=${N} L=${L} P=${P} Kx=${Kx} Ky=${Ky} Kz=${Kz} timeend=${TIMEEND} timestep=${TIMESTEP}"
echo "Resolution: $(( Kx * 2**L )) x $(( Ky * 2**L )) x $(( Kz * 2**L )) elements"
echo "DOF: $(( Kx * Ky * Kz * (2**L)**3 * (N+1)**3 ))"
echo "====================================================="

export JULIA_CUDA_MEMORY_POOL=none
#export MPICH_GPU_SUPPORT_ENABLED=1

nvidia-smi
echo ""

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/baro-compare-results/run_${TIMESTAMP}_job${SLURM_JOBID}_Kx${Kx}_Ky${Ky}_Kz${Kz}_L${L}_dense"
mkdir -p ${RUN_DIR}

echo "Results: ${RUN_DIR}"

for F in Float64 Float32; do
  VTK_DIR="${RUN_DIR}/vtk-${F}"
  mkdir -p ${VTK_DIR}
  prefix="${RUN_DIR}/timing-${F}-N${N}-L${L}-P${P}"

  echo ""
  echo ">>> Running ${F} N=${N} L=${L} Kx=${Kx} Ky=${Ky} Kz=${Kz} P=${P} timeend=${TIMEEND} ..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/baroclinicinstability.jl \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --time-end ${TIMEEND} \
    --time-step ${TIMESTEP} \
    --elements-in-direction-x ${Kx} \
    --elements-in-direction-y ${Ky} \
    --elements-in-direction-z ${Kz} \
    --enable-timing \
    --timing-prefix ${prefix} \
    --enable-triple-buffer-comm \
    2>&1 | tee -a ${RUN_DIR}/${F}.log

  echo "<<< Completed ${F}"
done

echo ""
echo "All runs complete!"
ls -lh ${RUN_DIR}/
