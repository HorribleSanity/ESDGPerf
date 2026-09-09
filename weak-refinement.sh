#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=weak-refinement
#SBATCH --output=weak-refinement_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive

# Refinement-based weak scaling (refining a single coarse element)

echo 'Job is running on nodes:'
echo $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: submission node:        $SLURM_SUBMIT_HOST
echo SLURM: partition:              $SLURM_JOB_PARTITION
echo SLURM: submission directory:   $SLURM_SUBMIT_DIR
echo SLURM: job identifier:         $SLURM_JOBID
echo SLURM: job name:               $SLURM_JOB_NAME
echo SLURM: current home directory: $HOME
echo SLURM: PATH:                   $PATH
echo ------------------------------------------------------

export JULIA_CUDA_MEMORY_POOL=none
#export MPICH_GPU_SUPPORT_ENABLED=1

echo "=== GPU Check ==="
nvidia-smi
echo ""

echo "=== Slurm Launcher Version ==="
srun --version
echo ""

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

echo "Using JULIA: $JULIA"
echo "Job ID: $SLURM_JOBID"

# Instantiate once
srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
F=Float64

# Create unique directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/weak-scaling-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_sfc"
mkdir -p ${RUN_DIR}
bash scripts/record_run_metadata.sh "$RUN_DIR" "$0"

echo "Results will be saved to: ${RUN_DIR}"

# Weak scaling: Kx=Ky=Kz=1, increase L.
# Each refinement level multiplies elements by 8, so GPUs scale by 8x.
# Single octree forces non-trivial SFC partitioning.
# L=5: 32^3 = 32,768 elements on 1 GPU  (32,768/GPU)
# L=6: 64^3 = 262,144 elements on 8 GPUs (32,768/GPU)
# L=7: 128^3 = 2,097,152 elements on 64 GPUs (32,768/GPU)
for L in 5 6 7; do
  P=$(( 8 ** (L - 5) ))
  prefix="${RUN_DIR}/weak-${SLURM_JOBID}-${F}-N${N}-L${L}-P${P}"

  echo "Running N=${N} L=${L} F=${F} with ${P} GPUs (32,768 elements/GPU)..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
    --enable-triple-buffer-comm \
    --number-of-steps 1000 \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --elements-in-direction-x 1 \
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1 \
    --timing-prefix ${prefix} \
    --enable-timing \
    2>&1 | tee -a ${prefix}.log

  echo "Completed L=${L}, ${P} GPU run"
done

echo "All weak scaling tests complete!"
ls -lh ${RUN_DIR}/*.log
