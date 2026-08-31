#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=strong-test
#SBATCH --output=strong_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive

# Set problem size, increasing processors

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

#module load julia/1.10.2
#module load openmpi
#module load cuda/12.4.0
export JULIA_CUDA_MEMORY_POOL=none
#export MPICH_GPU_SUPPORT_ENABLED=1



echo "=== GPU Check ==="
nvidia-smi
echo ""

echo "=== Slurm Launcher Version ==="
srun --version
echo ""

echo "=== MPI Environment Variables ==="
printenv | grep -i mpi | sort
echo ""

echo "=== CUDA Environment ==="
printenv | grep -i cuda
echo ""

#JULIA=$(which julia)
JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"
#MPI=/u/hwaterhouse/.julia/bin/mpiexecjl
#MPI=mpirun

echo "Using JULIA: $JULIA"
echo "Job ID: $SLURM_JOBID"

# Instantiate once
srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Fixed problem size for strong scaling
N=4
L=6
F=Float64

# Create unique directory for this run using timestamp and job ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_ROOT}/strong-scaling-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}"
mkdir -p ${RUN_DIR}

echo "Results will be saved to: ${RUN_DIR}"



# Loop through GPU counts
for P in 4 8 12 16; do
  VTK_DIR="${RUN_DIR}/vtk-out-P${P}"
  mkdir -p ${VTK_DIR}
  prefix="${RUN_DIR}/strong-${SLURM_JOBID}-${F}-N${N}-L${L}-P${P}"

  echo "Running N=${N} L=${L} F=${F} with ${P} GPUs..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
    --number-of-steps 1000 \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --elements-in-direction-x 1\
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1 \
    --timing-prefix ${prefix} \
    --enable-timing \
    --enable-vtk-output\
    --vtk-output-directory "${VTK_DIR}"\
    --vtk-output-number-of-saves 5\
    2>&1 | tee -a ${prefix}.log

  echo "Completed ${P} GPU run"
done

echo "All strong scaling tests complete!"
ls -lh ${RUN_DIR}/*.log
