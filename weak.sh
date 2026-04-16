#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=weak-test
#SBATCH --output=weak_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive

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
export MPICH_GPU_SUPPORT_ENABLED=1



# ===== ADD DIAGNOSTIC CHECKS HERE =====
echo "=== GPU Check ==="
nvidia-smi
echo ""

echo "=== MPI Version ==="
mpiexec --version
echo ""

echo "=== MPI Environment Variables ==="
printenv | grep -i mpi | sort
echo ""

echo "=== CUDA Environment ==="
printenv | grep -i cuda
echo ""

#JULIA=$(which julia)
JULIA="julia +1.10"
#MPI=/u/hwaterhouse/.julia/bin/mpiexecjl
#MPI=mpirun

echo "Using JULIA: $JULIA"
echo "Job ID: $SLURM_JOBID"

# Create launch script for GPU selection
cat > launch.sh <<'EOFA'
#!/bin/bash
ngpu=$(nvidia-smi -L | grep -c UUID)
rank="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"
device=$((rank % ngpu))
export CUDA_VISIBLE_DEVICES=$device
exec "$@"
EOFA
chmod 755 launch.sh

# Instantiate once
srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
L=6
F=Float64

# Create unique directory for this run using timestamp and job ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./weak-scaling-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}"
mkdir -p ${RUN_DIR}

echo "Results will be saved to: ${RUN_DIR}"



# Loop through GPU counts
for P in 4 8 12 16; do
  prefix="${RUN_DIR}/weak-${SLURM_JOBID}-${F}-N${N}-L${L}-P${P}"

  echo "Running N=${N} L=${L} F=${F} with ${P} GPUs..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
    --number-of-steps 1000 \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --elements-in-direction-x 1\
    --elements-in-direction-y 1 \
    --elements-in-direction-z ${P} \
    --timing-prefix ${prefix} \
    --enable-timing \
    2>&1 | tee -a ${prefix}.log

  echo "Completed ${P} GPU run"
done

echo "All weak scaling tests complete!"
ls -lh ${RUN_DIR}/*.log
