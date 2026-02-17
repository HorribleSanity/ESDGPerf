#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=baroclinic
#SBATCH --output=baroclinic_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive

echo "=== Baroclinic Instability Run ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "======================================"

export JULIA_CUDA_MEMORY_POOL=none

echo "=== GPU Check ==="
nvidia-smi
echo ""

JULIA="julia +1.10"

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
$JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
L=4
F=Float64
P=4
# 10 days in seconds
TIMEEND=864000

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./baroclinic-results/run_${TIMESTAMP}_job${SLURM_JOBID}"
VTK_DIR="${RUN_DIR}/vtk"
mkdir -p ${VTK_DIR}

echo "Results will be saved to: ${RUN_DIR}"

time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/baroclinicinstability.jl \
  --polynomial-order ${N} \
  --refinement-level ${L} \
  --floating-point-type ${F} \
  --time-end ${TIMEEND} \
  --enable-vtk-output \
  --vtk-output-directory "${VTK_DIR}" \
  --vtk-output-number-of-saves 100 \
  --enable-timing \
  --timing-prefix "${RUN_DIR}/timing" \
  2>&1 | tee -a ${RUN_DIR}/baroclinic.log

echo ""
echo "Run complete!"
ls -lh ${RUN_DIR}/
