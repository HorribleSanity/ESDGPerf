#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=quick-run
#SBATCH --output=quick-run_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive

echo "=== Quick Run ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "======================================"

export JULIA_CUDA_MEMORY_POOL=none
export MPICH_GPU_SUPPORT_ENABLED=1

nvidia-smi
echo ""

JULIA="julia +1.10"

cat > launch.sh <<'EOFA'
#!/bin/bash
ngpu=$(nvidia-smi -L | grep -c UUID)
rank="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"
device=$((rank % ngpu))
export CUDA_VISIBLE_DEVICES=$device
exec "$@"
EOFA
chmod 755 launch.sh

$JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
L=6
F=Float64
P=4

echo "Running N=${N} L=${L} F=${F} with ${P} GPUs..."

time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
  --number-of-steps 100 \
  --polynomial-order ${N} \
  --refinement-level ${L} \
  --floating-point-type ${F} \
  --elements-in-direction-x 1 \
  --elements-in-direction-y 1 \
  --elements-in-direction-z 1 \
  --enable-single-buffer-comm \
  2>&1

echo ""
echo "Done!"
