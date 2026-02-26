#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=logavg-compare
#SBATCH --output=comparison_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive

# ============================================================
# Set this tag before each submission:
#   "baseline"  = original logavg
#   "optimized" = modified logavg
# ============================================================
TAG="visual"

N=4
L=6
P=8
NSTEPS=1000

echo "=== Comparison run: TAG=${TAG} ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "N=${N} L=${L} P=${P} NSTEPS=${NSTEPS}"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./comparison-results/${TAG}_${TIMESTAMP}_job${SLURM_JOBID}"
mkdir -p ${RUN_DIR}

echo "Results will be saved to: ${RUN_DIR}"

for F in Float64 Float32; do
  prefix="${RUN_DIR}/${TAG}-${F}-N${N}-L${L}-P${P}"

  echo ""
  echo ">>> Running ${TAG} ${F} N=${N} L=${L} P=${P} ..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
    --time-end 1000 \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --elements-in-direction-x 1 \
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1 \
    --timing-prefix ${prefix} \
    --enable-timing \
    --enable-vtk-output \
    --vtk-output-directory "${RUN_DIR}/vtk-${TAG}-${F}" \
    --vtk-output-number-of-saves 3 \
    2>&1 | tee -a ${prefix}.log

  echo "<<< Completed ${TAG} ${F}"
done

echo ""
echo "All runs complete!"
ls -lh ${RUN_DIR}/
