#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=rb-visual
#SBATCH --output=rb-visual_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --exclusive

N=3
L=6
P=8
TIMEEND=1000

echo "=== Rising Bubble Visualization ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "N=${N} L=${L} P=${P} timeend=${TIMEEND}"
echo "======================================"

export JULIA_CUDA_MEMORY_POOL=none
#export MPICH_GPU_SUPPORT_ENABLED=1

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

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./rb-visual-results/run_${TIMESTAMP}_job${SLURM_JOBID}_N${N}_L${L}"
mkdir -p ${RUN_DIR}

echo "Results: ${RUN_DIR}"

for F in Float64 Float32; do
  VTK_DIR="${RUN_DIR}/vtk-${F}"
  mkdir -p ${VTK_DIR}
  prefix="${RUN_DIR}/timing-${F}-N${N}-L${L}-P${P}"

  echo ""
  echo ">>> Running ${F} N=${N} L=${L} P=${P} timeend=${TIMEEND} ..."

  time srun -n $P --ntasks-per-node=4 ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --time-end ${TIMEEND} \
    --elements-in-direction-x 1 \
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1 \
    --enable-vtk-output \
    --vtk-output-directory "${VTK_DIR}" \
    --vtk-output-number-of-saves 10 \
    --enable-timing \
    --timing-prefix ${prefix} \
    2>&1 | tee -a ${RUN_DIR}/${F}.log

  echo "<<< Completed ${F}"
done

echo ""
echo "All runs complete!"
ls -lh ${RUN_DIR}/
