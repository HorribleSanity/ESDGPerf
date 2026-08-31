#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=nsys-scaling
#SBATCH --output=profile-scaling_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive

# Collects per-rank Nsight Systems profiles for the
# refinement-based weak-scaling configurations.

echo "=== Multi-GPU Nsight Systems Profiling (Weak Scaling) ==="
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_JOB_NODELIST"
echo "========================================================="

export JULIA_CUDA_MEMORY_POOL=none
#export MPICH_GPU_SUPPORT_ENABLED=1

JULIA="julia +1.10"

srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
F=Float64
STEPS=50

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Mirror weak scaling: Kx=Ky=Kz=1, increase L with GPU count
# L=5/P=1, L=6/P=8, L=7/P=64
for L in 5 6 7; do
  P=$(( 8 ** (L - 5) ))
  OUTDIR="./profiles/scaling_${TIMESTAMP}_job${SLURM_JOBID}/L${L}_P${P}"
  mkdir -p ${OUTDIR}

  echo ""
  echo "Profiling N=${N} L=${L} P=${P} for ${STEPS} steps..."
  echo "Output: ${OUTDIR}/"

  srun -n $P --ntasks-per-node=4 ./launch.sh nsys profile \
    --trace=cuda,nvtx,mpi \
    --mpi-impl=mpich \
    --stats=true \
    --force-overwrite=true \
    -o ${OUTDIR}/rank%q{SLURM_PROCID} \
    ${JULIA} --project=. test/run/risingbubble.jl \
    --polynomial-order ${N} \
    --refinement-level ${L} \
    --floating-point-type ${F} \
    --number-of-steps ${STEPS} \
    --elements-in-direction-x 1 \
    --elements-in-direction-y 1 \
    --elements-in-direction-z 1

  echo "Completed L=${L}, P=${P}"
done

echo ""
echo "All profiling complete!"
find ./profiles/scaling_${TIMESTAMP}_job${SLURM_JOBID} -name "*.nsys-rep" -ls
