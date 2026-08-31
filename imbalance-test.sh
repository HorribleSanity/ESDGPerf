#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=imbalance-test
#SBATCH --output=imbalance-test_%j.txt
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive

set -euo pipefail

JULIA="julia +1.10"
OUTPUT_ROOT="${ESDG_OUTPUT_ROOT:-/scratch/bfit/${USER}/ESDGPerf}"

srun --nodes=1 --ntasks=1 $JULIA --startup-file=no --project=. -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile()'

N=4
L=6
RESULTS_DIR="${OUTPUT_ROOT}/imbalance-results"
mkdir -p "${RESULTS_DIR}"
RESULTS="${RESULTS_DIR}/imbalance-results_${SLURM_JOB_ID}.tsv"

echo "Writing per-rank data to ${RESULTS}"

for P in 1 2 4 8 16 32; do
    echo ""
    echo "=== Load imbalance: N=${N}, L=${L}, ranks=${P} ==="
    srun -n "$P" --ntasks-per-node=4 \
        $JULIA --startup-file=no --project=. \
        benchmarks/imbalance_test.jl -N "$N" -L "$L" \
        --output-file "$RESULTS"
done
