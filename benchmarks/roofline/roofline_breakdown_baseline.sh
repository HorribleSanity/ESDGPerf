#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=breakdown-baseline
#SBATCH --output=breakdown-baseline_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --gpu-bind=closest
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --constraint=perf,nvperf
#SBATCH --exclusive

# Profile baseline kernel for FULL kernel breakdown (volume + face + RK update)
# at all polynomial orders, L=5, on A100. Output is used to produce the
# before/after kernel-breakdown plot for the paper.
#
# Usage:
#   sbatch benchmarks/roofline/roofline_breakdown_baseline.sh

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR}"
BENCH="${REPO}/benchmarks/roofline"

# Step 00 is the pre-optimization baseline volume kernel.
KFILE="step00-baseline.jl"
CONTRACT="off"

# All polynomial orders, just L=5 (to match the existing kernel_breakdown plot).
N_VALUES=(2 3 4 5 6)
L_VALUES=(5)

log() { echo "$(date +%H:%M:%S) $1"; }

JULIA=$(which julia)
NCU=$(which ncu)
echo "julia: $JULIA"
echo "ncu:   $NCU"

unset MPICH_GPU_SUPPORT_ENABLED
dcgmi profile --pause 2>/dev/null || true
trap 'dcgmi profile --resume 2>/dev/null || true' EXIT

# ── Set up isolated worktree ──
wt="${REPO}/.worktrees/breakdown-baseline-${SLURM_JOBID}"
[ -d "$wt" ] && { git -C "$REPO" worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"; }
git -C "$REPO" worktree add "$wt" HEAD --detach --quiet
log "worktree: $wt"

cp "$BENCH/kernels/$KFILE" "$wt/src/kernels/volume_kernel_vfinal.jl"
cp "$BENCH/contract_${CONTRACT}.jl" "$wt/src/contract.jl"

log "precompiling..."
srun -n 1 $JULIA --project="$wt" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ── Profile all (N, L) ──
outdir="${REPO}/ncu-profiles/breakdown-baseline-a100"
mkdir -p "$outdir"

for N in "${N_VALUES[@]}"; do
    for L in "${L_VALUES[@]}"; do
        prefix="${outdir}/baseline-${SLURM_JOBID}-Float64-N${N}-L${L}"
        log "N=$N L=$L"

        # NOTE: kernel filter now includes face_knl and rkupdate_knl
        # so we get the full step breakdown, not just volume.
        $NCU --clock-control none \
             --print-units base \
             --set roofline \
             -k "regex:(volume_knl|face_knl|rkupdate_knl)" \
             -o "$prefix" -f \
             $JULIA --project="$wt" "$wt/test/run/risingbubble.jl" \
             --number-of-steps 1 -N "$N" -L "$L" -F Float64 \
             2>&1 | tee -a "${prefix}.log"

        if [ -f "${prefix}.ncu-rep" ]; then
            $NCU --csv --print-units base --page raw -i "${prefix}.ncu-rep" > "${prefix}.csv"
            log "ok: $(basename "$prefix").csv"
        else
            log "FAIL: no .ncu-rep for N=$N L=$L"
        fi
    done
done

git -C "$REPO" worktree remove "$wt" --force 2>/dev/null || true
log "── done ──"
ls "$outdir"/*.csv 2>/dev/null | wc -l | xargs -I{} echo "CSVs produced: {}"
