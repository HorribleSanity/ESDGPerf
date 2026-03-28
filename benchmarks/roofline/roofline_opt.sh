#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=roofline-opt
#SBATCH --output=roofline-opt_%A_%a.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --gpu-bind=closest
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --constraint=perf,nvperf
#SBATCH --exclusive
#SBATCH --array=0-13

# Roofline profiling across 14 volume kernel optimization steps.
#
# Each array task creates a git worktree (isolated repo copy), swaps in
# the pre-extracted kernel file for that step, and runs ncu over all (N,L).
# The original working tree is never modified.
#
# Usage:
#   sbatch benchmarks/roofline/roofline_opt.sh            # all 14 in parallel
#   sbatch --array=0,13 benchmarks/roofline/roofline_opt.sh   # just first & last
#   SEQUENTIAL=true sbatch benchmarks/roofline/roofline_opt.sh # all 14, one node

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR}"
BENCH="${REPO}/benchmarks/roofline"

# ── Step definitions ──────────────────────────────────────────────────────────
# Step 00 uses VolumeKernel{0} (the v0 baseline); steps 01-13 swap vfinal.jl.
# Steps 11-13 also enable contract nsz; steps 00-10 disable it.
#
#             step   name             kernel file                  contract
STEPS=(
    "00  baseline         BASELINE                       off"
    "01  prim             step01-prim.jl                 off"
    "02  logave           step02-logave.jl               off"
    "03  precomp-logs     step03-precomp-logs.jl         off"
    "04  dq-not-flux      step04-dq-not-flux.jl          off"
    "05  aux-shared       step05-aux-shared.jl           off"
    "06  fused            step06-fused.jl                off"
    "07  symmetry         step07-symmetry.jl             off"
    "08  inv-logave       step08-inv-logave.jl           off"
    "09  no-prefetch      step09-no-prefetch.jl          off"
    "10  wip-agg          step10-wip-agg.jl              off"
    "11  contract         step11-contract.jl             on"
    "12  tendency-opt     step12-tendency-opt.jl         on"
    "13  ldg-const        step13-ldg-const.jl            on"
)

N_VALUES=(2 3 4 5 6)
L_VALUES=(2 3 4 5 6)

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[step$1] $(date +%H:%M:%S) $2"; }

run_step() {
    local idx=$1
    read -r num name kfile contract <<< "${STEPS[$idx]}"

    log "$num" "── $name ──"

    # Create an isolated worktree (full repo checkout at HEAD in a separate dir).
    local wt="${REPO}/.worktrees/s${num}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOBID}"
    [ -d "$wt" ] && { git -C "$REPO" worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"; }
    git -C "$REPO" worktree add "$wt" HEAD --detach --quiet
    log "$num" "worktree: $wt"

    # ── Swap kernel ──
    if [ "$kfile" = "BASELINE" ]; then
        # Step 00: use the v0 kernel by changing which VolumeKernel the test selects.
        sed -i 's/VolumeKernel{:final}()/VolumeKernel{0}()/' "$wt/test/run/risingbubble.jl"
        grep -q 'VolumeKernel{0}' "$wt/test/run/risingbubble.jl" || { log "$num" "ERROR: baseline patch failed"; return 1; }
    else
        cp "$BENCH/kernels/$kfile" "$wt/src/kernels/volume_kernel_vfinal.jl"
    fi

    # ── Swap contract.jl ──
    cp "$BENCH/contract_${contract}.jl" "$wt/src/contract.jl"

    # ── Precompile (once per step; N/L changes don't trigger recompilation) ──
    log "$num" "precompiling..."
    $JULIA --project="$wt" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

    # ── Profile all (N, L) combinations ──
    local outdir="${REPO}/ncu-profiles/optimization"
    mkdir -p "$outdir"

    for N in "${N_VALUES[@]}"; do
        for L in "${L_VALUES[@]}"; do
            local prefix="${outdir}/opt${num}-${name}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOBID}-Float64-N${N}-L${L}"
            log "$num" "N=$N L=$L"

            $NCU --clock-control none \
                 --print-units base \
                 --set roofline \
                 -k "regex:volume_knl" \
                 -o "$prefix" -f \
                 $JULIA --project="$wt" "$wt/test/run/risingbubble.jl" \
                 --number-of-steps 1 -N "$N" -L "$L" -F Float64 \
                 2>&1 | tee -a "${prefix}.log"

            if [ -f "${prefix}.ncu-rep" ]; then
                $NCU --csv --print-units base --page raw -i "${prefix}.ncu-rep" > "${prefix}.csv"
                log "$num" "ok: $(basename "$prefix").csv"
            else
                log "$num" "FAIL: no .ncu-rep for N=$N L=$L"
            fi
        done
    done

    git -C "$REPO" worktree remove "$wt" --force 2>/dev/null || true
    log "$num" "── done ──"
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "=== roofline-opt | job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOBID} | task ${SLURM_ARRAY_TASK_ID:-all} | $(hostname) ==="

JULIA=$(which julia)
NCU=$(which ncu)
echo "julia: $JULIA"
echo "ncu:   $NCU"

unset MPICH_GPU_SUPPORT_ENABLED
dcgmi profile --pause 2>/dev/null || true
trap 'dcgmi profile --resume 2>/dev/null || true' EXIT

# Ensure packages are downloaded (shared depot).
$JULIA --project="$REPO" -e 'using Pkg; Pkg.instantiate()'

if [ "${SEQUENTIAL:-false}" = "true" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    for i in $(seq 0 $((${#STEPS[@]} - 1))); do run_step "$i"; done
else
    run_step "$SLURM_ARRAY_TASK_ID"
fi

echo "done. CSV count:"
ls "${REPO}/ncu-profiles/optimization/"opt*.csv 2>/dev/null | wc -l
