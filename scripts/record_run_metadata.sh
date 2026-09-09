#!/bin/bash
# Called from the repository root by the simulation batch scripts.
set -euo pipefail

RUN_DIR="${1:?Expected a result directory}"
SUBMITTED_SCRIPT="${2:?Expected the submitted batch script path}"
METADATA_DIR="${RUN_DIR}/metadata"
mkdir -p "$METADATA_DIR"

for file in Project.toml Manifest.toml LocalPreferences.toml; do
    if [[ -f "$file" ]]; then
        cp "$file" "${METADATA_DIR}/${file}"
    fi
done
cp "$SUBMITTED_SCRIPT" "${METADATA_DIR}/submitted-script.sh"
cp launch.sh "${METADATA_DIR}/launch.sh"

git rev-parse HEAD > "${METADATA_DIR}/git-commit.txt"
git status --porcelain=v1 > "${METADATA_DIR}/git-status.txt"
git diff --binary HEAD -- . > "${METADATA_DIR}/working-tree.patch"

{
    printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    for name in SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST SLURM_NTASKS \
        JULIA_CUDA_MEMORY_POOL JULIA_MPI_HAS_CUDA MPICH_GPU_SUPPORT_ENABLED \
        CRAY_MPICH_VERSION ESDG_SAMPLE_POWER ESDG_POWER_INTERVAL_MS; do
        printf '%s=%s\n' "$name" "${!name-}"
    done
} > "${METADATA_DIR}/run-environment.txt"
