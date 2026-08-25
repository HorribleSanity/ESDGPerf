#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=imbalance-test
#SBATCH --output=imbalance-test_%j.txt
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

srun julia --startup-file=no --project=. \
    benchmarks/imbalance_test.jl -N 1 -L 4
