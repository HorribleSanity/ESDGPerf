#!/bin/bash
#SBATCH --account=bfit-delta-gpu
#SBATCH --job-name=roofline-profile
#SBATCH --output=roofline_%j.txt
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

# Collects Nsight Compute roofline metrics for the configured N/L cases.

echo 'Job is running on nodes:'
echo $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: submission node:        $SLURM_SUBMIT_HOST
echo SLURM: partition:              $SLURM_JOB_PARTITION
echo SLURM: submission directory:   $SLURM_SUBMIT_DIR
echo SLURM: job identifier:         $SLURM_JOBID
echo SLURM: job name:               $SLURM_JOB_NAME
echo SLURM: current home directory: $HOME
echo SLURM: PATH:                   $PATH
echo ------------------------------------------------------

JULIA=$(which julia)
NCU=$(which ncu)

#export CUDA_VISIBLE_DEVICES=2

echo "Using NCU: $NCU"
echo "Using JULIA: $JULIA"
echo "Job ID: $SLURM_JOBID"

# Define which kernels to profile
kernels="regex:(rkupdate_knl|face_knl|volume_knl)"

# Instantiate once
srun -n 1 $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

outdir="./ncu-profiles/roofline-${SLURM_JOBID}"
mkdir -p "${outdir}"
echo "Results: ${outdir}"

# Loop through configurations
for N in 2 3 4 5 6; do
  for L in 2 3 4 5 6; do
    for F in Float64; do
      prefix="${outdir}/profile-${F}-N${N}-L${L}"
      
      echo "Profiling N=${N} L=${L} F=${F}..."
      $NCU --clock-control none \
	   --print-units base \
           --set roofline \
           -k ${kernels} \
           -o ${prefix} \
           -f \
           $JULIA --project=. test/run/risingbubble.jl \
           --number-of-steps 1 --polynomial-order ${N} \
           --refinement-level ${L} --floating-point-type ${F} 2>&1 | tee -a ${prefix}.log
      
      # Convert to CSV if profiling succeeded
      if [ -f ${prefix}.ncu-rep ]; then
        $NCU --csv --print-units base --page raw -i ${prefix}.ncu-rep > ${prefix}.csv

        echo "SUCCESS: ${prefix}.csv created"
      else
        echo "FAILED: No .ncu-rep file for N=${N} L=${L} F=${F}"
      fi
    done
  done
done

echo "All profiling complete!"
echo "Results: ${outdir}"
ls -lh "${outdir}"/profile-*.csv
