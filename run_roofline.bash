#!/bin/bash
#SBATCH --job-name=roofline
#SBATCH --output=%x-%j.txt
#SBATCH --time=4:10:00
#SBATCH --mem=100G
#SBATCH --partition=math,beards
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1           # node count
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=12  # cpu-cores per task
#
echo ------------------------------------------------------
if [ "${SLURM_NNODES}" -eq "1" ]
then
    echo 'CPUS(xNODES): '${SLURM_JOB_CPUS_PER_NODE}'(x1)'
else
    echo 'CPUS(xNODES): '${SLURM_JOB_CPUS_PER_NODE}
fi
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

source /etc/profile
case $(hostname) in
  compute-6-12*)
    module use --append $HOME/opt/spack/modules/linux-rhel8-zen3;;
  compute-6-4* | compute-6-6* | compute-6-8* | compute-6-10* | compute-8-21*)
    module use --append $HOME/opt/spack/modules/linux-rhel8-zen2;;
  compute-8-25*)
    module use --append $HOME/opt/spack/modules/linux-rhel8-skylake_avx512;;
  compute-*)
    module use --append $HOME/opt/spack/modules/linux-rhel8-x86_64;;
esac
module load cuda openmpi p4est

# have the shell echo the commands that are run
set -x

# The following shell script selects the correct GPU for multi-GPU runs.
cat > launch.sh <<'EOFA'
#!/bin/bash

# Choose a CUDA device number ($device) based on
# ${OMPI_COMM_WORLD_LOCAL_RANK}, cycling through the available GPU devices
# ($ngpu) on the node.
ngpu=$(nvidia-smi -L | grep -c UUID)
rank="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"
device=$((rank % ngpu))
export CUDA_VISIBLE_DEVICES=$device

# Run MPI program with any arguments
exec "$@"
EOFA
chmod 755 launch.sh

# Time
metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"
# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"
# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"
# DP,SP,HP Efficiency
metrics+="smsp__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,"
# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"
# DRAM, L2 and L1
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum"

kernels="regex:(rkupdate_knl|face_knl|volume_knl)"

export TMPDIR=/tmp/lwilcox

JULIA=/home/lwilcox/local/julia/1.10.4/bin/julia
NCU=/home/lwilcox/local/NVIDIA-Nsight-Compute-2024.2/ncu

# Have julia only use one thread.
 export JULIA_NUM_THREADS=12
 export OPENBLAS_NUM_THREADS=12

# MPI implementations will pin CUDA buffers which is a problem when
# using a memory pools.
# export JULIA_CUDA_MEMORY_POOL=none

# Trace GC allocations
export JULIA_NVTX_CALLBACKS=gc,alloc,free

$JULIA -e 'using Pkg; pkg"instantiate"'
$JULIA -e 'using Pkg; pkg"precompile"'

# Four optimization steps 
for N in `seq 2 6`
do
  for L in `seq 1 7`
  do
    for F in Float64 Float32
    do
      echo "Running N=$N L=$L"
      prefix="${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${F}-N${N}-L${L}"
      profilestr="${NCU} -k ${kernels} --metrics ${metrics} --csv --log-file ${prefix}.precsv"
      juliastr="${JULIA} --project=. test/run/risingbubble.jl --number-of-steps 1 --polynomial-order ${N} --refinement-level ${L} --floating-point-type ${F}"
      srun -n1 $profilestr ./launch.sh $juliastr 2>&1 | tee -a ${prefix}.log
      sed '/^==/d' < ${prefix}.precsv > ${prefix}.csv
    done
  done
done
