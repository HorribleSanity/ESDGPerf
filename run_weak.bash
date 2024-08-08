#!/bin/bash
#SBATCH --job-name=weak-v100
#SBATCH --output=%x-%j.txt
#SBATCH --time=2:10:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=math,beards
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1           # node count
#SBATCH --ntasks-per-node=8 # number of tasks per node
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

JULIA=/home/lwilcox/local/julia/1.10.4/bin/julia

export JULIA_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

$JULIA -e 'using Pkg; pkg"instantiate"'
$JULIA -e 'using Pkg; pkg"precompile"'

for N in `seq 2 6`
do
  for L in `seq 4 7`
  do
    for F in Float64 Float32
    do
      for P in `seq 1 8`
      do
      echo "Running F=${F} N=$N L=$L P=$P"
      prefix="${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${F}-N${N}-L${L}-P${P}-K1_1_${P}"
      mpirun -n $P ./launch.sh ${JULIA} --project=. test/run/risingbubble.jl \
        --number-of-steps 1000 \
        --polynomial-order ${N} \
        --refinement-level ${L} \
        --floating-point-type ${F} \
        --elements-in-direction-z ${P} \
        --timing-prefix ${prefix} \
        --enable-timing \
        2>&1 | tee -a ${prefix}.log
      done
    done
  done
done
