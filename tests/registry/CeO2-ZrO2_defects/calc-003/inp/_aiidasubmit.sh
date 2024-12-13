#!/bin/bash --login
#SBATCH --no-requeue
#SBATCH --job-name="aiida-222472"
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --partition=standard
#SBATCH --account=e05-pool
#SBATCH --qos=taskfarm
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=64
#SBATCH --time=1-00:00:00

module load PrgEnv-gnu
module load mkl

'srun' '--distribution=block:block' '--hint=nomultithread' '/work/e05/e05/zccaesa/CASTEP-21.11/bin/linux_x86_64_gfortran10-XT--mpi/castep.mpi' 'aiida'
