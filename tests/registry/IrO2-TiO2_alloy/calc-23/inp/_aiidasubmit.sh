#!bin/bash -l
#$ -cwd
#$ -S /bin/bash
#$ -r no
#$ -m n
#$ -N aiida-117623
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe mpi 18
#$ -l h_rt=48:00:00

'gerun' '/home/zccaesa/bin/CASTEP-20.11/bin/linux_x86_64_ifort--mpi/castep.mpi' 'aiida'   
