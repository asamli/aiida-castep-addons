#!/bin/bash
exec > _scheduler-stdout.txt
exec 2> _scheduler-stderr.txt


'mpirun' '-np' '4' '/home/alp/CASTEP-20.11/bin/linux_x86_64_gfortran10--mpi/castep.mpi' 'aiida'   
