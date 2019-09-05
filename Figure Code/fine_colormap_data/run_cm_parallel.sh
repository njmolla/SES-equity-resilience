#!/bin/bash -l
#SBATCH -n 100            # Total number of processors to request (32 cores per node)
#SBATCH -p high          # Queue name hi/med/lo
#SBATCH -t 200:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=njmolla@ucdavis.edu              # address for email notification
#SBATCH --mail-type=ALL                  # email at Begin and End of job

GDIR=/group/hermangrp

# IMPORTANT: Python3/Pyomo/CBC solver are all installed in group directory. Add it to the path.
export PATH=$GDIR/miniconda3/bin:$PATH

mpirun python cm_parallel.py