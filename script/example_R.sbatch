#!/bin/bash

#SBATCH --partition=bigmem2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=16:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load R/3.5.1

export OPENBLAS_NUM_THREADS=1
