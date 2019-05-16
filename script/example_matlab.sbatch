#!/bin/bash

#SBATCH --partition=bigmem2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=16:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load gcc/6.3.0
module load matlab/2018b

