#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --time=02:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load R/3.5.1

# Run the R script.
echo "Running generate_gtex_simulation"
export OPENBLAS_NUM_THREADS=1
R CMD BATCH --no-save --no-restore --slave generate_gtex_nnlm.R \
  ../output/generate_gtex_nnlm.Rout
