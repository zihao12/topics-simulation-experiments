#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --time=36:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load R/3.5.1

# Run the R script.
echo "Running fit_gtex_simulation_nnmf_F.R"
export OPENBLAS_NUM_THREADS=1
R CMD BATCH --no-save --no-restore --slave fit_gtex_simulation_nnmf_F.R \
  ../output/fit_gtex_simulation_nnmf_F.Rout
