#!/bin/bash

#SBATCH --partition=bigmem2
#SBATCH --account=pi-mstephens
#SBATCH --exclusive
#SBATCH --time=36:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load R/3.5.1

# Run the R script.
echo "Running fit_gtex_nnmf.R"
export OPENBLAS_NUM_THREADS=1
R CMD BATCH --no-save --no-restore --slave fit_gtex_simulation_maptpx.R \
  ../output/fit_gtex_simulation_maptpx.Rout

