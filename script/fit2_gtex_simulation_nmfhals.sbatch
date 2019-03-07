#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --account=pi-mstephens
#SBATCH --exclusive
#SBATCH --time=36:00:00

# For reproducibility, I add version numbers to the "module load" calls.
#module load R/3.5.1
module load Anaconda3/4.3.0
# Run the R script.
echo "fit2_gtex_simulation_nmfhals.py"
#export OPENBLAS_NUM_THREADS=1
python -u fit2_gtex_simulation_nmfhals.py > ../output/fit2_gtex_simulation_nmfhals.pyout
