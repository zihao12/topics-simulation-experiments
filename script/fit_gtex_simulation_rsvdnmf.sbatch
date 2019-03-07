#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --account=pi-mstephens
#SBATCH --exclusive
#SBATCH --time=36:00:00

# For reproducibility, I add version numbers to the "module load" calls.
#module load R/3.5.1
module load Anaconda3/4.3.0
# Run the R script.
echo "fit_gtex_simulation_rsvdnmf.py"
#export OPENBLAS_NUM_THREADS=1
python -u fit_gtex_simulation_rsvdnmf.py > ../output/fit_gtex_simulation_rsvdnmf.pyout
