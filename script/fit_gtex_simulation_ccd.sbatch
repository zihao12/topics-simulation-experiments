#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --time=16:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load gcc/6.3.0
module load matlab/2018b

# Run the MATLAB script.
echo "Running fit_gtex_ccd.m"
matlab -nodisplay -nosplash -nodesktop \
  -r "run('fit_gtex_simulation_ccd.m'); exit;" \
  > ../output/fit_gtex_simulation_ccd.out
