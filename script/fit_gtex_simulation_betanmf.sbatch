#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --time=10:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load matlab/2018b

# Run the MATLAB script.
echo "Running fit_gtex_betanmf.m"
matlab -nodisplay -nosplash -nodesktop \
  -r "run('fit_gtex_simulation_betanmf.m'); exit;" \
  > ../output/fit_gtex_simulation_betanmf.out
