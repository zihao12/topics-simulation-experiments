#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --time=16:00:00

# For reproducibility, I add version numbers to the "module load" calls.
module load gcc/6.3.0
module load matlab/2018b

echo 'run fit_nndsvd.m on gtex data' 
matlab -nodisplay -nosplash -nodesktop -r  "dataname = 'gtex';run('fit_nndsvd.m'); exit;" > ../output/fit_gtex_nndsvd.out