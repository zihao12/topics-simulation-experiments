## submit job for generating data from fits of a method
## first argument is method
## second argument is data (original)


import os
import sys

method = sys.argv[1]
data = sys.argv[2]

path = "./"

bshname = "generate_{}_{}.sbatch".format(data, method)
echo = "run generate_sim.R {} data with {} method".format(data, method)

command = 'Rscript generate_sim.R {} {}'.format(method, data)

with open(path + bshname, 'w') as rsh:
	with open(path + "example_R.sbatch", "r") as exa:
		for item in exa.readlines():
			rsh.write(item)
	rsh.write("echo '{}' \n".format(echo))
	rsh.write(command)

## submit job
print("sbatch {}".format(bshname))
os.system("sbatch {}".format(bshname))