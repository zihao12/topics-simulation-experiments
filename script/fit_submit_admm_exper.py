## submit fitting job for matlab algorithms
# "python fit_submit_matlab.py method data"

import os
import sys

data = sys.argv[1]
r = int(sys.argv[2])
method = "admm"

path = "./"

bshname = "fit_{}_{}_r{}.sbatch".format(data, method,r)
echo = "run fit_{}.m on {} data with r {}".format(method, data, r)

matcom0 = "dataname = '{}'".format(data)
matcom1 = "r = {}".format(r)
matcom2 = "run('fit_{}_exper.m')".format(method)
matcomm = ' "{};{};{}; exit;"'.format(matcom0,matcom1, matcom2)

output = '../output/fit_{}_{}_r{}.out'.format(data, method, r)

command = 'matlab -nodisplay -nosplash -nodesktop -r {} > {}'.format(matcomm, output)

with open(path + bshname, 'w') as rsh:
	with open(path + "example_matlab.sbatch", "r") as exa:
		for item in exa.readlines():
			rsh.write(item)
	rsh.write("echo '{}' \n".format(echo))
	rsh.write(command)

## submit job
print("sbatch {}".format(bshname))
os.system("sbatch {}".format(bshname))
