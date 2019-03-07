
# SCRIPT SETTINGS
# ---------------
## need to mannually set 
test = False
data_dir           = "../../topics-simulation-bigdata/output/"
out_dir           = "../../topics-simulation-bigdata/output/"
read_counts_file   = "gtex_simulation_nnlm.csv"

## specify data used and output name
dataset_name = "gtexsim"
if test:
	dataset_name = "test"
	read_counts_file = "test.csv"

method_name = "rnmfhals"
factors_out_file  = dataset_name + "_F_factors_" + method_name + ".csv"
loadings_out_file  = dataset_name + "_F_loadings_" + method_name + ".csv"
loss_out_file  = dataset_name + "_F_loss_"+ method_name + ".pkl"
surloss_out_file  = dataset_name + "_F_surloss_" + method_name + ".pkl"


# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
import numpy as np
import os
import sys
import time
import pickle
sys.path.insert(0,'../code/')
from utility import compute_loglik
from utility import compute_least_sqr_loss
sys.path.insert(0,'../myRistretto/')
from mynmf_exper import compute_nmf
from mynmf_exper import compute_rnmf
from scipy.stats import poisson
from decimal import Decimal

# LOAD GTEX DATA
# ------------------
print("Loading GTEx data.")
counts = np.loadtxt(open(data_dir + read_counts_file, "rb"), delimiter=",", skiprows=0)
#print("data shape: p = " + str(X.shape[0], "  n = " str(X.shape[1])))
print("data shape after transposing:")
print(counts.shape)
## p = 55863
## n = 11688
## k = 20



# LOAD INITIAL ESTIMATES
# ----------------------

## loading is [p, K]


# Get the number of factors ("topics").
K = 20

# RUN RNMFHALS OPTIMIZATION METHOD
# ---------------------------
MAX_ITER = 10000

print("start fitting")
start = time.time()
A,W,loss, surloss = compute_rnmf(counts.T,rank=K,oversample=20,init = 'nndsvd', tol=1e-10, 
	maxiter= MAX_ITER, verbose = 1, evaluate_every=100,random_state= 12345)
runtime = time.time() - start
print("finish fitting after: {:3f}".format(runtime))

print("save file")
np.savetxt(out_dir+factors_out_file, A, delimiter=",")
np.savetxt(out_dir+loadings_out_file, W.T, delimiter=",")
with open(out_dir + loss_out_file, "wb") as f:
	pickle.dump(loss, f)
with open(out_dir + surloss_out_file, "wb") as f:
	pickle.dump(surloss, f)


# compute loss 
print("compute loss")
lsq = compute_least_sqr_loss(counts.T, A.dot(W)) 
print("square error: {:10f}".format(lsq))

# if test:
# 	print(loss)
# 	print(surloss)













