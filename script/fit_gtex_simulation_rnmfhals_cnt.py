
# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data_dir           = "../../topics-simulation-bigdata/output/"
read_counts_file   = "gtex_simulation_nnlm.csv"
#init_factors_file  = "gtex_simulation_rough_factors.csv"
#init_loadings_file = "gtex_simulation_rough_loadings.csv"

# These variables specify the names of the output files.
out_dir           = "../../topics-simulation-bigdata/output/"
factors_out_file  = "gtex_simulation_factors_rnmfhals_cnt.csv"
loadings_out_file  = "gtex_simulation_loadings_rnmfhals_cnt.csv"

 # # ## only for testing
#read_counts_file   = "test.csv"
#init_factors_file  = "test_factors.csv"
#init_loadings_file = "test_loadings.csv"
#factors_out_file  = "test_simulation_factors_rnmfhals.csv"
#loadings_out_file  = "test_simulation_loadings_rnmfhals.csv"
##

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
import numpy as np
import os
import sys
import time
sys.path.insert(0,'../code/')
from utility import compute_loglik
from utility import compute_least_sqr_loss
sys.path.insert(0,'../myRistretto/')
from mynmf import compute_nmf
from mynmf import compute_rnmf
from scipy.stats import poisson

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
np.random.seed(12345)
A,W = compute_rnmf(counts.T,rank=K,oversample=20,init = 'nndsvd', tol=1e-8, maxiter= MAX_ITER)
runtime = time.time() - start
print("finish fitting after: " + str(runtime))


print("save file")
## write skdlda files to file
np.savetxt(out_dir+factors_out_file, A, delimiter=",")
np.savetxt(out_dir+loadings_out_file, W.T, delimiter=",")

# compute loglikelihood 
print("compute least square")
#out = compute_loglik(counts.T,A,W)
lsq = compute_least_sqr_loss(counts.T, A.dot(W)) 

# A_nmf, W_nmf = poisson2multinom(A_nmf, W_nmf)

# mn_ll = loglik_multinom(X,A_nmf, W_nmf)
#print("type: " + out["type"])
#print("poisson loglikelihood	: " + str(out["poisson_ll"]))
#print("poisson (+eps) loglikelihood	: " + str(out["poisson_ll_eps"]))
#print("multinomial loglikelihood: " + str(out["multinom_ll"]))
print("square error             : " + str(lsq))

## write skdlda files to file
#np.savetxt(out_dir+factors_out_file, A, delimiter=",")
#np.savetxt(out_dir+loadings_out_file, W.T, delimiter=",")
# with open(out_dir + result_out_file, "wb") as f:
# 	pickle.dump(result, f)
## session info















