## fit gtex data using skd.LDA

# GLOBAL VARIABLE
tol = 1e-8
neglogtol = 8
# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data_dir           = "../../topics-simulation-bigdata/output/"
read_counts_file   = "gtex_simulation_nnlm.csv"
init_factors_file  = "gtex_simulation_rough_factors.csv"
init_loadings_file = "gtex_simulation_rough_loadings.csv"

# # # ## only for testing
# read_counts_file   = "test.csv"
# init_factors_file  = "test_factors.csv"
# init_loadings_file = "test_loadings.csv"

# These variables specify the names of the output files.
out_dir           = "../../topics-simulation-bigdata/output/"
factors_out_file  = "gtex_simulation_factors_skdnmf_logtol_" + str(neglogtol) + ".csv"
loadings_out_file  = "gtex_simulation_loadings_skdnmf_logtol_" + str(neglogtol) + ".csv"
#result_out_file  = "gtex_perform_skdnmf_logtol_" + str(neglogtol) + ".pkl"

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
import sys
sys.path.insert(0,'../code/')
import utility
from utility import compute_loglik
import pickle

# LOAD GTEX DATA
# ------------------
print("Loading GTEx data.")
X = np.loadtxt(open(data_dir + read_counts_file, "rb"), delimiter=",", skiprows=0)
X = X.T
#print("data shape: p = " + str(X.shape[0], "  n = " str(X.shape[1])))
print("data shape after transposing:")
print(X.shape)
## p = 55863
## n = 11688
## k = 20



# LOAD INITIAL ESTIMATES
# ----------------------

## loading is [p, K]
A0 = np.loadtxt(open(data_dir + init_factors_file, "rb"), delimiter=",", skiprows=0)
W0 = np.loadtxt(open(data_dir + init_loadings_file, "rb"), delimiter=",", skiprows=0)
W0 = W0.T


# Get the number of factors ("topics").
K = A0.shape[1]

# RUN SKDLDA OPTIMIZATION METHOD
# ---------------------------
MAX_ITER = 100000
model = NMF(n_components=K, init="custom", tol = tol, beta_loss="kullback-leibler",solver = "mu",
                random_state=0, max_iter = MAX_ITER, verbose = True)

print("start fitting")
start = time.time()
model.fit(X.T, W = W0.T, H = A0.T)
L1 = model.transform(X.T)
F1 = model.components_
runtime = time.time() - start
print("finish fitting after: " + str(runtime))

# compute loglikelihood 
print("compute loglikelihood")
W = L1.T
A = F1.T

out = compute_loglik(X,A,W)


# A_nmf, W_nmf = poisson2multinom(A_nmf, W_nmf)

# mn_ll = loglik_multinom(X,A_nmf, W_nmf)
print("type: " + out["type"])
print("poisson loglikelihood	: " + str(out["poisson_ll"]))
print("multinomial loglikelihood: " + str(out["multinom_ll"]))
# result = {"runtime":runtime, "mn_ll":mn_ll}

## write skdlda files to file
np.savetxt(out_dir+factors_out_file, A, delimiter=",")
np.savetxt(out_dir+loadings_out_file, W.T, delimiter=",")
# with open(out_dir + result_out_file, "wb") as f:
# 	pickle.dump(result, f)
## session info















