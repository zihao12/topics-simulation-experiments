## test mynmf.py

## setup import numpy as np
import numpy as np
import os
import sys
import time
sys.path.insert(0,'../code/')
from utility import compute_loglik
sys.path.insert(0,'.')
from mynmf_exper import compute_nmf
from mynmf_exper import compute_rnmf
from scipy.stats import poisson

## utility function
def compute_least_sqr_loss(X,Lam):
    p, n = X.shape
    return (np.square((X - Lam))).sum()/(n*p)

def simulate_normal(n, p, rank, seed=0):
    np.random.seed(seed)
    W = np.random.normal(loc = 1, size=(rank, n))
    W = np.exp(W)
    A = np.random.normal(loc = 1, size=(p, rank))
    A = np.exp(A)
    Lam = A.dot(W)
    X = np.random.normal(loc=Lam, scale = 1, size = (p,n)) 
    X[np.where(X < 0 )] = 0
    ll = compute_least_sqr_loss(X,Lam)
    return X, Lam, ll

## simulation
n = 1000
p = 5000
r = 5
np.random.seed(123)
X, Lam,ll = simulate_normal(n,p,r)

## test randomized NMF HALS
print("test randomized  NMF HALS")
start = time.time()
A,W,loss, ploss = compute_rnmf(X,rank=r, oversample=20,init = 'nndsvd',random_state = 0,tol=1e-04, maxiter= 200, verbose = 1, evaluate_every=20)
runtime = time.time() - start
print("runtime: " + str(runtime))
ls_ll = compute_least_sqr_loss(X,A.dot(W))
print("mean least square ll: " + str(ls_ll))
print(loss)
print(ploss)










