## test mynmf.py

## setup import numpy as np
import numpy as np
import os
import sys
import time
sys.path.insert(0,'../code/')
from utility import compute_loglik
sys.path.insert(0,'.')
from mynmf_1 import compute_nmf
from mynmf_1 import compute_rnmf
from scipy.stats import poisson
from sklearn.decomposition import NMF

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

## test NMF HALS
print("test skdnmf")
start = time.time()
model = NMF(n_components=r,init = 'nndsvd', tol=1e-05, max_iter= 10)
L1 = model.fit_transform(X.T)
F1 = model.components_
runtime = time.time() - start
print("runtime: " + str(runtime))
W = L1.T
A = F1.T
ls_ll = compute_least_sqr_loss(X,A.dot(W))
print("mean least square ll: " + str(ls_ll))


## test randomized NMF HALS
print("test randomized  NMF HALS")
start = time.time()
A,W = compute_rnmf(X,rank=r,beta_loss='kullback-leibler', oversample=20,init = 'nndsvda',random_state = 0,tol=1e-04, maxiter= 200)
runtime = time.time() - start
print("runtime: " + str(runtime))
ls_ll = compute_least_sqr_loss(X,A.dot(W))
print("mean least square ll: " + str(ls_ll))










