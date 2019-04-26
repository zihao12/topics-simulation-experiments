import time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.stats import multinomial
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD


## -----------------------
## This two functions is to give the same likelihood measure as `code/misc.R`
# Convert the parameters (factors & loadings) for the Poisson model to
# the factors and loadings for the multinomial model. The return value
def poisson2multinom(A,W):
    ca = A.sum(axis = 0)
    Ahat = A.dot(np.diag(ca**-1))

    What = np.diag(ca).dot(W)
    cw = What.sum(axis = 0)
    What = What.dot(np.diag(cw**-1))

    # Xhat = Ahat.dot(What) 
    return Ahat, What

def loglik_multinom(X,A,W, e = np.finfo(float).eps):
    Xhat = A.dot(W) + e
    return (X * np.log(Xhat)).sum()

#----------------------------------

## compute the poisson and multinomial loglikelihood for results from sklearn.decomposition.nmf
def loglik_nmf(A,W,X):
    A = A
    W = W
    e = np.finfo(float).eps
    poisson_ll = (poisson.logpmf(X.flatten(),(A.dot(W)+e).flatten())).sum()

    ca = A.sum(axis = 0)
    Ahat = A.dot(np.diag(ca**-1))

    What = np.diag(ca).dot(W)
    cw = What.sum(axis = 0)
    What = What.dot(np.diag(cw**-1))

    Xhat = Ahat.dot(What) + e   
    mn_ll = 0
    for x, xtil in zip(X.T, Xhat.T):
        mn_ll = mn_ll +  multinomial.logpmf(x,n = x.sum(), p = xtil)
    return {"poisson_ll": poisson_ll,"mn_ll":mn_ll}

## compute the poisson and multinomial loglikelihood for results from sklearn.decomposition.LatentDirichletAllocation
def loglik_lda(A,W,X):
    A = A
    W = W
    e = np.finfo(float).eps
    ### Multinomial ll
    Xhat = A.dot(W) + e ## when there are 0s 
    mn_ll = 0
    for x, xtil in zip(X.T, Xhat.T):
        mn_ll = mn_ll +  multinomial.logpmf(x,n = x.sum(), p = xtil)

    ## Poisson loglikelihood
    Wtilde = W.dot(np.diag(X.sum(axis = 0))) ## transform W into Wtilde
    poisson_ll = (poisson.logpmf(X.flatten(),(A.dot(Wtilde)+e).flatten())).sum()
    return {"poisson_ll": poisson_ll,"mn_ll":mn_ll}

def multinom2poisson(A,W,X):
    Wtilde = W.dot(np.diag(X.sum(axis = 0))) ## transform W into Wtilde
    return A,Wtilde



def nmf_exper(X, K,init, tol, max_iter = 100000, random_state = 0):
    nmf = NMF(n_components=K, init=init, tol = tol, beta_loss="kullback-leibler",solver = "mu", 
                random_state=random_state, max_iter = max_iter)
    ## cd does not handle KL!!
    start = time.time()
    nmf.fit(X.T)
    L1 = nmf.transform(X.T)
    F1 = nmf.components_
    runtime = time.time() - start
        
    A_nmf = F1.T
    W_nmf = L1.T

    ll = loglik_nmf(A_nmf, W_nmf, X)
    poisson_ll = ll["poisson_ll"]
    mn_ll = ll["mn_ll"]
    return {"runtime":runtime,"n_iter":nmf.n_iter_ ,"poisson_ll":poisson_ll, "mn_ll":mn_ll}


## full process of nmf experiments: data generation, computation , summary
## N: n_sample
## P: n_feature
## K: n_groups
## sparse: if None means just random matrix; if [A_p,A_w, W_p, W_w], then we assign weight w to p proportion to  A,W
## init: init methods
## n_repeats: number of nmf on the same X
def nmf_exper_full(N,P,K, sparse, init, tol, max_iter = 10000, n_repeats = None, random_state=0):
    if sparse is None:
        X = rand_matrix_gen(N,P,K, seed = random_state)
    else:
#         A_prop = sparse[0] 
#         A_wei = sparse[1]
#         W_prop = sparse[2]
#         W_wei = sparse[3]
        X = sparse_data_gen(N, P,K, sparse,seed=random_state)
        
    
    out = {'runtime':[],'n_iter':[],'poisson_ll':[], 'mn_ll':[]}
    for i in range(n_repeats):
        exper = nmf_exper(X,K, init, tol, max_iter=max_iter, random_state=i)
        for key in out.keys():
            out[key].append(exper[key])	
    return out



def lda_exper(X, K, p_tol, e_tol, max_iter = 100000, n_jobs = 1, random_state = 0, evaluate_every = 10):
    MAX_ITER = max_iter
    ## fit to model
    lda = LDA(n_components=K, random_state=random_state,\
                learning_method='batch',max_iter = MAX_ITER,\
               evaluate_every = 10, perp_tol = p_tol, mean_change_tol = e_tol,\
               max_doc_update_iter = MAX_ITER, n_jobs = n_jobs)
    start = time.time()
    ## fit to data and transform it 
    lda.fit(X.T)
    L2 = lda.transform(X.T)
    F2 = lda.components_ 
    runtime = time.time() - start

    ## get A (p*K),W (K*n), column sums to 1
    W_lda = L2.T 
    A_lda = F2.T
    A_lda = A_lda / A_lda.sum(axis = 0)[np.newaxis,:]

    ll = loglik_lda(A_lda, W_lda, X)
    poisson_ll = ll["poisson_ll"]
    mn_ll = ll["mn_ll"]
    return {"runtime":runtime, "n_iter":lda.n_iter_, "poisson_ll":poisson_ll, "mn_ll":mn_ll}


def comparison_show(result):
#sns.boxplot(x = "method",y = "runtime", data = dAdW)
    sns.boxplot(x = "method",y = "runtime", 
                data = result) 
    plt.show()
    sns.boxplot(x = "method",y = "n_iter", 
                data = result) 
    plt.show()
    sns.boxplot(x = "method",y = "mn_ll", 
               data = result)
    plt.show()
    sns.boxplot(x = "method",y = "poisson_ll", 
                data = result)
    plt.show()

def comparison_show_save(result, path):
    with PdfPages(path) as pdf_pages:
        figu = plt.figure(1)
        sns.boxplot(x = "method",y = "runtime", 
                    data = result) 
        pdf_pages.savefig(figu)
        figu = plt.figure(2)
        sns.boxplot(x = "method",y = "n_iter",
                data = result)
        pdf_pages.savefig(figu)
        figu = plt.figure(3)
        sns.boxplot(x = "method",y = "mn_ll", 
                   data = result)
        pdf_pages.savefig(figu)
        figu = plt.figure(4)
        sns.boxplot(x = "method",y = "poisson_ll", 
                    data = result)
        pdf_pages.savefig(figu)
        
def lda_exper_etol(X, K, etols, ptol = 1, n_repeats=10):
    P,N = X.shape
    lda_results = {}
    for etol in etols:
        lda_outs = {'runtime':[],'n_iter':[], 'poisson_ll':[], 'mn_ll':[]}
        for i in range(n_repeats):
            lda_out = lda_exper(X, K, p_tol = ptol, e_tol=etol, 
                                max_iter = 10000, n_jobs = 1, random_state = i, evaluate_every = 10)
            for key in lda_out.keys():
                lda_outs[key].append(lda_out[key])
        lda_results[etol] = lda_outs   

    ## turn the dictionary to dataframe
    frames = []
    for etol in etols:
        dat = lda_results[etol]
        dat["method"] = [etol]*n_repeats
        diction = pd.DataFrame.from_dict(dat)
        frames.append(diction)
    lda_results_df = pd.concat(frames)
    
    return lda_results_df  

def nmf_exper_init(X, K, init_methods, n_repeats=10):
    P,N = X.shape
    nmf_results = {}
    for init_method in init_methods:
        nmf_outs = {'runtime':[],'n_iter':[],'poisson_ll':[], 'mn_ll':[]}
        for i in range(n_repeats):
            nmf_out = nmf_exper(X = X, K = K, init = init_method , tol = 1e-5, 
                                max_iter=10000, random_state=i)
            for key in nmf_out.keys():
                nmf_outs[key].append(nmf_out[key])
        nmf_results[init_method] = nmf_outs   

    ## turn the dictionary to dataframe
    frames = []
    for init_method in init_methods:
        dat = nmf_results[init_method]
        dat["method"] = [init_method]*n_repeats
        diction = pd.DataFrame.from_dict(dat)
        frames.append(diction)
    nmf_results_df = pd.concat(frames)
    
    return nmf_results_df  
