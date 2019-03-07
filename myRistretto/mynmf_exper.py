"""
Nonnegative Matrix Factorization.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division

import numpy as np
from scipy import linalg

from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition.nmf import _initialize_nmf
from sklearn.utils import check_random_state

from qb import compute_rqb
from mysvd_exper import compute_rsvd

## my 
from utils import compute_least_sqr_loss

## my import
import time
from decimal import Decimal


_VALID_DTYPES = (np.float32, np.float64)


def compute_nmf(A, rank, init='nndsvd', shuffle=False,
                l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
                tol=1e-5, maxiter=200, random_state=None, verbose = 1, evaluate_every = 10):

    """Nonnegative Matrix Factorization.

    Hierarchical alternating least squares algorithm
    for computing the approximate low-rank nonnegative matrix factorization of
    a rectangular `(m, n)` matrix `A`. Given the target rank `rank << min{m,n}`,
    the input matrix `A` is factored as `A = W H`. The nonnegative factor
    matrices `W` and `H` are of dimension `(m, rank)` and `(rank, n)`, respectively.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    rank : integer, `rank << min{m,n}`.
        Target rank.

    init :  'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure. Default: 'nndsvd'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

    shuffle : boolean, default: False
        If true, randomly shuffle the update order of the variables.

    l2_reg_H : float, (default ``l2_reg_H = 0.1``).
        Amount of ridge shrinkage to apply to `H` to improve conditionin.

    l2_reg_W : float, (default ``l2_reg_W = 0.1``).
        Amount of ridge shrinkage to apply to `W` to improve conditionin.

    l1_reg_H : float, (default ``l1_reg_H = 0.0``).
        Sparsity controlling parameter on `H`.
        Higher values lead to sparser components.

    l1_reg_W : float, (default ``l1_reg_W = 0.0``).
        Sparsity controlling parameter on `W`.
        Higher values lead to sparser components.

    tol : float, default: `tol=1e-4`.
        Tolerance of the stopping condition.

    maxiter : integer, default: `maxiter=100`.
        Number of iterations.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    verbose : boolean, default: `verbose=False`.
        The verbosity level.


    Returns
    -------
    W:  array_like, `(m, rank)`.
        Solution to the non-negative least squares problem.

    H : array_like, `(rank, n)`.
        Solution to the non-negative least squares problem.

    ## zihao:
    loss: a list of (niter, loss)


    Notes
    -----
    This HALS update algorithm written in cython is adapted from the
    scikit-learn implementation for the deterministic NMF. We also have
    adapted the initilization scheme.

    See: https://github.com/scikit-learn/scikit-learn


    References
    ----------
    [1] Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    [2] C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd


    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> import ristretto as ro
    >>> W, H = ro.nmf(X, rank=2)
    """
    random_state = check_random_state(random_state)
    loss_out = []

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if np.any(A < 0):
        raise ValueError("Input matrix with nonnegative elements is required.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    W, H = _initialize_nmf(A, rank, init=init, eps=1e-6, random_state=random_state)
    Ht = np.array(H.T, order='C')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H and normalize columns
    # ii)  Update low-dimensional factor matrix W
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    violation = 0.0
    for niter in range(maxiter):
        start = time.time()
        # Update factor matrix H with regularization
        WtW = W.T.dot(W)
        WtW.flat[::rank + 1] += l2_reg_H  # adds l2_reg only on the diagonal
        AtW = A.T.dot(W) - l1_reg_H

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation = _update_cdnmf_fast(Ht, WtW, AtW, permutation)

        # Update factor matrix W with regularization
        HHt = Ht.T.dot(Ht)
        HHt.flat[::rank + 1] += l2_reg_W # adds l2_reg only on the diagonal

        AHt = A.dot(Ht) - l1_reg_W

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation += _update_cdnmf_fast(W, HHt, AHt, permutation) 


        # Compute stopping condition.
        if niter == 0:
            if violation == 0: break
            violation_init = violation

        if violation / violation_init <= tol:
            break
        runtime = time.time() - start
        if verbose:
            #print("n_iter: "+ str(niter) + "   runtime: "+ str(runtime))	
            print("n_iter: {:d}     runtime: {:.3e}  rel violation: {:.5e}".format(niter, Decimal(runtime), Decimal(violation / violation_init)))
            if niter % evaluate_every == 0:
                loss = compute_least_sqr_loss(A,W.dot(Ht.T))
                loss_out.append((niter, loss))
                print("mean square error: {:.5e}".format(Decimal(loss)))



    # Return factor matrices
    return W, Ht.T, loss_out


def compute_rnmf(A, rank, oversample=20, n_subspace=2, init='nndsvd', shuffle=False,
                 l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
                 tol=1e-5, maxiter=200, random_state=None, verbose = 1, evaluate_every=100):
    """
    Randomized Nonnegative Matrix Factorization.

    Randomized hierarchical alternating least squares algorithm
    for computing the approximate low-rank nonnegative matrix factorization of
    a rectangular `(m, n)` matrix `A`. Given the target rank `rank << min{m,n}`,
    the input matrix `A` is factored as `A = W H`. The nonnegative factor
    matrices `W` and `H` are of dimension `(m, rank)` and `(rank, n)`, respectively.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and the parameter `n_subspace` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    rank : integer, `rank << min{m,n}`.
        Target rank, i.e., number of components to extract from the data

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    init :  'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure. Default: 'nndsvd'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

    shuffle : boolean, default: False
        If true, randomly shuffle the update order of the variables.

    l2_reg_H : float, (default ``l2_reg_H = 0.1``).
        Amount of ridge shrinkage to apply to `H` to improve conditioning.

    l2_reg_W : float, (default ``l2_reg_W = 0.1``).
        Amount of ridge shrinkage to apply to `W` to improve conditioning.

    l1_reg_H : float, (default ``l1_reg_H = 0.0``).
        Sparsity controlling parameter on `H`.
        Higher values lead to sparser components.

    l1_reg_W : float, (default ``l1_reg_W = 0.0``).
        Sparsity controlling parameter on `W`.
        Higher values lead to sparser components.

    tol : float, default: `tol=1e-5`.
        Tolerance of the stopping condition.

    maxiter : integer, default: `maxiter=200`.
        Number of iterations.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    verbose : boolean, default: `verbose=False`.
        The verbosity level.


    Returns
    -------
    W:  array_like, `(m, rank)`.
        Solution to the non-negative least squares problem.

    H : array_like, `(rank, n)`.
        Solution to the non-negative least squares problem.

    ## zihao:
    loss_out: list of tuple (niter, loss)
    ploss_out: list of tuple (niter, loss in projected space)


    Notes
    -----
    This HALS update algorithm written in cython is adapted from the
    scikit-learn implementation for the deterministic NMF.  We also have
    adapted the initilization scheme.

    See: https://github.com/scikit-learn/scikit-learn


    References
    ----------
    [1] Erichson, N. Benjamin, Ariana Mendible, Sophie Wihlborn, and J. Nathan Kutz.
    "Randomized Nonnegative Matrix Factorization."
    Pattern Recognition Letters (2018).

    [2] Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    [3] C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> import ristretto as ro
    >>> W, H = ro.rnmf(X, rank=2, oversample=0)
    """
    random_state = check_random_state(random_state)
    loss_out = []
    ploss_out = []

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    flipped = False
    if n > m:
        A = A.T
        m, n = A.shape
        flipped = True
## zihao: I have to comment it out and run `python setup.py install` again as it causes some error. 
#    if A.dtype not in _VALID_DTYPES:
#        raise ValueError('A.dtype must be one of %s, not %s'
#                         % (' '.join(_VALID_DTYPES), A.dtype))

    if np.any(A < 0):
        raise ValueError("Input matrix with nonnegative elements is required.")

    start = time.time()
    Q, B = compute_rqb(A, rank, oversample=oversample,
                       n_subspace=n_subspace, random_state=random_state)
    ## zihao: Q: [m, k + oversample]; B: [k + oversample,n]
    runtime = time.time() - start 
    if verbose:    
        print("QB factorization runtime: "+ str(runtime))

    #  Initialization methods for factor matrices W and H
    W, H = _initialize_nmf(A, rank, init=init, eps=1e-6, random_state=random_state)
    ## zihao: W [m,k]; H [k,n]

    Ht = np.array(H.T, order='C') ## zihao: row major
    W_tilde = Q.T.dot(W) ## zihao: W_tilde [k+oversample, k]
    #del A ## zihao: need this to compute loss

    #  Iterate the HALS algorithm until convergence or maxiter is reached
    violation = 0.0
    for niter in range(maxiter):
        start = time.time()

        # Update factor matrix H
        WtW = W.T.dot(W)
        WtW.flat[::rank + 1] += l2_reg_H # adds l2_reg only on the diagonal
        BtW = B.T.dot(W_tilde) - l1_reg_H

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation = _update_cdnmf_fast(Ht, WtW, BtW, permutation) ## zihao: from sklearn

        # Update factor matrix W
        HHt = Ht.T.dot(Ht)
        HHt.flat[::rank + 1] += l2_reg_W # adds l2_reg only on the diagonal

        # Rotate AHt back to high-dimensional space
        BHt = Q.dot(B.dot(Ht)) - l1_reg_W

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation += _update_cdnmf_fast(W, HHt, BHt, permutation)

        # Project W to low-dimensional space
        W_tilde = Q.T.dot(W)

        # Compute stopping condition.
        if niter == 0:
            if violation == 0: break
            violation_init = violation

        if violation / violation_init <= tol:
            break
        runtime = time.time() - start
        if verbose:
            print("n_iter: {:d}     runtime: {:.3e}  rel violation: {:.5e}".format(niter, Decimal(runtime), Decimal(violation / violation_init)))
            if niter % evaluate_every == 0:
                ploss = compute_least_sqr_loss(B,W_tilde.dot(Ht.T))
                loss = compute_least_sqr_loss(A,W.dot(Ht.T))
                print("projected mean square error: {:.5e}".format(Decimal(ploss)))
                print("mean square error          : {:.5e}".format(Decimal(loss)))
                loss_out.append((niter, loss))
                ploss_out.append((niter, ploss))

    # Return factor matrices
    if flipped:
        return(Ht, W.T)
    return(W, Ht.T, loss_out, ploss_out)


def compute_rsvdnmf(A, ranksvd, rank, svdoversample = 20, init='nndsvd', shuffle=False,
                l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
                tol=1e-5, maxiter=200, random_state=None, verbose = 1, evaluate_every = 10):
    start = time.time()
    U,s,Vt = compute_rsvd(A, rank=ranksvd, oversample=svdoversample, n_subspace=2, n_blocks=1, sparse=False, random_state=random_state)
    Ahat = U.dot(np.diag(s)).dot(Vt)
    if verbose:
        print("rsvd and reconstruct time: " + str(time.time() - start))
        nega = Ahat[Ahat <0]
        print("number of negative s in rsvd:" + str(nega.shape[0]))
        print("median of negative s in rsvd:" + str(np.median(nega)))
        print("mean of negative s in rsvd:" + str(np.mean(nega)))
    Ahat[Ahat <0] = 0
    start = time.time()
    W,H,loss,svdloss = compute_nmf_util(Ahat,A,rank=rank,init = 'nndsvd',random_state = 0,tol=1e-10, maxiter= maxiter, verbose = 1,evaluate_every= evaluate_every)
    if verbose:
        print("nmf time: " + str(time.time() - start))
    return W,H,loss, svdloss

def compute_nmf_util(A, A0, rank, init='nndsvd', shuffle=False,
                l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
                tol=1e-5, maxiter=200, random_state=None, verbose = 1, evaluate_every = 10):

    ## only useful for computing loss for compute_rsvdnmf
    random_state = check_random_state(random_state)
    loss_out = [] ## loss against A0
    svdloss_out = [] ## loss against A

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if np.any(A < 0):
        raise ValueError("Input matrix with nonnegative elements is required.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    W, H = _initialize_nmf(A, rank, init=init, eps=1e-6, random_state=random_state)
    Ht = np.array(H.T, order='C')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H and normalize columns
    # ii)  Update low-dimensional factor matrix W
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    violation = 0.0
    for niter in range(maxiter):
        start = time.time()
        # Update factor matrix H with regularization
        WtW = W.T.dot(W)
        WtW.flat[::rank + 1] += l2_reg_H  # adds l2_reg only on the diagonal
        AtW = A.T.dot(W) - l1_reg_H

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation = _update_cdnmf_fast(Ht, WtW, AtW, permutation)

        # Update factor matrix W with regularization
        HHt = Ht.T.dot(Ht)
        HHt.flat[::rank + 1] += l2_reg_W # adds l2_reg only on the diagonal

        AHt = A.dot(Ht) - l1_reg_W

        # compute violation update
        permutation = random_state.permutation(rank) if shuffle else np.arange(rank)
        violation += _update_cdnmf_fast(W, HHt, AHt, permutation) 


        # Compute stopping condition.
        if niter == 0:
            if violation == 0: break
            violation_init = violation

        if violation / violation_init <= tol:
            break
        runtime = time.time() - start
        if verbose:
            print("n_iter: {:d}     runtime: {:.3e}  rel violation: {:.5e}".format(niter, Decimal(runtime), Decimal(violation / violation_init)))

            if niter % evaluate_every == 0:
                loss = compute_least_sqr_loss(A0,W.dot(Ht.T))
                loss_out.append((niter, loss))
                svdloss = compute_least_sqr_loss(A,W.dot(Ht.T))
                svdloss_out.append((niter, svdloss))
                print("mean square error original data           :{:.5e}".format(Decimal(loss)))
                print("mean square error reconstructed data      : {:.5e}".format(Decimal(svdloss)))
    # Return factor matrices
    return W, Ht.T, loss_out, svdloss_out














