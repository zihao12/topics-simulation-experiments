import numpy as np
from scipy.stats import poisson
## utility function 
def compute_loglik(X,A,W):
	p,n = X.shape
	e = np.finfo(float).eps
	if A.dot(W).sum(axis = 0).mean() < 2: ## multinomial model
		method = 'multinom'
		theta = A.dot(W)
		multinom_ll = (X * np.log(theta + e)).sum()
		lam = theta.dot(np.diag(X.sum(axis = 0)))	
		poisson_ll = poisson.logpmf(X,lam).sum()/(n*p)	
	else:
		method = "poisson"
		lam = A.dot(W)
		poisson_ll = poisson.logpmf(X,lam).sum()/(n*p)

		ca = A.sum(axis = 0)
		Ahat = A.dot(np.diag(ca**-1))
		What = np.diag(ca).dot(W)
		cw = What.sum(axis = 0)
		What = What.dot(np.diag(cw**-1))
		theta = Ahat.dot(What)
		multinom_ll = (X * np.log(theta + e)).sum()

	return {"type":method, "poisson_ll":poisson_ll, "multinom_ll":multinom_ll}


def compute_least_sqr_loss(X,Lam):
    p, n = X.shape
    return (np.square((X - Lam))).sum()/(n*p)
