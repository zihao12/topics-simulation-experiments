# Simulate count data from poisson model
simulate_pois <- function(n,p,k, seed = 0){
    set.seed(seed)
    A = matrix(rnorm(p*k, 0,1), nrow = p)
    W = matrix(rnorm(k*n, 0,1), nrow = k)
    lam = exp(A) %*% exp(W)
    X = matrix(rpois(n*p,lam), nrow = p)
    return(list(X = X, lam = lam))
}

## compute poisson loglikelihood
pois_lk <- function(X,lam){
    return(sum(dpois(X,lam, log= TRUE)))
}

## genertae data from oracle
generateForacle <- function(A,W, seed = 0){
    set.seed(seed)
    Lam = A %*% W
    p = nrow(Lam)
    n = ncol(Lam)
    X = matrix(rpois(n*p,Lam), nrow = p)
    return(X)
}

## compute poisson loglikelihood from of multinomial model
multinom2poisson_ll <- function(X,A,W){
    Lam  = A %*% W %*% diag(colSums(X))
    ll = pois_lk(X,Lam)
}

## compute both multinomial and poisson logliklihood for both multinomial and poisson model
compute_ll <- function(X,A,W, e = .Machine$double.eps){
  p = nrow(X)
  n = ncol(X)
  if(mean(colSums(A %*% W)) < 1.1){ ## this is multinomial model
    theta = A %*% W ## parameter for multinom distribution
    multinom_ll = sum(X * log(theta + e))
    lam = theta %*% diag(colSums(X))
    pois_ll = sum(dpois(X,lam, log= TRUE))/(n*p)
    return(list(type = "multinom", multinom_ll = multinom_ll, pois_ll = pois_ll))
  }
  else{
    pois_ll = sum(dpois(X,A %*% W, log= TRUE))/(n*p)
    Ahat = A %*% diag(colSums(A)^(-1))
    Wtild = diag(colSums(A)) %*% W
    What = Wtild %*% diag(colSums(Wtild)^(-1))
    theta = Ahat %*% What
    multinom_ll = sum(X * log(theta + e))
    return(list(type = "poisson", multinom_ll = multinom_ll, pois_ll = pois_ll))
  }
}


