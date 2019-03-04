
poisson2multinom <- function (F, L) {
  L <- t(t(L) * colSums(F))
  s <- rowSums(L)
  L <- L / s
  F <- scale.cols(F)
  return(list(F = F,L = L,s = s))
}

multinom2poisson <- function(counts,F,L){
  s = rowSums(counts)
  L = diag(s) %*% L
  return(list(L = L, F = F))
}

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
    E = .Machine$integer.max ## too large poisson mean will cause NAs!
    set.seed(seed)
    Lam = A %*% W
    Lam[which(Lam > E)] = E
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
  else{ ## this is poisson model
    pois_ll = sum(dpois(X,A %*% W, log= TRUE))/(n*p)
    out = poisson2multinom(A,t(W))
    Ahat = out$F
    What = t(out$L)
    theta = Ahat %*% What
    multinom_ll = sum(X * log(theta + e))
    return(list(type = "poisson", multinom_ll = multinom_ll, pois_ll = pois_ll))
  }
}

compute_ll_eps <- function(X,A,W, e = .Machine$double.eps){
  p = nrow(X)
  n = ncol(X)
  if(mean(colSums(A %*% W)) < 1.1){ ## this is multinomial model
    theta = A %*% W ## parameter for multinom distribution
    multinom_ll = sum(X * log(theta + e))
    lam = theta %*% diag(colSums(X))
    pois_ll = sum(dpois(X,lam + e, log= TRUE))/(n*p)
    return(list(type = "multinom", multinom_ll = multinom_ll, pois_ll = pois_ll))
  }
  else{ ## this is poisson model
    pois_ll = sum(dpois(X,A %*% W + e, log= TRUE))/(n*p)
    out = poisson2multinom(A,t(W))
    Ahat = out$F
    What = t(out$L)
    theta = Ahat %*% What
    multinom_ll = sum(X * log(theta + e))
    return(list(type = "poisson", multinom_ll = multinom_ll, pois_ll = pois_ll))
  }
}


poiss_ll_matrix <- function(X,A,W){
  p = nrow(X)
  n = ncol(X)
  if(mean(colSums(A %*% W)) < 1.1){ ## this is multinomial model
    out = multinom2poisson(X,F,t(L))
    A = out$F
    W = t(out$L)    
  }
  p_matrix = dpois(X, A %*% W, log = TRUE)
  return(p_matrix)
}

percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

compute_mse <- function(X,A,W){
	p = nrow(X)
	n = ncol(X)
	return(sum((X-A %*% W)^2)/(n*p))
}













