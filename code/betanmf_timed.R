# This function decomposes the input matrix X = A*B by nonnegative
# matrix factorization (NMF) based on the beta-divergence criterion
# (negative Poisson log-likelihood) and multiplicative update
# rules. All entries of initial estimates A and B should be
# positive. This is adapted from the MATLAB code by D. Kitamura
# (http://d-kitamura.net).

cost <- function(X,AB, e){
        return(sum(AB - X * log(AB + e)))
}

update_A_timed <- function(X,A,B,E,e){
  # original update is:
  #A <- A * (((X / (A %*% B)) %*% t(B)) / (E %*% t(B)))
  #A <- pmax(A,e)
  time = c()

  ## step 1
  start = proc.time()
  M = A %*% B
  runtime = proc.time() - start
  time = c(time, runtime[[3]])

  ## step 2
  start = proc.time()
  M = X / M
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 3
  start = proc.time()
  M = M %*% t(B)
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 4
  start = proc.time()
  D = E %*% t(B)
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 5
  start = proc.time()
  M = M/D
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 6
  start = proc.time()
  A = A*M
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 7
  start = proc.time()
  A <- pmax(A,e)
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 
  return(list(A = A, time = time))
}


update_B_timed <- function(X,A,B,E,e){
  # original update is:
  # B <- B * ((t(A) %*% (X / (A %*% B))) / (t(A) %*% E))
  # B <- pmax(B,e)
  time = c()

  ## step 1
  start = proc.time()
  M = A %*% B
  runtime = proc.time() - start
  time = c(time, runtime[[3]])

  ## step 2
  start = proc.time()
  M = X / M
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 3
  start = proc.time()
  M = t(A) %*% M
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 4
  start = proc.time()
  D = t(A) %*% E
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 5
  start = proc.time()
  M = M/D
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 6
  start = proc.time()
  B = B*M
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 

  ## step 7
  start = proc.time()
  B <- pmax(B,e)
  runtime = proc.time() - start
  time = c(time, runtime[[3]]) 
  return(list(B = B, time = time))
}






betanmf_timed <- function (X, A, B, numiter = 1000, e = .Machine$double.eps, verbose = TRUE) {
  if (inherits(X,"matrix"))
    X <- as.matrix(X)
  n <- nrow(X)
  m <- ncol(X)
  E <- matrix(1,n,m)
  # progress <- data.frame(iter = 1:numiter,objective = 0,
  #                        max.diff = 0,timing = 0)

  time_A = c()
  time_B = c()
  time_cost = c()
      
  # Repeat until we reach the number of requested iterations.
  if (verbose)
    cat("iter         objective max.diff\n")
  for (i in 1:numiter) {

    # Save the current estimates of the factors and loadings.
    A0 <- A
    B0 <- B

    

    # Update the loadings ("activations").
    #A <- A * (((X / (A %*% B)) %*% t(B)) / (E %*% t(B)))
    #A <- pmax(A,e)
    A_up <- update_A_timed(X,A,B,E,e)
    A <- A_up$A
    time_A = rbind(time_A, A_up$time)

    # Update the factors ("basis vectors").
    # B <- B * ((t(A) %*% (X / (A %*% B))) / (t(A) %*% E))
    # B <- pmax(B,e)
    B_up <- update_B_timed(X,A,B,E,e)
    B <- B_up$B
    time_B = rbind(time_B, B_up$time)
  

    # Compute the value of the objective (cost) function at the
    # current estimates of the factors and loadings.
    start = proc.time()
    f <- cost(X,A %*% B,e)
    time = proc.time() - start
    time_cost = rbind(time_cost, time[[3]])
    d <- max(max(abs(A - A0)),max(abs(B - B0)))
    # progress[i,"objective"] <- f
    # progress[i,"max.diff"]  <- d
    # progress[i,"timing"]    <- timing["elapsed"]
    if (verbose)
      cat(sprintf("%4d %0.10e %0.2e\n",i,f,d))
  }

  return(list(A = A,B = B,value = f,time_A = time_A, time_B = time_B, time_cost = time_cost))
}
