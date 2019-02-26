## prepare some test data 
## generate a test count data with n = 100, p = 300, k = 10
## fit with NNLM and get factors and loadings 


# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
count.out.file <- "test.csv"
factors.out.file  <- "test_factors.csv"
loadings.out.file  <- "test_loadings.csv"

# Environment
library(readr)
library(NNLM)
source(file.path("..","code","misc.R"))
source(file.path("..","code","utility.R"))
set.seed(12345)

##
n = 100
p = 300
k = 10
out = simulate_pois(n,p,k, seed = 12345)
X = out$X
rm(out)

timing <- system.time(
  fit <- nnmf(t(X),k,method = "scd",loss = "mkl",rel.tol = 1e-5,
              n.threads = 0,max.iter = 30,inner.max.iter = 4,trace = 1,
              verbose = 2)
  )
A = t(fit$H)
W = t(fit$W)

out = compute_ll(X,A,W)
cat(sprintf("method type: %s\n 
	poisson_ll : %0.12f\n 
	multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))

# WRITE NNMF RESULTS TO FILE
# --------------------------
cat("Writing results to file.\n")
count.out.file  <- file.path(out.dir,count.out.file)
factors.out.file  <- file.path(out.dir,factors.out.file)
loadings.out.file <- file.path(out.dir,loadings.out.file)
write_csv(as.data.frame(t(X)),count.out.file,col_names = FALSE)
write_csv(as.data.frame(A),factors.out.file,col_names = FALSE)
write_csv(as.data.frame(t(W)),loadings.out.file,col_names = FALSE)

