## Generate new data from GTEx data by first getting fitted Lambda from topic modeling methods, then generate new data out of Lambda
## The data has been fitted by some previous experiments, so here use the fitted model to generate data


# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.factors.file   <- "gtex_factors_nnmf.csv.gz"
read.loadings.file   <- "gtex_loadings_nnmf.csv.gz"

# read.factors.file   <- "test_factors.csv"
# read.loadings.file   <- "test_loadings.csv"


# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
count.out.file  <- "gtex_simulation_nnlm.csv"
factors.out.file  <- "gtex_simulation_rough_factors.csv"
loadings.out.file <- "gtex_simulation_rough_loadings.csv"

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
library(NNLM)
source(file.path("..","code","misc.R"))
source(file.path("..","code","utility.R"))

# LOAD FITTED MODEL
# --------------
cat("Loading model.\n")
read.factors.file <- file.path(data.dir,read.factors.file)
read.loadings.file <- file.path(data.dir,read.loadings.file)

F <- read.csv.matrix(read.factors.file)
L <- read.csv.matrix(read.loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F),ncol(F)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L),ncol(L)))

K <- ncol(F)
# GENERATE DATA
# --------------
cat("Generate data from Oracle\n")
X = generateForacle(F,t(L), seed = 12345)
cat(sprintf("Generated %d x %d transpose count matrix\n",nrow(X),ncol(X)))

# COMPUTE LOGLIKELIHOOD
# --------------
cat("Compute loglikelihood of rough fit\n")
out = compute_ll(X,F,t(L))
cat(sprintf("method type: %s\n 
	poisson_ll :%0.12f\n 
	multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))



# RUN ROUGH FIT
# --------------
cat("Run initial fit\n")
set.seed(12345)
timing <- system.time(
  fit <- nnmf(t(X),K,method = "scd",loss = "mkl",rel.tol = 1e-8,
              n.threads = 0,max.iter = 10,inner.max.iter = 4,trace = 1,
              verbose = 2)
  )
cat(sprintf("Computation took %0.2f seconds.\n",timing["elapsed"]))

# COMPUTE LOGLIKELIHOOD
# --------------
cat("Compute loglikelihood of rough fit\n")
A = t(fit$H)
W = t(fit$W)

out = compute_ll(X,A,W)
cat(sprintf("method type: %s\n 
	poisson_ll :%0.12f\n 
	multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))

# WRITE NNMF RESULTS TO FILE
# --------------------------
cat("Writing results to file.\n")
count.out.file  <- file.path(out.dir,count.out.file)
write_csv(as.data.frame(t(X)),count.out.file,col_names = FALSE)

factors.file  <- file.path(out.dir,factors.out.file)
loadings.file <- file.path(out.dir,loadings.out.file)
F <- t(fit$H)
F <- round(F,digits = 4)
F <- as.data.frame(F)
L <- fit$W
L <- round(L,digits = 4)
L <- as.data.frame(L)
write_csv(F,factors.file,col_names = FALSE)
write_csv(L,loadings.file,col_names = FALSE)


# SESSION INFO
# ------------
cat("Session info:\n")
sessionInfo()


