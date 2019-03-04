# Fit a non-megative matrix factorization to the GTEx data using the
# sequential coordinate descent algorithm implemented in the NNLM
# package. The SCD algorithm appears to be based on the methods
# described in:
#
#   Li, Lebanon & Park. Fast Bregman divergence NMF using Taylor
#   expansion and coordinate descent. KDD 2012.
#
# See fit_gtex_nnmf.sbatch for SLURM settings used on the RCC
# cluster.

# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file   <- "gtex_simulation_nnlm.csv"
init.factors.file  <- "gtex_simulation_rough_factors.csv"
init.loadings.file <- "gtex_simulation_rough_loadings.csv"

# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
factors.out.file  <- "gtex_simulation_factors_nnmf_F.csv"
loadings.out.file <- "gtex_simulation_loadings_nnmf_F.csv"

#read.counts.file  <- "test.csv"
#init.factors.file <- "test_factors.csv"
#init.loadings.file <- "test_loadings.csv"
#factors.out.file  <- "test_factors_nnmf.csv"
#loadings.out.file <- "test_loadings_nnmf.csv"


# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
library(NNLM)
source(file.path("..","code","misc.R"))
source(file.path("..","code","utility.R"))

# LOAD GTEX DATA
# --------------
cat("Loading GTEx simulation data.\n")
read.counts.file <- file.path(data.dir,read.counts.file)
counts <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d count matrix.\n",nrow(counts),ncol(counts)))

# LOAD INITIAL ESTIMATES
# ----------------------

# Get the number of factors ("topics").
K <- 20

# RUN NMF OPTIMIZATION METHOD
# ---------------------------
cat("Fitting Poisson topic model using nnmf.\n")
timing <- system.time(
  fit <- nnmf(counts,K,init = NULL,method = "scd",
              loss = "mse",rel.tol = 1e-8,n.threads = 0,max.iter = 200,
              inner.max.iter = 4,trace = 1,verbose = 2))
cat(sprintf("Computation took %0.2f seconds.\n",timing["elapsed"]))

# Convert the Poisson model parameters to the parameters for the
# multinomial model.

# COMPUTE LOGLIKELIHOOD
# ---------------------------
F = t(fit$H)
L = fit$W

cat("Compute loglikelihood and mse\n")
out = compute_ll(t(counts),F,t(L))
mse = compute_mse(t(counts),F,t(L))
cat(sprintf("method type: %s\n 
	poisson_ll :%0.12f\n 
	multinom_ll:%0.12f\n
	mse        :%0.12f\n",out$type, out$pois_ll,out$multinom_ll, mse))

# WRITE NNMF RESULTS TO FILE
# --------------------------
cat("Writing results to file.\n")
factors.out.file  <- file.path(out.dir,factors.out.file)
loadings.out.file <- file.path(out.dir,loadings.out.file)
write_csv(as.data.frame(F),factors.out.file,col_names = FALSE)
write_csv(as.data.frame(L),loadings.out.file,col_names = FALSE)

# SESSION INFO
# ------------
cat("Session info:\n")
sessionInfo()
