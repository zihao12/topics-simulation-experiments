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

args <- commandArgs(trailingOnly=TRUE)
dataname <- args[1]
# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file   <- sprintf("%s.csv", dataname)
init.factors.file  <- sprintf("%s_factors_rough.csv", dataname)

cat(sprintf("%s\n",read.counts.file))
cat(sprintf("%s\n",init.factors.file))

# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
factors.out.file  <- sprintf("%s_factors_maptpx.csv", dataname)
loadings.out.file <- sprintf("%s_loadings_maptpx.csv", dataname)

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
library(maptpx)
source(file.path("..","code","misc.R"))
source(file.path("..","code","utility.R"))

# LOAD GTEX DATA
# --------------
cat("Loading GTEx data.\n")
read.counts.file <- file.path(data.dir,read.counts.file)
counts <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d count matrix.\n",nrow(counts),ncol(counts)))

# LOAD INITIAL ESTIMATES
# ----------------------
cat("Loading initial estimates of factors and loadings.\n")
init.factors.file <- file.path(data.dir,init.factors.file)
F0 <- read.csv.matrix(init.factors.file)
F0 <- scale.cols(F0 + 0.01)
cat(sprintf("Loaded %d x %d factors matrix.\n",nrow(F0),ncol(F0)))

# Get the number of factors ("topics").
K <- ncol(F0)

# RUN MAPTPX OPTIMIZATION METHOD
# ------------------------------
# Note that maptpx is computing a maximum a posteriori estimate of L,
# not a maximum-likelihood estimate. (By setting shape = 1, we recover
# a maximum-likelihood estimate of F.)
cat("Fitting multinomial topic model using maptpx.\n")
timing <- system.time(
  fit <- topics(counts,K,shape = 1,initopics = F0,tol = 1e-4,
                tmax = 100,verb = 2))
cat(sprintf("Computation took %0.2f seconds.\n",timing["elapsed"]))

cat("Compute loglikelihood\n")
F <- fit$theta
L <- fit$omega

out <- compute_ll(t(counts),F,t(L))
cat(sprintf("method type: %s\n
        poisson_ll :%0.12f\n
        multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))


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
