# Use maptpx to fit a multinomial topic model to the GTEx data. 
#
# Note that I am using am using a slightly modified version of maptpx
# which is available at https://github.com/pcarbo/maptpx.
#
# Also note that maptpx is more memory intensive than some of the
# other methods I tested; see fit_gtex_maptpx.sbatch for the Slurm
# settings used on the RCC cluster.
#

# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file   <- "gtex_simulation_nnlm.csv"
init.factors.file  <- "gtex_simulation_rough_factors.csv"
init.loadings.file <- "gtex_simulation_rough_loadings.csv"

# read.counts.file  <- "test.csv"
# init.factors.file <- "test_factors.csv"
# init.loadings.file <- "test_loadings.csv"


# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
factors.out.file  <- "gtex_simulation_factors_maptpx.csv"
loadings.out.file <- "gtex_simulation_loadings_maptpx.csv"

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
library(maptpx)
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

# Get the estimated factors and loadings.
F <- fit$theta
L <- fit$omega

cat("Compute loglikelihood of rough fit\n")
out = compute_ll(t(counts),F,t(L))
cat(sprintf("method type: %s\n 
	poisson_ll :%0.12f\n 
	multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))


# f <- multinom2poisson_ll(t(counts),F,t(L))
# cat(sprintf("Poisson likelihood : %0.12f\n",f))

# # Compute the multinomial likelihood for the maptpx solution.
# f <- loglik.multinom(counts,F,L)
# cat(sprintf("Multinomial likelihood : %0.12f\n",f))

# WRITE MAPTPX RESULTS TO FILE
# ----------------------------
cat("Writing results to file.\n")
factors.out.file  <- file.path(out.dir,factors.out.file)
loadings.out.file <- file.path(out.dir,loadings.out.file)
write_csv(as.data.frame(F),factors.out.file,col_names = FALSE)
write_csv(as.data.frame(L),loadings.out.file,col_names = FALSE)

# SESSION INFO
# ------------
cat("Session info:\n")
sessionInfo()
