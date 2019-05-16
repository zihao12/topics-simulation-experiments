args <- commandArgs(trailingOnly=TRUE)
dataname <- args[1]
# SCRIPT SETTINGS
# ---------------
# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file   <- sprintf("%s.csv", dataname)
init.factors.file  <- sprintf("%s_factors_rough.csv", dataname)
init.loadings.file <- sprintf("%s_loadings_rough.csv", dataname)

cat(sprintf("%s\n",read.counts.file))
cat(sprintf("%s\n",init.factors.file))

# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
factors.out.file  <- sprintf("%s_factors_betanmfr.csv", dataname)
loadings.out.file <- sprintf("%s_loadings_betanmfr.csv", dataname)
timeA.out.file <- sprintf("%s_timesA_betanmfr.csv", dataname)
timeB.out.file <- sprintf("%s_timesB_betanmfr.csv", dataname)
timeCost.out.file <- sprintf("%s_timesCost_betanmfr.csv", dataname)

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
library(NNLM)
source(file.path("..","code","misc.R"))
source(file.path("..","code","utility.R"))
source(file.path("..","code","betanmf_timed.R"))

# LOAD GTEX DATA
# --------------
cat("Loading GTEx data.\n")
read.counts.file <- file.path(data.dir,read.counts.file)
counts <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d count matrix.\n",nrow(counts),ncol(counts)))
cat(sprintf("class %s; mode: %s; storage.mode: %s .\n",class(counts),mode(counts), storage.mode(counts)))



# LOAD INITIAL ESTIMATES
# ----------------------
cat("Loading initial estimates of factors and loadings.\n")
init.factors.file  <- file.path(data.dir,init.factors.file)
init.loadings.file <- file.path(data.dir,init.loadings.file)
F0                 <- read.csv.matrix(init.factors.file)
L0                 <- read.csv.matrix(init.loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F0),ncol(F0)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L0),ncol(L0)))

# Get the number of factors ("topics").
K <- ncol(F0)

# RUN NMF OPTIMIZATION METHOD
# ---------------------------
cat("Fitting Poisson topic model using nnmf.\n")
timing <- system.time(
	fit <- betanmf_timed(counts, L0, t(F0), numiter=20)
)

cat(sprintf("Computation took %0.2f seconds.\n",timing["elapsed"]))

cat("Compute loglikelihood\n")
F <- t(fit$B)
L <- fit$A
time = list(time_A = fit$time_A, time_B = fit$time_B, time_cost = fit$time_cost)

out <- compute_ll(t(counts),F,t(L))
cat(sprintf("method type: %s\n
        poisson_ll :%0.12f\n
        multinom_ll:%0.12f\n",out$type, out$pois_ll,out$multinom_ll))


# WRITE NNMF RESULTS TO FILE
# --------------------------
cat("Writing results to file.\n")
# factors.out.file  <- file.path(out.dir,factors.out.file)
# loadings.out.file <- file.path(out.dir,loadings.out.file)
timeA.out.file  <- file.path(out.dir,timeA.out.file)
timeB.out.file  <- file.path(out.dir,timeB.out.file)
timeCost.out.file  <- file.path(out.dir,timeCost.out.file)

# write_csv(as.data.frame(F),factors.out.file,col_names = FALSE)
# write_csv(as.data.frame(L),loadings.out.file,col_names = FALSE)
write_csv(as.data.frame(time$time_A),timeA.out.file,col_names = FALSE)
write_csv(as.data.frame(time$time_B),timeB.out.file,col_names = FALSE)
write_csv(as.data.frame(time$time_cost),timeCost.out.file,col_names = FALSE)


# SESSION INFO
# ------------
cat("Session info:\n")
sessionInfo()
