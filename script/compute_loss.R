## 
#data.dir           <- file.path("../../topics-simulation-bigdata","output")
data.dir           <- file.path("../../topics_bigdata","output")
read.counts.file <- "gtex.csv"

library(readr)
source("../code/misc.R")
source("../code/utility.R")

# LOAD GTEX DATA
# --------------
cat("Loading GTEx data.\n")
read.counts.file <- file.path(data.dir,read.counts.file)
counts <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d count matrix.\n",nrow(counts),ncol(counts)))

# LOAD INITIAL ESTIMATES
# ----------------------
cat("rnmfhals.\n")
init.factors.file  <- "gtex_factors_rnmfhals.csv"
init.loadings.file <- "gtex_loadings_rnmfhals.csv"
init.factors.file  <- file.path(data.dir,init.factors.file)
init.loadings.file <- file.path(data.dir,init.loadings.file)
F0                 <- read.csv.matrix(init.factors.file)
L0                 <- read.csv.matrix(init.loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F0),ncol(F0)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L0),ncol(L0)))
out = compute_ll_eps(t(counts), F0, t(L0))
print(out)

# LOAD INITIAL ESTIMATES
# ----------------------
cat("nnmf_F.\n")
init.factors.file  <- "gtex_factors_nnmf_F.csv"
init.loadings.file <- "gtex_loadings_nnmf_F.csv"
init.factors.file  <- file.path(data.dir,init.factors.file)
init.loadings.file <- file.path(data.dir,init.loadings.file)
F1                 <- read.csv.matrix(init.factors.file)
L1                 <- read.csv.matrix(init.loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F1),ncol(F1)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L1),ncol(L1)))
out = compute_ll_eps(t(counts), F1, t(L1))
print(out)
