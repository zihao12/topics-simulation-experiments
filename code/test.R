## 
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file  <- "test.csv"
init.factors.file <- "test_factors.csv"
init.loadings.file <- "test_loadings.csv"

library(readr)
library(NNLM)
source(file.path(".","misc.R"))
source(file.path(".","utility.R"))

# LOAD GTEX DATA
# --------------
cat("Loading GTEx simulation data.\n")
read.counts.file <- file.path(data.dir,read.counts.file)
counts <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d count matrix.\n",nrow(counts),ncol(counts)))

# LOAD INITIAL ESTIMATES
# ----------------------
cat("Loading estimates of factors and loadings.\n")
init.factors.file  <- file.path(data.dir,init.factors.file)
init.loadings.file <- file.path(data.dir,init.loadings.file)
F0                 <- read.csv.matrix(init.factors.file)
L0                 <- read.csv.matrix(init.loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F0),ncol(F0)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L0),ncol(L0)))

## test
#pve = compute_pve(t(counts), F0, t(L0))
#cat(sprintf("mse for each F,L pair:\n"))
##cat(sprintf(pve$x))
##print(pve$x)
#for(x in pve$x){
#	cat(sprintf("%d\n", as.integer(x)))
#}
#cat(sprintf("\n"))
#cat(sprintf("order is:\n"))
#print(pve$ix)
#

out = compute_pve(t(counts), F0, t(L0))












