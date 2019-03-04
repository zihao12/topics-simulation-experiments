## compute poisson loglikelihood matrix
args = commandArgs(trailingOnly=TRUE)

method = args[1]
print(method)


# These variables specify the names of the input files.
data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.counts.file   <- "gtex_simulation_nnlm.csv"
factors.file  <- paste0("gtex_factors_",method,".csv.gz")
loadings.file <- paste0("gtex_loadings_",method,".csv.gz")

# These variables specify the names of the output files.
out.dir           <- file.path("../../topics-simulation-bigdata","output")
#poiss.out.file  <- paste0("gtex_",method,"_poissll.csv")
poiss.out.file  <- "gtex_simulation_oracle_poissll.csv"

print(poiss.out.file)


# ## FOR TESTING
# read.counts.file   <- "test.csv"
# factors.file  <- "test_factors.csv"
# loadings.file <- "test_loadings.csv"
# poiss.out.file  <- "test_poissll.csv"

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
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
cat("Loading estimates of factors and loadings.\n")
factors.file  <- file.path(data.dir,factors.file)
loadings.file <- file.path(data.dir,loadings.file)
F0                 <- read.csv.matrix(factors.file)
L0                 <- read.csv.matrix(loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F0),ncol(F0)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L0),ncol(L0)))

# COMPUTE POISSON LL
# -----------------------
cat("Compute Poisson ll matrix\n")
p_matrix = poiss_ll_matrix(t(counts),F0,t(L0))

# Writing results to file
# -----------------------
cat("Writing results to file.\n")
poiss.out.file   <- file.path(out.dir,poiss.out.file )
write_csv(as.data.frame(round(p_matrix,4)),poiss.out.file,col_names = FALSE)
