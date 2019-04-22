## print summary statistics of matrix
## input is name of a file saved in csv format
args = commandArgs(trailingOnly=TRUE)
name = args[1]

data.dir           <- file.path("../../topics-simulation-bigdata","output")
read.X.file   <- name

# SET UP ENVIRONMENT
# ------------------
# Load packages and function definitions.
library(readr)
#library(NNLM)
source(file.path("..","code","misc.R"))

# LOAD GTEX DATA
# --------------
cat(sprintf("Loading data from %s\n", name))
read.counts.file <- file.path(data.dir,read.X.file)
X <- read.csv.matrix(read.counts.file)
cat(sprintf("Loaded %d x %d matrix.\n",nrow(X),ncol(X)))

cat(sprintf("mean: %f; median: %f in log10 scale\n", log10(mean(X)), log10(median(X))))
# for(i in 1:ncol(X)){
# 	cat(sprintf("summary for column %d\n", i))
# 	cat(quantile(X[,i], seq(0,1,0.1)),"\n",sep="\t")
# }

