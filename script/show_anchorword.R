args <- commandArgs(trailingOnly=TRUE)
method = args[1]
dataname <- args[2]

source("../code/misc.R")
library(readr)
library(ggplot2)

data.dir  <- file.path("../../topics-simulation-bigdata","output")

labelname = paste0(dataname, "_label.csv")
loadings.file = paste0(dataname,"_loadings_",method,".csv.gz")
factors.file = paste0(dataname,"_factors_",method,".csv.gz")
plot.file = paste0(dataname,"_loadings_plot_",method,".pdf")

# LOAD ESTIMATES
# ----------------------
cat("Loading estimates of factors and loadings.\n")
factors.file  <- file.path(data.dir,factors.file)
loadings.file <- file.path(data.dir,loadings.file)
F               <- read.csv.matrix(factors.file)
L                <- read.csv.matrix(loadings.file)
cat(sprintf("Loaded %d x %d factors matrix, ",nrow(F),ncol(F)))
cat(sprintf("and %d x %d loadings matrix.\n",nrow(L),ncol(L)))

# from poisson to multinomial model
if(!(method %in% c("maptpx", "skdlda"))){
        cat(sprintf("Transform : poisson2multinom\n"))
        out <- poisson2multinom(F,L)
        F   <- out$F
        L   <- out$L
        rm(out)
}


ntop_total = 20
total_sort = sort(rowSums(F), decreasing = T, index.return = T)
for(i in 1:ntop_total){
  cat(sprintf("################TOP %d word ################\n", i))
  cat(sprintf("gene %d\n",total_sort$ix[i]))
  out = round(F[total_sort$ix[i],],5)
  cat(sprintf("%s\n",paste(sort(out, decreasing = T), collapse = " ")))
  #cat(print(sort(out, decreasing = T)))
}


for(k in 1:ncol(F)){
  cat(sprintf("################TOPIC %d ################\n", k))
  topic = F[,k]
  ntop = 20
  lst <- sort(topic, index.return=TRUE, decreasing=TRUE)
  for(i in 1:ntop){
    cat(sprintf("gene %d;\t  prob %6f\n", lst$ix[i], lst$x[i]))
  }
}
