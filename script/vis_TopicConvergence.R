## utility functions for plotting
args <- commandArgs(trailingOnly=TRUE)
method = args[1]
dataname = args[2]
k = as.integer(args[3])
type = args[4]

data.dir = "../../topics-simulation-bigdata/output"
# dataname = "gtex"
# method = "betanmf"
#type = "factors"
#k = 5

init_stage = 25;
iter_init = 2;
mid_stage = 2;
iter_mid = 100;
iters = c(iter_init*(1:init_stage), iter_mid*(1:mid_stage) + init_stage*iter_init)
iters = c(0,iters)


source("../code/misc.R")
library(readr)

## function

## read loadings or factors
get_lf <- function(data.dir,dataname, method_iter,type){
  ## type is "factors" or "loadings"
  loadings.file = sprintf("%s_%s_%s.csv", dataname,type, method_iter)
  loadings.file <- file.path(data.dir,loadings.file)
  out <- read.csv.matrix(loadings.file)
  return(out)
}

compare_topic <- function(F1,F2,dataname,method1, method2, k){
  # F1 = get_lf(data.dir,dataname, method1, "factors")
  # F2 = get_lf(data.dir,dataname, method2, "factors")
  topic1 = F1[,k]
  topic2 = F2[,k]
  p = plot(topic1, topic2, xlab = paste0(method1, "_topic_", k), ylab = paste0(method2, "_topic_", k),
           main = paste0(method1," VS ", method2, " in topic ", k))
  return(p)
}


plot.file = paste0("TopicPlot_",method,"_", type,"_topic",k,".pdf")
pdf(file.path("../docs", plot.file), compress = T)

method_iter1 = paste0(method, "_iter", iters[length(iters)])
F1 = get_lf(data.dir,dataname, method_iter1,type)
for(i in iters){
  print(paste0("plot of iter ",i))
  method_iter2 = paste0(method, "_iter", i)
  F2 = get_lf(data.dir,dataname, method_iter2,type)
  p = compare_topic(F1,F2,dataname,method_iter1, method_iter2, k)
  print(p)
}
dev.off()


