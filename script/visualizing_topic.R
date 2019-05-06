## select a topic, 

args <- commandArgs(trailingOnly=TRUE)
k <- as.integer(args[1])
method = "betanmf"
dataname = "gtex"
k = 5

init_stage = 25;
iter_init = 2;
mid_stage = 10;
iter_mid = 100;
iters = c(iter_init*(1:init_stage), iter_mid*(1:mid_stage) + init_stage*iter_init)

## environment
source("../code/misc.R")
library(readr)
library(ggplot2)

data.dir  <- file.path("../../topics-simulation-bigdata","output")
labelname = paste0(dataname, "_label.csv")
#factors.file = paste0(dataname,"_factors_",method,".csv")
plot.file = paste0(dataname,"_loadings_plot_",method,"_topic_",k,".pdf")


## function
get_loading <- function(data.dir,dataname, method, iter){
  method_iter = sprintf("%s_iter%d",method,iter)
  loadings.file = sprintf("%s_loadings_%s.csv", dataname, method_iter)
  loadings.file <- file.path(data.dir,loadings.file)
  L <- read.csv.matrix(loadings.file)
  return(L)
}


# LOAD LABELS
# ----------------------
cat("Loading labels.\n")
samples_id = read.csv(file.path(data.dir, labelname))[,1]

pdf(file.path("../docs", plot.file), compress = T)
for(iter in iters){
  L = get_loading(data.dir,dataname, method, iter)
  L = data.frame(L, row.names = 1:nrow(L))
  L["label"] = samples_id
  data = L
  topic = data[,c(k,ncol(L))]
  names(topic) = c("topic","label")
  p <- ggplot(topic, aes(x = label, y = topic,group = label)) +
    geom_point()+
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    ggtitle(paste0("topic ", k, "iter ", iter))
  print(p)
}
dev.off()



