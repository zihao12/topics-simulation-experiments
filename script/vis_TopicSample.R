## plot how each "topic" is expressed in a subgroup of samples 
args <- commandArgs(trailingOnly=TRUE)
method = args[1]
dataname <- args[2]

source("../code/misc.R")
library(readr)
library(ggplot2)

data.dir  <- file.path("../../topics-simulation-bigdata","output")

labelname = paste0(dataname, "_label.csv")
loadings.file = paste0(dataname,"_loadings_",method,".csv")
factors.file = paste0(dataname,"_factors_",method,".csv")
plot.file = paste0(dataname,"_",method,"_topics&sample_plot.pdf")


# LOAD LABELS
# ----------------------
cat("Loading labels.\n")
samples_id = read.csv(file.path(data.dir, labelname))[,1]

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
if(method %in% c("ccd", "admm", "betanmf","nnmf","skdnmf","nndsvd")){
	cat(sprintf("Transform : poisson2multinom\n"))
	out <- poisson2multinom(F,L)
	F   <- out$F
	L   <- out$L
	rm(out)
}

L = data.frame(L, row.names = 1:nrow(L))
L["label"] = samples_id

data = L
## data has K+1 columns, first K are topic weights (sum to 1); last is label

pdf(file.path("../docs", plot.file))
for(k in 1:(ncol(L)-1)){
  topic = data[,c(k,ncol(L))]
  names(topic) = c("topic","label")
  p <- ggplot(topic, aes(x = label, y = topic,group = label)) +
    geom_point()+
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    ggtitle(paste0(dataname,"_",method,"_topic ", k))
  print(p)
}
dev.off()


