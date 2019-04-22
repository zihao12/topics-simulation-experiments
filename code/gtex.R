# This file implements functions to import, prepare and analyze the
# GTEx data.

# This function imports and prepares the GTEx data for analysis in
# R. The return value is list with two list elements: (1) a data frame
# containing the sample attributes (specifically, tissue labels), and
# (2) an n x p matrix containing gene expression data (read counts),
# where n is the number of samples, and p is the number of genes.
#
# This function is known to work for these two files downloaded from
# the GTEx Portal:
#
#   GTEx_v7_Annotations_SampleAttributesDS.txt
#   GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct.gz
# 
read.gtex.data <- function (samples.file, read.counts.file) {

  # Read the sample attributes from the tab-delimited file.
  suppressMessages(samples <- read_delim(samples.file,delim = "\t",
                                         progress = FALSE))
  class(samples)    <- "data.frame"
  rownames(samples) <- samples$SAMPID
  samples           <- samples[c("SMTS","SMTSD")]
  names(samples)    <- c("general","specific")

  # Simplify the tissue labels.
  x <- samples$specific
  x <- gsub(" - "," ",x,samples$specific,fixed = TRUE)
  x <- gsub("(","",x,samples$specific,fixed = TRUE)
  x <- gsub(")","",x,samples$specific,fixed = TRUE)
  x <- tolower(x)
  samples <- transform(samples,
                       general  = tolower(general),
                       specific = tolower(specific))

  # Load the gene expression data (i.e., the read counts) as an n x p
  # double-precision matrix, where n is the number of samples, and p
  # is the number of genes for which we have expression data (read
  # counts).
  suppressWarnings(
    suppressMessages(
      counts <- read_delim(read.counts.file,delim = "\t",skip = 2,
                           progress = FALSE,
                           col_types = cols(.default    = col_double(),
                                            Name        = col_character(),
                                            Description = col_character()))))
  class(counts)    <- "data.frame"
  genes            <- counts$Name
  rownames(counts) <- genes
  counts           <- counts[-(1:2)]
  counts           <- as.matrix(counts)
  counts           <- t(counts)

  # Remove any genes that do not vary in the sample.
  cols   <- apply(counts,2,sd) > 0
  counts <- counts[,cols]
  
  # Get the subset of sample attributes corresponding the tissue
  # samples with gene expression data.
  rows    <- is.element(rownames(samples),rownames(counts))
  samples <- samples[rows,]
  rows    <- match(rownames(counts),rownames(samples))
  samples <- samples[rows,]

  # Convert the tissue labels to factors.
  samples <- transform(samples,
                       general  = factor(general),
                       specific = factor(specific))

  # Return a data frame containing the sample attributes (samples),
  # and a matrix containing the gene expression data (counts).
  return(list(samples = samples,counts = counts))
}

# TO DO: Explain here what this function does, and how to use it.
plot.gtex.pcs <- function (tissues, pcs, x = "PC1", y = "PC2",
                           guide = "legend") {

  # Specify the colours and shapes used in the scatterplot.
  colors <- c("#E69F00","#56B4E9","#009E73","#0072B2","#D55E00")
  shapes <- c(19,17,8,1,3,2)
  colors <- rep(colors,each = 6)
  shapes <- rep(shapes,times = 5)

  # Collect all the data used for the plot into a single data frame.
  pdat <- cbind(data.frame(tissue = tissues),pcs)

  # Create the scatterplot.
  return(ggplot(pdat,aes_string(x = x,y = y,color = "tissue",
                                shape = "tissue")) +
         geom_point() +
         scale_x_continuous(breaks = NULL) +
         scale_y_continuous(breaks = NULL) +
         scale_color_manual(values = colors,guide = guide) +
         scale_shape_manual(values = shapes,guide = guide) +
         theme_cowplot(font_size = 12) +
         theme(legend.title = element_blank()))
}

