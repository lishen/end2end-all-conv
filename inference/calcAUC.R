#!/usr/bin/env Rscript

# if (!require(pROC)) {
#   install.packages("pROC")
# }

suppressMessages(library(pROC))
## These functions assume that the gold standard data and the predictions
## have already been matched 

## computes AUC and partial AUC focusing on sensitivity
##
#Assume label and prediction are matched
GetScores <- function(label, prediction, sensitivityRange = c(0.8, 1)) {
  roc1 <- roc(label, prediction, direction = "<")
  AUC <- auc(roc1)[1]
  pAUCse <- auc(roc1, partial.auc = sensitivityRange, 
                partial.auc.focus = "sensitivity", 
                partial.auc.correct = FALSE)[1]
  SpecAtSens <- coords(roc1, sensitivityRange[1], input = "sensitivity", 
                       ret = "specificity")
  list(AUC = AUC, pAUCse = pAUCse, SpecAtSens = SpecAtSens)
}

##
args <- commandArgs(T)
pred.tbl <- read.table(args[1], sep="\t", header=T)
scores = GetScores(pred.tbl$target, pred.tbl$confidence)
cat("==========================================\n")
cat(sprintf("AUC=%.4f, pAUC=%.4f, SpecAtSens=%.4f\n", 
            scores$AUC, scores$pAUCse, scores$SpecAtSens))
cat("==========================================\n")

