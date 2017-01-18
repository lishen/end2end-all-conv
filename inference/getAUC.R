# This is the original script from DM organizers.

if (!require(pROC)) {
  install.packages("pROC")
}

library(pROC)
## These functions assume that the gold standard data and the predictions
## have already been matched 

## computes AUC and partial AUC focusing on sensitivity
##
#Assume label and prediction are matched
GetScores <- function(label, prediction, sensitivityRange = c(0.8, 1)) {
  roc1 <- roc(label, prediction, direction = "<")
  AUC <- auc(roc1)[1]
  pAUCse <- auc(roc1, partial.auc = sensitivityRange, partial.auc.focus = "sensitivity", partial.auc.correct = FALSE)[1]
  SpecAtSens <- coords(roc1, sensitivityRange[1], input = "sensitivity", ret = "specificity")
  list(AUC = AUC, pAUCse = pAUCse, SpecAtSens = SpecAtSens)
}
