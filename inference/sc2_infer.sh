#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/inferenceData"
# DL_STATE="./modelState/2017-01-13_resnet50_288/resnet50_288_bestAuc_model.h5"
DL_STATE="/resnet50_288_bestAuc_model_4.h5"
ENET_STATE="/enet_288_bestAuc_model.pkl"
XGB_STATE="/bst_288_bestAuc_model.pkl"
OUT_PRED="/output/predictions.tsv"

echo -n "Start inference: " && date
echo

    # --dl-state $DL_STATE \
python dm_sc2_infer.py \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --enet-state $DL_STATE $ENET_STATE \
    --xgb-state $XGB_STATE \
    --no-validation-mode \
    --no-use-mean \
    --out-pred $OUT_PRED \
    $IMG_FOLDER


echo
echo -n "End inference: " && date
echo
echo "Print number of lines of the prediction table:"
wc -l $OUT_PRED
echo
echo "Print head of the predictions:"
head $OUT_PRED
echo "==============================="
echo "Print tail of the predictions:"
tail $OUT_PRED
