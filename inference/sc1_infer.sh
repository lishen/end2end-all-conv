#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
IMG_FOLDER="/inferenceData"
SAVED_STATE="/resnet50_288_bestAuc_model.h5"
OUT_PRED="/output/predictions.tsv"

echo -n "Start inference: " && date
echo

python dm_sc1_infer.py \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --batch-size 32 \
    --saved-state $SAVED_STATE \
    --no-validation-mode \
    --no-use-mean \
    --out-pred $OUT_PRED \
    $IMG_FOLDER

echo
echo -n "End inference: " && date
echo "Print number of lines of the prediction table:"
wc -l $OUT_PRED
echo "Print head of the predictions:"
head $OUT_PRED
echo "==============================="
echo "Print tail of the predictions:"
tail $OUT_PRED
