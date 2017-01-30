#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./inferenceData"
# DL_STATE="./modelState/2017-01-13_resnet50_288/resnet50_288_bestAuc_model.h5"
DL_STATE="./modelState/2017-01-15_resnet50_288_4/resnet50_288_bestAuc_model_4.h5"
ENET_STATE="./modelState/2017-01-18_enet_288/enet_288_bestAuc_model.pkl"
XGB_STATE="modelState/2017-01-24_xgb_288/xgb_2017-01-25-10am/bst_288_bestAuc_model.pkl"
OUT_PRED="./output/predictions.tsv"

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
    --validation-mode \
    --no-use-mean \
    --out-pred $OUT_PRED \
    $IMG_FOLDER

Rscript ./calcAUC.R $OUT_PRED
