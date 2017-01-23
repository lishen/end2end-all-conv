#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
# DL_STATE="./modelState/2017-01-13_resnet50_288/resnet50_288_bestAuc_model.h5"
DL_STATE="/resnet50_288_bestAuc_model_4.h5"
ENET_STATE="/enet_288_bestAuc_model.pkl"
OUT_PRED="/modelState/predictions.tsv"

    # --dl-state $DL_STATE \
python dm_sc1_infer.py \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --batch-size 32 \
    --enet-state $DL_STATE $ENET_STATE \
    --validation-mode \
    --no-use-mean \
    --out-pred $OUT_PRED \
    $IMG_FOLDER

Rscript ./calcAUC.R $OUT_PRED
