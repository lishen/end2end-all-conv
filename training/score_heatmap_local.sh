#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="preprocessedData/jpg_org"
IMG_EXT="jpg"
DL_STATE="modelState/ma_vs_ba_best_model2.009-0.93.h5"
OUT="./modelState/prob_heatmap.pkl"

echo -n "Start training: " && date
echo

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 255.0 \
    --neg-vs-pos-ratio 1.0 \
    --net vgg16 \
    --batch-size 32 \
    --patch-size 96 \
    --stride 96 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













