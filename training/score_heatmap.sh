#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
IMG_EXT="dcm"
DL_STATE="/3cls_best_model2.h5"
OUT="modelState/3cls_best_model2_phm_s64.pkl"

echo -n "Start training: " && date
echo

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 4096 \
    --img-scale 255.0 \
    --neg-vs-pos-ratio 1.0 \
    --net resnet50 \
    --batch-size 128 \
    --patch-size 256 \
    --stride 64 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













