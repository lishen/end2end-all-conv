#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="trainingData"
IMG_EXT="dcm"
DL_STATE="modelState/2017-03-31_patch_im1024_96/all_cls_best_model3.h5"
OUT="./modelState/2017-03-31_patch_im1024_96/all_cls_best_model3_phm_s6.pkl"

echo -n "Start training: " && date
echo

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 255.0 \
    --no-neg-vs-pos-ratio \
    --net vgg19 \
    --batch-size 200 \
    --patch-size 96 \
    --stride 6 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













