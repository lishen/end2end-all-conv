#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
IMG_EXT="dcm"
DL_STATE="/3cls_best_model5_finetuned.h5"
OUT="modelState/3cls_best_model5_finetuned_phm_s128_add.pkl"
PREDICTED="/m5_ftu_phm_s128_predicted_subjs.npy"

echo -n "Start training: " && date
echo

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 4096 \
    --img-scale 255.0 \
    --equalize-hist \
    --featurewise-center \
    --featurewise-mean 91.6 \
    --neg-vs-pos-ratio 1.0 \
    --net resnet50 \
    --batch-size 256 \
    --patch-size 256 \
    --stride 128 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    --predicted-subj-file $PREDICTED \
    --add-subjs 800 \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













