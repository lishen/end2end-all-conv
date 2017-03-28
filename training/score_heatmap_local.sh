#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./trainingData"
IMG_EXT="dcm"
# DL_STATE="modelState/2017-03-16_candidROI_net50_mc5/resnet50_candidROI_mulcls_bestAuc_model5.h5"
DL_STATE="modelState/2017-02-02_roi_256/roi_clf4.h5"
TRAIN_OUT="./modelState/prob_heatmap_roiclf_s16_train.pkl"
TEST_OUT="./modelState/prob_heatmap_roiclf_s16_test.pkl"

echo -n "Start training: " && date
echo

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --test-size 0.3 \
    --neg-vs-pos-ratio 10.0 \
    --featurewise-norm \
    --featurewise-mean 1111.6 \
    --featurewise-std 718.1 \
    --batch-size 32 \
    --patch-size 256 \
    --stride 16 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --train-out $TRAIN_OUT \
    --test-out $TEST_OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













