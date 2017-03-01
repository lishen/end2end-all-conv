#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./trainingData"
IMG_EXT="dcm"
# DL_STATE="modelState/resnet50_candidROI_mulcls_local_bestAuc_model.h5"
DL_STATE="modelState/2017-02-21_candidROI_net50_mc/resnet50_candidROI_mulcls_bestAuc_model.h5"
TRAIN_OUT="./modelState/meta_prob_train.pkl"
TEST_OUT="./modelState/meta_prob_test.pkl"

echo -n "Start training: " && date
echo

python dm_candidROI_score.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --val-size 0.3 \
    --neg-vs-pos-ratio 10.0 \
    --featurewise-norm \
    --featurewise-mean 873.6 \
    --featurewise-std 739.3 \
    --img-per-batch 4 \
    --roi-per-img 8 \
    --roi-size 256 256 \
    --low-int-threshold 0.05 \
    --blob-min-area 3 \
    --blob-min-int 0.5 \
    --blob-max-int 0.85 \
    --blob-th-step 10 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --train-out $TRAIN_OUT \
    --test-out $TEST_OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













