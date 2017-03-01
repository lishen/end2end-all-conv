#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
IMG_EXT="dcm"
# resnet50_candidROI_mulcls_bestAuc_model2.h5  resnet50_candidROI_mulcls_final_model2.h5
DL_STATE="/resnet50_candidROI_mulcls_final_model2.h5"
TRAIN_OUT="/modelState/meta_prob_train_mc2_final.pkl"
TEST_OUT="/modelState/meta_prob_test_mc2_final.pkl"

echo -n "Start training: " && date
echo

python dm_candidROI_score.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --val-size 8000 \
    --neg-vs-pos-ratio 10.0 \
    --featurewise-norm \
    --featurewise-mean 873.6 \
    --featurewise-std 739.3 \
    --img-per-batch 4 \
    --roi-per-img 32 \
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













