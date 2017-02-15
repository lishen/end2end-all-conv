#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./trainingData"
IMG_EXT="dcm"
DL_STATE="modelState/resnet50_candidROI_local_bestAuc_model8.h5"
ROI_STATE="./modelState/2017-02-02_roi_256/roi_clf4.h5"
KM_STATE="./modelState/dlrepr_km_model.pkl"
BOW_TRAIN_OUT="./modelState/bow_dat_train.pkl"
BOW_TEST_OUT="./modelState/bow_dat_test.pkl"

echo -n "Start training: " && date
echo

python dm_bow_train.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --val-size 0.3 \
    --featurewise-norm \
    --featurewise-mean 918.6 \
    --featurewise-std 735.2 \
    --img-per-batch 2 \
    --roi-per-img 16 \
    --roi-size 256 256 \
    --low-int-threshold 0.05 \
    --blob-min-area 3 \
    --blob-min-int 0.5 \
    --blob-max-int 0.85 \
    --blob-th-step 10 \
    --roi-state $ROI_STATE \
    --roi-clf-bs 32 \
    --nb-pos-samples 1040 \
    --nb-neg-samples 4160 \
    --aug-for-neg \
    --sample-per-pos 8 \
    --sample-per-neg 4 \
    --dl-clf-bs 32 \
    --nb-words 4 \
    --km-max-iter 100 \
    --km-bs 200 \
    --km-patience 30 \
    --km-init 30 \
    --exam-neg-vs-pos-ratio 4.0 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --km-state $KM_STATE \
    --bow-train-out $BOW_TRAIN_OUT \
    --bow-test-out $BOW_TEST_OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













