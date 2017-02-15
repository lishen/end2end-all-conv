#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
IMG_EXT="dcm"
DL_STATE="/resnet50_candidROI_bestAuc_model.h5"
ROI_STATE="/roi_clf4.h5"
KM_STATE="/modelState/dlrepr_km_model.pkl"
BOW_TRAIN_OUT="/modelState/bow_dat_train.pkl"
BOW_TEST_OUT="/modelState/bow_dat_test.pkl"

echo -n "Start training: " && date
echo

python dm_bow_train.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --val-size 80 \
    --featurewise-norm \
    --featurewise-mean 873.6\
    --featurewise-std 739.3 \
    --img-per-batch 2 \
    --roi-per-img 16 \
    --roi-size 256 256 \
    --low-int-threshold 0.05 \
    --blob-min-area 3 \
    --blob-min-int 0.5 \
    --blob-max-int 0.85 \
    --blob-th-step 10 \
    --roi-state $ROI_STATE \
    --roi-clf-bs 64 \
    --nb-pos-samples 384 \
    --nb-neg-samples 1536 \
    --no-aug-for-neg \
    --sample-per-pos 4 \
    --sample-per-neg 2 \
    --dl-clf-bs 64 \
    --nb-words 512 \
    --km-max-iter 100 \
    --km-bs 10 \
    --km-patience 100 \
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













