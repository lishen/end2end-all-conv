#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./trainingData"
IMG_EXT="dcm"
BEST_MODEL="./modelState/resnet50_candidROI_local_bestAuc_model.h5"
RESUME_FROM="./modelState/2017-02-02_roi_256/roi_clf4.h5"
ROI_STATE="./modelState/2017-02-02_roi_256/roi_clf4.h5"
FINAL_MODEL="NOSAVE"

python dm_candidROI_train.py \
    --img-extension $IMG_EXT \
    --img-height 1024 \
    --img-scale 4095 \
    --featurewise-norm \
    --norm-fit-size 10 \
    --net resnet50 \
    --resume-from $RESUME_FROM \
    --val-size 0.3 \
    --loadval-ram \
    --img-per-batch 2 \
    --roi-per-img 16 \
    --roi-size 256 256 \
    --low-int-threshold 0.05 \
    --blob-min-area 3 \
    --blob-min-int 0.5 \
    --blob-max-int 0.85 \
    --blob-th-step 10 \
    --roi-state $ROI_STATE \
    --clf-bs 32 \
    --patches-per-epoch 12800 \
    --nb-epoch 20 \
    --lr-patience 1 \
    --es-patience 10 \
    --allneg-skip 0.8 \
    --pos-class-weight 5.0 \
    --nb-init-filter 32 \
    --init-filter-size 5 \
    --init-conv-stride 2 \
    --max-pooling-size 2 \
    --max-pooling-stride 2 \
    --weight-decay 0.0001 \
    --alpha 0.0001 \
    --l1-ratio 0.0 \
    --inp-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-learningrate 0.01 \
    --best-model $BEST_MODEL \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --final-model $FINAL_MODEL \
    $IMG_FOLDER















