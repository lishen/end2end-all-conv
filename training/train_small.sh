#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH

IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
IMG_EXT="dcm"
BEST_MODEL="/modelState/dummy_model.h5"
FINAL_MODEL="NOSAVE"


python dm_resnet_train.py \
    --img-extension $IMG_EXT \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --multi-view \
    --val-size 0.2 \
    --featurewise-norm \
    --batch-size 8 \
    --samples-per-epoch 80 \
    --nb-epoch 1 \
    --lr-patience 3 \
    --es-patience 10 \
    --balance-classes 1.0 \
    --allneg-skip \
    --pos-class-weight 1.0 \
    --net resnet50 \
    --nb-init-filter 16 \
    --init-filter-size 3 \
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
    --final-model $FINAL_MODEL \
    $IMG_FOLDER
