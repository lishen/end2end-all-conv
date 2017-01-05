#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
IMG_FOLDER="./preprocessedData/png_288x224"
IMG_EXT="png"

python dm_resnet_train.py \
    --img-extension $IMG_EXT \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --multi-view \
    --val-size 0.2 \
    --featurewise-norm \
    --batch-size 4 \
    --samples-per-epoch 120 \
    --nb-epoch 500 \
    --balance-classes 1.0 \
    --allneg-skip \
    --net resnet18 \
    --nb-init-filter 32 \
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
    --lr-patience 10 \
    --es-patience 100 \
    --best-model ./modelState/dm_resnet18_local_best_model.h5 \
    --final-model ./modelState/dm_resnet18_local_final_model.h5 \
    $IMG_FOLDER

    # --exam-tsv ./metadata/exams_metadata.tsv \
