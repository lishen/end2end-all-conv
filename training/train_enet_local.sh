#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./trainingData"
# IMG_FOLDER="./preprocessedData/png_288x224"
IMG_EXT="dcm"
# IMG_EXT="png"
BEST_MODEL="./modelState/enet_288_local_bestAuc_model.h5"
FINAL_MODEL="./modelState/enet_288_local_final_model.h5"
# FINAL_MODEL="NOSAVE"
# SAVED_MODEL="./modelState/2017-01-11_resnet50_288/resnet50_288_bestAuc_model.h5"
DL_STATE="./modelState/2017-01-13_resnet50_288/resnet50_288_bestAuc_model.h5"

    # --resume-from $SAVED_MODEL \
echo -n "Start training: " && date
echo

python dm_enet_train.py \
    --img-extension $IMG_EXT \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --multi-view \
    --val-size 0.3 \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --batch-size 8 \
    --samples-per-epoch 160 \
    --nb-epoch 50 \
    --balance-classes 1.0 \
    --allneg-skip 1.0 \
    --pos-class-weight 1.0 \
    --alpha 0.01 \
    --l1-ratio 0.5 \
    --power-t 0.75 \
    --init-learningrate 0.1 \
    --dl-state $DL_STATE \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $IMG_FOLDER

echo
echo -n "End training: " && date
