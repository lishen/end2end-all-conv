#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/trainingData"
# IMG_FOLDER="./preprocessedData/png_288x224"
IMG_EXT="dcm"
# IMG_EXT="png"
BEST_MODEL="/modelState/enet_288_bestAuc_model.pkl"
# FINAL_MODEL="./modelState/enet_288_final_model.pkl"
FINAL_MODEL="NOSAVE"
# SAVED_MODEL="./modelState/2017-01-11_resnet50_288/resnet50_288_bestAuc_model.h5"
DL_STATE="/resnet50_288_bestAuc_model_4.h5"
    
echo -n "Start training: " && date
echo

# --resume-from $SAVED_MODEL \
python dm_enet_train.py \
    --img-extension $IMG_EXT \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --multi-view \
    --val-size 7500 \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --batch-size 32 \
    --samples-per-epoch 64000 \
    --nb-epoch 10 \
    --balance-classes 1.0 \
    --allneg-skip 1.0 \
    --pos-class-weight 1.0 \
    --alpha 0.01 \
    --l1-ratio 0.5 \
    --init-learningrate 0.1 \
    --lr-patience 2 \
    --es-patience 4 \
    --dl-state $DL_STATE \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $IMG_FOLDER

echo
echo -n "End training: " && date
