#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk_prediction.tsv"
IMG_FOLDER="./inferenceData"
SAVED_STATE="./modelState/2017-01-13_resnet50_288/resnet50_288_bestAuc_model.h5"

# echo -n "Start training: " && date
# echo

python dm_sc1_infer.py \
    --img-size 288 224 \
    --img-tsv $IMG_CW_TSV \
    --featurewise-norm \
    --featurewise-mean 485.9 \
    --featurewise-std 765.2 \
    --batch-size 32 \
    --saved-state $SAVED_STATE \
    --validation-mode \
    --use-mean \
    --out-pred ./output/predictions.tsv \
    $IMG_FOLDER

# echo
# echo -n "End training: " && date
