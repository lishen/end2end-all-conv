#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
# IMG_CW_TSV="./metadata/images_crosswalk_prediction.tsv"
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
IMG_FOLDER="/inferenceData"
DL_STATE="/resnet50_candidROI_local_bestAuc_model2.h5"
OUT_PRED="/output/predictions.tsv"

python dm_sc1_candidROI_infer.py \
    --img-height 1024 \
    --img-scale 4095 \
    --roi-per-img 64 \
    --roi-size 256 256 \
    --low-int-threshold 0.05 \
    --blob-min-area 3 \
    --blob-min-int 0.5 \
    --blob-max-int 0.85 \
    --blob-th-step 10 \
    --featurewise-norm \
    --featurewise-mean 915.5 \
    --featurewise-std 735.1 \
    --img-tsv $IMG_CW_TSV \
    --dl-state $DL_STATE \
    --dl-bs 64 \
    --no-validation-mode \
    --out-pred $OUT_PRED \
    $IMG_FOLDER

