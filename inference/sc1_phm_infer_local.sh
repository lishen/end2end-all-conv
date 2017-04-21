#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk_prediction.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
# IMG_FOLDER="./inferenceData"
IMG_FOLDER="preprocessedData/jpg_org"
IMG_EXT="jpg"
DL_STATE="./modelState/3cls_best_model4_finetuned.h5"
CLF_INFO_STATE="./modelState/model4_ftu_clf_info.pkl"
OUT="./output/predictions.tsv"

# echo -n "Start training: " && date
# echo

python dm_sc1_phm_infer.py \
    --img-extension $IMG_EXT \
    --img-height 4096 \
    --img-scale 255.0 \
    --equalize-hist \
    --featurewise-center \
    --featurewise-mean 91.6 \
    --net resnet50 \
    --batch-size 64 \
    --patch-size 256 \
    --stride 256 \
    --no-exam-tsv \
    --img-tsv $IMG_CW_TSV \
    --validation-mode \
    --no-use-mean \
    --out-pred $OUT \
    $IMG_FOLDER $DL_STATE $CLF_INFO_STATE

# echo
# echo -n "End training: " && date
