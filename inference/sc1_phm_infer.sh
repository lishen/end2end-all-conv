#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
IMG_FOLDER="/inferenceData"
IMG_EXT="dcm"
DL_STATE="/3cls_best_model4_finetuned.h5"
CLF_INFO_STATE="/model4_ftu_clf_info.pkl"
OUT="/output/predictions.tsv"
PROGRESS="/progress.txt"

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
    --batch-size 200 \
    --patch-size 256 \
    --stride 128 \
    --no-exam-tsv \
    --img-tsv $IMG_CW_TSV \
    --no-validation-mode \
    --no-use-mean \
    --out-pred $OUT \
    --progress $PROGRESS \
    $IMG_FOLDER $DL_STATE $CLF_INFO_STATE

# echo
# echo -n "End training: " && date
