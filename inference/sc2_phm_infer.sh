#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="/inferenceData"
IMG_EXT="dcm"
DL_STATE="/3cls_best_model5_finetuned.h5"
CLF_INFO_STATE="/model5_ftu_clf_info.pkl"
META_CLF_STATE="/model5_ftu_based_meta_clf.pkl"
OUT="/output/predictions.tsv"
PROGRESS="/output/progress.txt"

echo -n "Start training: " && date
echo

python dm_sc2_phm_infer.py \
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
    --exam-tsv $EXAM_TSV \
    --img-tsv $IMG_CW_TSV \
    --no-validation-mode \
    --use-mean \
    --out-pred $OUT \
    --progress $PROGRESS \
    $IMG_FOLDER $DL_STATE $CLF_INFO_STATE $META_CLF_STATE


# echo
# echo -n "End training: " && date
