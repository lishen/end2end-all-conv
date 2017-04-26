#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
# IMG_CW_TSV="./metadata/images_crosswalk_prediction.tsv"
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="./inferenceData"
# IMG_FOLDER="preprocessedData/jpg_org"
IMG_EXT="dcm"
DL_STATE="modelState/2017-04-19_patch_im4096_256/3cls_best_model5_finetuned.h5"
#CLF_INFO_STATE="./modelState/2017-04-10_patch_im4096_256/model4_ftu_clf_info.pkl"
CLF_INFO_STATE="modelState/2017-04-19_patch_im4096_256/model5_ftu_clf_info.pkl"
META_CLF_STATE="modelState/2017-04-19_patch_im4096_256/model5_ftu_based_meta_clf.pkl"
OUT="./output/model5_ftu_predictions_sc2_scratch.tsv"
PROGRESS="./output/progress.txt"

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
    --batch-size 64 \
    --patch-size 256 \
    --stride 128 \
    --exam-tsv $EXAM_TSV \
    --img-tsv $IMG_CW_TSV \
    --validation-mode \
    --use-mean \
    --out-pred $OUT \
    --progress $PROGRESS \
    $IMG_FOLDER $DL_STATE $CLF_INFO_STATE $META_CLF_STATE


# echo
# echo -n "End training: " && date
