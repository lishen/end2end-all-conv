#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="./metadata/images_crosswalk.tsv"
EXAM_TSV="./metadata/exams_metadata.tsv"
IMG_FOLDER="trainingData"
IMG_EXT="dcm"
#DL_STATE="modelState/2017-04-19_patch_im4096_256/3cls_best_model5.h5"
#DL_STATE="modelState/2017-04-19_patch_im4096_256/3cls_best_model5_pilot_finetuned.h5"
DL_STATE="modelState/2017-04-19_patch_im4096_256/3cls_best_model5_finetuned.h5"
#DL_STATE="modelState/2017-03-31_patch_im1024_96/all_cls_best_model3.h5"
#DL_STATE="modelState/2017-03-16_candidROI_net50_mc5/resnet50_candidROI_mulcls_bestAuc_model5.h5"
#OUT="./modelState/2017-03-31_patch_im1024_96/all_cls_best_model3_phm_s24.pkl"
#OUT="./modelState/2017-04-19_patch_im4096_256/pilot_model5_DMfinetuned_phm_s128.pkl"
OUT="./scratch/add_subj_phm.pkl"
PREDICTED="./scratch/predicted_subjs.npy"
#OUT="modelState/2017-03-16_candidROI_net50_mc5/mc5_best_phm_s6.h5"

echo -n "Start training: " && date
echo

#export CUDA_VISIBLE_DEVICES=""

python dm_heatmap_score.py \
    --img-extension $IMG_EXT \
    --img-height 4096 \
    --img-scale 255.0 \
    --equalize-hist \
    --featurewise-center \
    --featurewise-mean 91.6 \
    --no-neg-vs-pos-ratio \
    --net resnet50 \
    --batch-size 64 \
    --patch-size 256 \
    --stride 128 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    --predicted-subj-file $PREDICTED \
    --add-subjs 10 \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













