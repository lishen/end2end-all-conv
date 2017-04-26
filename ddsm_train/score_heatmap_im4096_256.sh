#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
PAT_CSV="./Combined_full_ROI/pat_side_cancer.csv"
PAT_LST="./Combined_full_ROI/patient_test.txt"
IMG_FOLDER="./Combined_full_ROI/full_img_png"
# DL_STATE="./Combined_patches_im1024_96_v3/3cls_best_model1.h5"
DL_STATE="./Combined_patches_im4096_256_v3/3cls_best_model4.h5"
# OUT="./Combined_patches_im1024_96_v3/3cls_model1_out/prob_heatmap_s12_test_val.pkl"
OUT="./Combined_patches_im4096_256_v3/3cls_model4_out/prob_heatmap_s64_test.pkl"

echo -n "Start training: " && date
echo

python heatmap_score.py \
    --img-height 4096 \
    --img-scale 255.0 \
    --net resnet50 \
    --batch-size 64 \
    --patch-size 256 \
    --stride 64 \
    --pat-csv $PAT_CSV \
    --pat-list $PAT_LST \
    --out $OUT \
    $IMG_FOLDER $DL_STATE

echo
echo -n "End training: " && date













