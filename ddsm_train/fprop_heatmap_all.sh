#!/bin/bash

PAT_CSV="CBIS-DDSM/Combined_full_ROI/pat_side_cancer.csv"
IMG_FOLDER="CBIS-DDSM/Combined_full_ROI/full_img_png"
DL_STATE="CBIS-DDSM/Combined_patches_im1152_224_s10/resnet_prt_best1.h5"

echo -n "Start training: " && date
echo

for set in train val test; do
    PAT_LST="CBIS-DDSM/Combined_full_ROI/patient_${set}.txt"
    OUT_DIR="CBIS-DDSM/Combined_patches_im1152_224_s10/resnet_prt_best1_out"
    mkdir -p $OUT_DIR
    OUT="$OUT_DIR/phm_s1x1_${set}.pkl"
    python heatmap_score.py \
        --fprop-mode \
        --img-size 1152 896 \
        --no-img-height \
        --no-img-scale \
        --rescale-factor 0.003891 \
        --no-equalize-hist \
        --featurewise-center \
        --featurewise-mean 52.18 \
        --avg-pool-size 7 7 \
        --hm-strides 1 1 \
        --net resnet50 \
        --pat-csv $PAT_CSV \
        --pat-list $PAT_LST \
        --out $OUT \
        $IMG_FOLDER $DL_STATE
done

echo
echo -n "End training: " && date













