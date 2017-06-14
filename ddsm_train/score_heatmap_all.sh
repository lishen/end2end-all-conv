#!/bin/bash

PAT_CSV="CBIS-DDSM/Combined_full_ROI/pat_side_cancer.csv"
IMG_FOLDER="CBIS-DDSM/Combined_full_ROI/full_img_png"
DL_STATE="CBIS-DDSM/Combined_patches_im4096_256_v3/keras2_3cls_best_model1.h5"

echo -n "Start training: " && date
echo

for stride in 128; do
    echo ">>>> Stride=${stride} <<<<"
    for set in train val test; do
        PAT_LST="CBIS-DDSM/Combined_full_ROI/patient_${set}.txt"
        OUT="CBIS-DDSM/Combined_patches_im4096_256_v3/keras2_3cls_model1_out/phm_s${stride}_${set}_run2.pkl"
        python heatmap_score.py \
            --img-height 4096 \
            --img-scale 255.0 \
            --no-equalize-hist \
            --featurewise-center \
            --featurewise-mean 59.6 \
            --net resnet50 \
            --batch-size 64 \
            --patch-size 256 \
            --stride ${stride} \
            --pat-csv $PAT_CSV \
            --pat-list $PAT_LST \
            --out $OUT \
            $IMG_FOLDER $DL_STATE
    done
    echo ">>>> Done <<<<"
done

echo
echo -n "End training: " && date













