#!/bin/bash

TRAIN_DIR="Combined_patches_im4096_256_v3/train"
VAL_DIR="Combined_patches_im4096_256_v3/val"
TEST_DIR="Combined_patches_im4096_256_v3/test"
#RESUME_FROM="Combined_patches_im4096_256_v3/3cls_best_model3.h5"
BEST_MODEL="Combined_patches_im4096_256_v3/3cls_best_model5.h5"
#FINAL_MODEL="Combined_patches_im4096_256_v3/3cls_final_model4.h5"
FINAL_MODEL="NOSAVE"

export NUM_CPU_CORES=3

python patch_clf_train.py \
	--img-size 256 256 \
    --img-scale 255.0 \
	--featurewise-center \
    --featurewise-mean 59.6 \
    --equalize-hist \
	--batch-size 64 \
    --train-bs-multiplier 0.5 \
	--augmentation \
	--class-list background malignant benign \
	--nb-epoch 1 \
    --top-layer-epochs 5 \
    --all-layer-epochs 15 \
    --no-load-val-ram \
    --no-load-train-ram \
    --net resnet50 \
    --optimizer nadam \
    --use-pretrained \
    --no-top-layer-nb \
    --nb-init-filter 64 \
    --init-filter-size 7 \
    --init-conv-stride 2 \
    --max-pooling-size 3 \
    --max-pooling-stride 2 \
    --weight-decay 0.01 \
    --weight-decay2 0.0001 \
    --bias-multiplier 0.1 \
    --alpha 0.0001 \
    --l1-ratio 0.0 \
    --inp-dropout 0.0 \
    --hidden-dropout 0.5 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.01 \
    --top-layer-multiplier 0.01 \
    --all-layer-multiplier 0.0001 \
	--lr-patience 2 \
	--es-patience 5 \
	--no-resume-from \
	--auto-batch-balance \
    --pos-cls-weight 1.0 \
	--neg-cls-weight 1.0 \
	--best-model $BEST_MODEL \
	--final-model $FINAL_MODEL \
	$TRAIN_DIR $VAL_DIR $TEST_DIR	
