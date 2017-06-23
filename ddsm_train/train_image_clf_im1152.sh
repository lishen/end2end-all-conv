#!/bin/bash

TRAIN_DIR="Combined_full_images/full_train_1152x896"
VAL_DIR="Combined_full_images/full_val_1152x896"
TEST_DIR="Combined_full_images/full_test_1152x896"
PATCH_STATE="Combined_patches_im1152_224_s1/resnet_prt_best1.h5"
BEST_MODEL="Combined_full_images/resnet_1152x896_prt_addtop1.h5"
FINAL_MODEL="NOSAVE"

export NUM_CPU_CORES=4

# 255/65535 = 0.003891.
python image_clf_train.py \
	--patch-model-state $PATCH_STATE \
	--no-resume-from \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
	--featurewise-center \
    --featurewise-mean 52.18 \
    --no-equalize-hist \
    --top-depths 512 512 \
    --top-repetitions 3 3 \
	--kept-layer-idx -5 \
    --batch-size 20 \
    --train-bs-multiplier 0.5 \
	--augmentation \
	--class-list neg pos \
	--nb-epoch 1 \
    --all-layer-epochs 2 \
    --no-load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.0001 \
    --weight-decay2 0.0001 \
    --bias-multiplier 0.1 \
    --hidden-dropout 0.0 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.001 \
    --all-layer-multiplier 0.1 \
	--lr-patience 2 \
	--es-patience 10 \
	--auto-batch-balance \
    --pos-cls-weight 1.0 \
	--neg-cls-weight 1.0 \
	--best-model $BEST_MODEL \
	--final-model $FINAL_MODEL \
	$TRAIN_DIR $VAL_DIR $TEST_DIR	
