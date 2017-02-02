#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH

X_TRAIN="referenceData/MIAS/X_train.npy"
X_TEST="referenceData/MIAS/X_test.npy"
Y_TRAIN="referenceData/MIAS/y_train.npy"
Y_TEST="referenceData/MIAS/y_test.npy"
SAVED_MODEL="modelState/2017-02-02_roi_256/roi_clf.h5"
BEST_MODEL="modelState/2017-02-02_roi_256/roi_clf2.h5"
# BEST_MODEL="NOSAVE"
FINAL_MODEL="NOSAVE"
# SAVED_MODEL="/resnet50_288_bestAuc_model.h5"

echo -n "Start training: " && date
echo

python roi_clf_train.py \
    --img-size 256 256 \
    --featurewise-norm \
    --rotation-range 0 \
    --width-shift-range 0.0 \
    --height-shift-range 0.0 \
    --zoom-range 1.0 1.0 \
    --horizontal-flip \
    --vertical-flip \
    --batch-size 32 \
    --nb-epoch 100 \
    --lr-patience 10 \
    --es-patience 40 \
    --pos-class-weight 5.0 \
    --nb-init-filter 32 \
    --init-filter-size 5 \
    --init-conv-stride 2 \
    --max-pooling-size 2 \
    --max-pooling-stride 2 \
    --weight-decay 0.0001 \
    --alpha 0.0001 \
    --l1-ratio 0.0 \
    --inp-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-learningrate 0.001 \
    --resume-from $SAVED_MODEL \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $X_TRAIN $X_TEST $Y_TRAIN $Y_TEST

echo
echo -n "End training: " && date



