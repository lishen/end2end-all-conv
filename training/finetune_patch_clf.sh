#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
IMG_CW_TSV="/metadata/images_crosswalk.tsv"
EXAM_TSV="/metadata/exams_metadata.tsv"
IMG_FOLDER="trainingData"
IMG_EXT="dcm"
DL_STATE="/3cls_best_model6.h5"
BEST_MODEL="/modelState/3cls_best_model6_finetuned.h5"
TRAIN_OUT="/scratch/train"
VAL_OUT="/scratch/val"
TEST_OUT="/scratch/test"
OUT_EXT="png"
NEG_NAME="benign"
POS_NAME="malignant"
BKG_NAME="background"
OUT="/modelState/3cls_best_model6_subjs.pkl"

echo -n "Start training: " && date
echo

mkdir -p $TRAIN_OUT/$NEG_NAME $TRAIN_OUT/$POS_NAME $TRAIN_OUT/$BKG_NAME
mkdir -p $VAL_OUT/$NEG_NAME $VAL_OUT/$POS_NAME $VAL_OUT/$BKG_NAME
mkdir -p $TEST_OUT/$NEG_NAME $TEST_OUT/$POS_NAME $TEST_OUT/$BKG_NAME

python dm_patchClf_finetune.py \
    --img-extension $IMG_EXT \
    --img-height 4096 \
    --img-scale 255.0 \
    --equalize-hist \
    --featurewise-center \
    --featurewise-mean 91.6 \
    --neg-vs-pos-ratio 1.0 \
    --test-size 0.15 \
    --val-size 0.1 \
    --net resnet50 \
    --batch-size 200 \
    --train-bs-multiplier 0.5 \
    --patch-size 256 \
    --stride 256 \
    --roi-cutoff 0.9 \
    --bkg-cutoff 0.5 0.9 \
    --sample-bkg \
    --train-out $TRAIN_OUT \
    --test-out $TEST_OUT \
    --val-out $VAL_OUT \
    --out-img-ext $OUT_EXT \
    --neg-name $NEG_NAME \
    --pos-name $POS_NAME \
    --bkg-name $BKG_NAME \
    --augmentation \
    --load-train-ram \
    --load-val-ram \
    --no-top-layer-nb \
    --nb-epoch 0 \
    --top-layer-epochs 3 \
    --all-layer-epochs 10 \
    --optim nadam \
    --init-lr 0.01 \
    --top-layer-multiplier 0.01 \
    --all-layer-multiplier 0.0001 \
    --es-patience 5 \
    --lr-patience 2 \
    --weight-decay2 0.0001 \
    --bias-multiplier 0.1 \
    --hidden-dropout2 0.0 \
    --img-tsv $IMG_CW_TSV \
    --exam-tsv $EXAM_TSV \
    --out $OUT \
    $IMG_FOLDER $DL_STATE $BEST_MODEL

echo
echo -n "End training: " && date













