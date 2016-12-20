#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
python dm_resnet_train.py --img-extension png \
    --img-size 288 224 \
    --val-size 0.1 \
    --nb-worker 4 \
    --exam-tsv ./metadata/exams_metadata.tsv \
    --img-tsv ./metadata/images_crosswalk.tsv \
    --net resnet50 \
    --batch-size 32 \
    --samples-per-epoch 64 \
    --nb-epoch 5 \
    --weight-decay 0.0001 \
    --lr-patience 5 \
    --es-patience 10 \
    --trained-model ./modelState/dm_resnet_model.h5 \
    ./preprocessedData/png_288x224

