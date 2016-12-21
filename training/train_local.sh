#!/bin/bash

# export PYTHONPATH=/:$PYTHONPATH
python dm_resnet_train.py --img-extension png \
    --img-size 288 224 \
    --val-size 0.2 \
    --nb-worker 4 \
    --exam-tsv ./metadata/exams_metadata.tsv \
    --img-tsv ./metadata/images_crosswalk.tsv \
    --net resnet18 \
    --batch-size 64 \
    --samples-per-epoch 640 \
    --nb-epoch 100 \
    --balance-classes 4.0 \
    --weight-decay 0.0001 \
    --lr-patience 10 \
    --es-patience 20 \
    --trained-model ./modelState/dm_resnet_model.h5 \
    ./preprocessedData/png_288x224

