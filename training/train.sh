#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
python dm_resnet_train.py --img-extension png \
    --img-size 576 448 \
    --val-size 0.1 \
    --nb-worker 20 \
    --exam-tsv /metadata/exams_metadata.tsv \
    --img-tsv /metadata/images_crosswalk.tsv \
    --net resnet50 \
    --batch-size 32 \
    --samples-per-epoch 32 \
    --nb-epoch 100 \
    --balance-classes .0 \
    --weight-decay 0.0001 \
    --lr-patience 5 \
    --es-patience 10 \
    --trained-model /modelState/dm_resnet50_lger_model.h5 \
    /preprocessedData/png_576x448

