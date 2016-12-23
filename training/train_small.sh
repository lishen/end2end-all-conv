#!/bin/bash

export PYTHONPATH=/:$PYTHONPATH
python dm_resnet_train.py --img-extension png \
    --img-size 288 224 \
    --val-size 0.2 \
    --nb-worker 4 \
    --exam-tsv /metadata/exams_metadata_pilot_20160906.tsv \
    --img-tsv /metadata/images_crosswalk_pilot_20160906.tsv \
    --net resnet50 \
    --batch-size 16 \
    --samples-per-epoch 160 \
    --nb-epoch 30 \
    --balance-classes 4.0 \
    --weight-decay 0.0001 \
    --inp-dropout 0.2 \
    --hidden-dropout 0.5 \
    --lr-patience 5 \
    --es-patience 10 \
    --best-model /modelState/dm_resnet50_best_model.h5 \
    --final-model /modelState/dm_resnet50_final_model.h5 \
    /preprocessedData/png_288x224

