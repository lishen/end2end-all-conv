#!/bin/bash

TRAIN_DATA_DIR=/trainingData
CONVERTED_PNG_DIR=/preprocessedData/png_org
RESIZED_PNG_DIR1=/preprocessedData/png_1152x896
RESIZED_PNG_DIR2=/preprocessedData/png_576x448
RESIZED_PNG_DIR3=/preprocessedData/png_288x224
# IMG_TSV=./metadata/images_crosswalk.tsv

echo "[$(date)] >>> Create folders for converted and processed .png files"
mkdir -p $CONVERTED_PNG_DIR $RESIZED_PNG_DIR1 $RESIZED_PNG_DIR2 $RESIZED_PNG_DIR3
echo

echo "[$(date)] >>> Convert .dcm files to .png files"
find $TRAIN_DATA_DIR/ -maxdepth 1 -name '*.dcm' | parallel --no-notice "convert {} $CONVERTED_PNG_DIR/{/.}.png"
echo "[$(date)] Done converted $(find $CONVERTED_PNG_DIR/ -name '*.png'|wc -l) dcm files to png files."
echo

echo "[$(date)] >>> Resize .png files to 1152x896 (HxW)"
find $CONVERTED_PNG_DIR/ -maxdepth 1 -name '*.png' | parallel --no-notice "convert {} -resize 896x1152! $RESIZED_PNG_DIR1/{/.}.png"
echo "[$(date)] Done resized $(find $RESIZED_PNG_DIR1/ -name '*.png'|wc -l) png files."
echo

echo "[$(date)] >>> Resize .png files to 576x448 (HxW)"
find $CONVERTED_PNG_DIR/ -maxdepth 1 -name '*.png' | parallel --no-notice "convert {} -resize 448x576! $RESIZED_PNG_DIR2/{/.}.png"
echo "[$(date)] Done resized $(find $RESIZED_PNG_DIR2/ -name '*.png'|wc -l) png files."
echo

echo "[$(date)] >>> Resize .png files to 288x224 (HxW)"
find $CONVERTED_PNG_DIR/ -maxdepth 1 -name '*.png' | parallel --no-notice "convert {} -resize 224x288! $RESIZED_PNG_DIR3/{/.}.png"
echo "[$(date)] Done resized $(find $RESIZED_PNG_DIR3/ -name '*.png'|wc -l) png files."
echo

# find $TRAIN_DATA_DIR/ -maxdepth 1 -name '*.dcm' | parallel "echo 'convert: {} => $CONVERTED_PNG_DIR/{/.}.png'; convert {} $CONVERTED_PNG_DIR/{/.}.png"

# echo "[$(date)] >>> Process .png files using the image preprocessor"
# awk 'NR > 1 && $4 == "MLO" {print $6}' $IMG_TSV | parallel "echo 'process MLO: $CONVERTED_PNG_DIR/{/.}.png => $PROCESSED_PNG_DIR/{/.}.png'; python preprocess.py --remove-pectoral $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
# awk 'NR > 1 && $4 == "MLO" {print $6}' $IMG_TSV | parallel "python preprocess.py --remove-pectoral $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
# awk 'NR > 1 && $4 != "MLO" {print $6}' $IMG_TSV | parallel "echo 'process non-MLO: $CONVERTED_PNG_DIR/{/.}.png => $PROCESSED_PNG_DIR/{/.}.png'; python preprocess.py $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
# awk 'NR > 1 && $4 != "MLO" {print $6}' $IMG_TSV | parallel "python preprocess.py $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
# find $PROCESSED_PNG_DIR -name '*.png' -execdir ls -lh {} \;
# echo "[$(date)] Done processed $(find $PROCESSED_PNG_DIR/ -name '*.png'|wc -l) png files."
# echo
