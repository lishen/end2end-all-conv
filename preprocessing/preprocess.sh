#!/bin/bash

TRAIN_DATA_DIR=/trainingData
CONVERTED_PNG_DIR=/preprocessedData/png_org
PROCESSED_PNG_DIR=/preprocessedData/png_prep
IMG_TSV=/metadata/images_crosswalk.tsv

echo ">>> Create folders for converted and processed .png files"
mkdir -p $CONVERTED_PNG_DIR $PROCESSED_PNG_DIR
echo

echo ">>> Convert .dcm files to .png files"
find $TRAIN_DATA_DIR/ -maxdepth 1 -name '*.dcm' | parallel "echo 'convert: {} => $CONVERTED_PNG_DIR/{/.}.png'; convert {} $CONVERTED_PNG_DIR/{/.}.png"
# find $CONVERTED_PNG_DIR -name '*.png' -execdir ls -lh {} \;
echo "Done."
echo

echo ">>> Process .png files using the image preprocessor"
awk 'NR > 1 && $4 == "MLO" {print $6}' $IMG_TSV | parallel "echo 'process MLO: $CONVERTED_PNG_DIR/{/.}.png => $PROCESSED_PNG_DIR/{/.}.png'; python preprocess.py --remove-pectoral $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
awk 'NR > 1 && $4 != "MLO" {print $6}' $IMG_TSV | parallel "echo 'process non-MLO: $CONVERTED_PNG_DIR/{/.}.png => $PROCESSED_PNG_DIR/{/.}.png'; python preprocess.py $CONVERTED_PNG_DIR/{/.}.png $PROCESSED_PNG_DIR/{/.}.png"
# find $PROCESSED_PNG_DIR -name '*.png' -execdir ls -lh {} \;
echo "Done."
echo
