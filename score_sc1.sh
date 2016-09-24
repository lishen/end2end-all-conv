#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Testing inference method

# Run testing
python DREAM_DM_starter_tf.py --net GoogLe --ms 224 --test 1 --out /output/out_1.txt --pf /scoringData &> /output/out_2.txt
