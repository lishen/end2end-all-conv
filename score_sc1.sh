#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Testing inference method

# Run testing
python DREAM_DM_starter_tf.py --net Le --ms 32 --test 1 --out /output/out_1.txt --pf /scoringData &> /output/out_2.txt
