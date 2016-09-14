#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Testing inference method

# Run testing
python DREAM_DM_starter_tf.py --net GoogLe --ms 224 --test 1 -pf /scoringData
