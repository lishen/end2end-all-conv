#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Testing inference method

#export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Run testing
python DREAM_DM_starter_tf.py --net Le --ms 32 --test 1 -pf /scoringData
#echo "0.5 0.5" > /output/out.txt
