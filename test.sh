#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Testing inference method

export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Run testing
python DREAM_DM_pilot_tf.py --lr 0.0001 --decay 0.985 --bs 10 --epoch 2 --net Le --dropout 0.5 --test 1
