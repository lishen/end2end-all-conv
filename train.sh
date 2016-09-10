#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Training inference method

nvcc --version
lspci | grep -i nvidia
nvidia-smi
python -V
pip show tensorflow


# Run training
python DREAM_DM_starter_tf.py --lr 0.0001 --decay 0.985 --bs 10 --epoch 2 --net Le --ms 32 --dropout 0.5
