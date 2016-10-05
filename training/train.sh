#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Training inference method

echo "==== nvidia-smi command output ===="
nvidia-smi
echo "==== python version ===="
python -V
echo "==== tensorflow package info ===="
pip show tensorflow


# Run training
echo "==== Start training ===="
python DREAM_DM_starter_tf.py --lr 0.0001 --reg 0.0001 --decay 0.985 --bs 50 --time 240 --net GoogLe --ms 224 --dropout 0.6
