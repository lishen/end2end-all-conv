#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Training inference method

nvcc --version
lspci | grep -i nvidia
nvidia-smi
python -V
pip show tensorflow

#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#export DYLD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:$DYLD_LIBRARY_PATH
#export CUDA_HOME=/usr/local/cuda-7.5
#export CUDA_ROOT=/usr/local/cuda-7.5

#echo $LD_LIBRARY_PATH

# Run training
python DREAM_DM_starter_tf.py --lr 0.0001 --decay 0.985 --bs 10 --epoch 2 --net Le --ms 32 --dropout 0.5
