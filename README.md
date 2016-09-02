


# DREAM_DM_starter_code

This is some starter code for the DREAM Digital Mammography challenge.  This directory will give examples of how to Dockerize your code, implement a working data I/O pipeline, and some basic examples of Deep Learning with the challenge data.  The python code utilizes the Tensorflow 0.9.0 framework and provides four popular CNN architectures:

- [Le Net](yann.lecun.com/exdb/lenet)
- [Alex Net](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [VGG16 Net](https://arxiv.org/pdf/1409.1556.pdf)
- [GoogLe Net](www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)


## Dependencies

You must have docker or nvidia-docker running on your system to be able to build the docker image.

## Docker

As shown on the [submission wiki page](https://www.synapse.org/#!Synapse:syn5644795/wiki/392923) for the DREAM challenge, you can build and push the docker image with the following three lines:

```
docker build -t docker.synapse.org/syn123456/dm-trivial-model .
docker login -u <synapse username> -p <synapse password> docker.synapse.org
docker push docker.synapse.org/syn123456/dm-trivial-model
```

where "syn123456" is the ID of the Synapse project where your team collaborates on this challenge.  You can find more information on the wiki page.

## Usage

The python program takes care of training and testing a model (saving a model when necessary).  We have strived to create an extremely modular program, and thus, just from the flags you should be able to capture most of the hyperparameter variance that a vanilla convolutional network can use.  These are the flags:

- `--pf`

To submit to the challenge, you need to submit a training and testing script.  This can be done with a single line with our code.  For example, the training line


```
python DREAM_DM_pilot_tf.py --net Le -- ms 32 --lr 0.0001 --decay 0.985 --bs 10 --epoch 2 --dropout 0.5
```

and the training line

```
python DREAM_DM_pilot_tf.py --net Le --ms 32 --test 1
```

**Note**: As the test script will depend on the model that is saved by the training script, you **MUST** define the same network and matrix size between both the training and testing scripts for this pipeline to work.