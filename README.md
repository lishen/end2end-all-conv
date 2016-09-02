


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

- `--pf`: The path to the data that must be read in.  The default is set to `"/trainingData"`, but this must be changed to `"/scoringData"` for when you run the test script.
- `--csv1`: The first metadata file for training.  The default is correctly set to `"/metadata/images_crosswalk_SubChallenge1.tsv"`.
- `--csv2`: The second metadata file for training.  The default is correctly set to `"/metadata/exams_metadata_SubChallenge1.tsv"`.
- `--csv3`: The first and only metadata file for testing.  The default is correctly set to `"/scoringData/image_metdata.tsv"`.
- `--net`: The network architecture to use.  There are primarily four archtectures as listed above with the names: `Le`, `Alex`, `VGG16`, and `GoogLe`.  The default is set to GoogLe Net.
- `--lr`: The learning rate.  The default is set to `0.001`.
- `--reg`: The L2 regularization parameter.  The default is set to `0.00001`.
- `--out`: An output textfile.  This is depreciated and should be taken out.
- `--outtxt`: The path of your output textfile that will be scored by the pipeline.  The default is correctly set to `"/output/out.txt"`.
- `--saver`: The path of the model to save.  The default is set to a correct path `"/modelState/model.ckpt"`, but you can change the name of the checkpoint file if you please.
- `--decay`: The learning rate decay.  Default is set to `1.0`, ergo, no decay.
- `--dropout`: The chance of keeping a neuron during the layers of dropout.  Default is set to `0.5`.
- `--bs`: The batchsize.  On a single 12GB GPU, Alex Net can handle more than 200, VGG16 can handle about 15, and GoogLe net can handle about 80.  Le Net is small, so it can handle more than 1000.  Default is set to `10`.
- `--epoch`: This will set the maximum number of epochs (when each image has statistically been seen once.  The default is set to `10`.
- `--test`: If set to anything greater than 0, the program will run a test time loop, and do no training while producing the output file.  Default is set to `0`.
- `--ms`: The matrix size the input should be resized to.  The default is set to `224`, but for Le Net, you should use a size of `32`.
- `--time`: The maximum amount of time (in minutes) that you want the training to run for.  The default is set to `1000000`, forcing the ending criteria to be decided by the epoch count.

**Note:** The program will take into account both the time and epoch parameters and stop the program at the more conservative upper bound.

To submit to the challenge, you need to submit a training and testing script.  This can be done with a single line with our code.  For example, the training line

```
python DREAM_DM_pilot_tf.py --net Le -- ms 32 --lr 0.0001 --decay 0.985 --bs 10 --epoch 2 --dropout 0.5
```

and the training line

```
python DREAM_DM_pilot_tf.py --net Le --ms 32 --test 1
```

**Note**: As the test script will depend on the model that is saved by the training script, you **MUST** define the same network and matrix size between both the training and testing scripts for this pipeline to work.

## Advanced Usage

The module `general_conv` lets you define your own convolution layer.  Basically, since every convolutional neural network can be defined by an extremely simple set of numbers, this program will take in those numbers and give you the corresponding network.  You can feed in your customized CNN architecture to `general_conv` through the list of list variable `architecture_conv`.  This will either be a list of 3 numbers (filter size, filter count, and stride) for a convolution layer or a list of two numbers (0 followed by the pooling size) for a max pool layer.  For example, below, I can use these modules to quickly define and call Alex Net or VGG16 Net.

```
# AlexNet with an input tensor X and 2 outputs.
architecture_Alex = [[11, 96, 4], [0, 2],
                     [11, 256, 1], [0, 2],
                     [3, 384, 1], [3, 384, 1], [3, 256, 1], [0, 2]]
layer = general_conv(X, architecture_Alex)
layer = dense(layer, 4096)
layer = dense(layer, 4096)
pred_Alex = output(layer, 2)

# VGG16 Net with an input tensor X and 2 outputs.
architecture_VGG16 = [[3, 64, 1], [3, 64, 1], [0, 2],
                      [3, 128, 1], [3, 128, 1], [0, 2],
                      [3, 256, 1], [3, 256, 1], [3, 256, 1], [0, 2],
                      [3, 512, 1], [3, 512, 1], [3, 512, 1], [0, 2],
                      [3, 512, 1], [3, 512, 1], [3, 512, 1], [0, 2]]
layer = general_conv(X, architecture_VGG16)
layer = dense(layer, 4096)
layer = dense(layer, 4096)
pred_VGG16 = output(layer, 2)
```