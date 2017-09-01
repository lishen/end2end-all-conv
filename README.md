# End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design
Li Shen, Ph.D. CS

Icahn School of Medicine at Mount Sinai

New York, New York, USA

![Fig1](https://raw.githubusercontent.com/lishen/end2end-all-conv/master/ddsm_train/Fig%20-%20convert%20patch%20to%20whole%20image%20classifier.png "Whole image end-to-end training")

## Introduction
This is the companion site for our paper - "End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design" at arXiv. Access the paper [here](https://arxiv.org/abs/1708.09427).

For our entry in the DREAM2016 Digital Mammography challenge, see this [write-up](https://www.synapse.org/LiShenDMChallenge). This work is much improved from our method used in the challenge.

## Model downloads
A few best models are available for downloading at this Google Drive [folder](https://drive.google.com/open?id=0B1PVLadG_dCKV2pZem5MTjc1cHc). Here is a table for individual downloads:

| Database  | Patch Classifier  | Top Layers (two blocks)  | Single AUC  | Augmented AUC  | Link  |
|---|---|---|---|---|---|
| DDSM  | Resnet50  | \[512-512-1024\]x2  | 0.86  | 0.88  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKSUJYdzNyZjVsZHc)  |
| DDSM  | VGG16  | 512x1  | 0.83  | 0.86  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKYnREWlJQZ2JaSDQ)  |
| DDSM  | VGG16  | \[512-512-1024\]x2  | 0.85  | 0.88  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKdVQzbDRLNTZ4TXM)  |
| INbreast  | VGG16  | 512x1  | 0.92  | 0.94  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKN0ZxNFdCRWxHRFU)  |
| INbreast  | VGG16  | \[512-512-1024\]x2  | 0.95  | 0.96  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKUnQwYVhOd2NfQlk)  |

- Inference level augmentation is obtained by horizontal and vertical flips to generate 4 predictions.
- The listed scores are single model AUC and prediction averaged AUC.
- 3 Model averaging on DDSM gives AUC of 0.91
- 2 Model averaging on INbreast gives AUC of 0.96.

## Transfer learning is as easy as 1-2-3
In order to transfer a model to your own data, follow these easy steps.
### Determine the rescale factor
The rescale factor is used to rescale the pixel intensities so that the max value is 255. For PNG format, the max value is 65535, so the rescale factor is 255/65535 = 0.003891. If your images are already in the 255 scale, set rescale factor to 1.
### Calculate the pixel-wise mean
This is simply the mean pixel intensity of your train set images.
### Image size
This is currently fixed at 1152x896 for the models in this study. However, you can change the image size when converting from a patch classifier to a whole image classifier.
### Let the Python modules be seen
I simpy modify the `PYTHONPATH` variable like this: `export PYTHONPATH=$PYTHONPATH:your_path_to_repo/end2end-all-conv`
### Finetune
Now you can finetune a model on your own data for cancer predictions! You may check out this shell [script](ddsm_train/train_image_clf_inbreast.sh). Alternatively, copy & paste from here:
```shell
TRAIN_DIR="INbreast/train"
VAL_DIR="INbreast/val"
TEST_DIR="INbreast/test"
RESUME_FROM="ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"
BEST_MODEL="INbreast/transferred_inbreast_best_model.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

python image_clf_train.py \
    --no-patch-model-state \
    --resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
    --featurewise-center \
    --featurewise-mean 44.33 \
    --no-equalize-hist \
    --batch-size 4 \
    --train-bs-multiplier 0.5 \
    --augmentation \
    --class-list neg pos \
    --nb-epoch 0 \
    --all-layer-epochs 50 \
    --load-val-ram \
    --load-train-ram \
    --optimizer adam \
    --weight-decay 0.001 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.01 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
    --es-patience 10 \
    --auto-batch-balance \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR
```
Some explanations of the arguments:
- The batch size for training is the product of `--batch-size` and `--train-bs-multiplier`. Because training uses roughtly twice (both forward and back props) the GPU memory of testing, `--train-bs-multiplier` is set to 0.5 here.
- For model finetuning, only the second stage of the two-stage training is used here. So `--nb-epoch` is set to 0.
- `--load-val-ram` and `--load-train-ram` will load the image data from the validation and train sets into memory. You may want to turn off these options if you don't have sufficient memory. When turned off, out-of-core training will be used.
- `--weight-decay` and `--hidden-dropout` are for stage 1. `--weight-decay2` and `--hidden-dropout2` are for stage 2.
- The learning rate for stage 1 is `--init-learningrate`. The learning rate for stage 2 is the product of `--init-learningrate` and `--all-layer-multiplier`.

## Computational environment
The research in this study is carried out on a Linux workstation with 8 CPU cores and a single NVIDIA Quadro M4000 GPU with 8GB memory. The deep learning framework is Keras 2 with Tensorflow as the backend. 




