# End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design
Li Shen, Ph.D.

Icahn School of Medicine at Mount Sinai

New York, New York, USA

## Introduction
This is the companion site for our manuscript - "End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design" at XXX.

For our entry in the DREAM2016 Digital Mammography challenge, see this [write-up](https://www.synapse.org/LiShenDMChallenge).

## Model downloads
A few best models are available for downloading at this Google Drive [folder](https://drive.google.com/open?id=0B1PVLadG_dCKV2pZem5MTjc1cHc). Here is a table for individual downloads:

| Database  | Patch Classifier  | Top Layers  | Single AUC  | Augmented AUC  | Link  |
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






