# End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design
Li Shen, Ph.D.

Icahn School of Medicine at Mount Sinai

New York, New York, USA

## Introduction
This is the companion site for our manuscript - "End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design" at XXX.

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
- The scores are single model AUC and prediction averaged AUC.
- 3 Model averaging on DDSM gives AUC of 0.91
- 2 Model averaging on INbreast gives AUC of 0.96.

## Transfer learning is as easy as 1-2-3



This repository was originally for my entry in the DREAM 2016 Digital Mammography challenge. In the challenge, I have developed a classifier based on residual net + probabilistic heatmap + gradient boosting trees. After the challenge ended, I switched the direction into developing an end-to-end training strategy to directly predict cancer status at whole-image level.

For my entry in the challenge, see: https://www.synapse.org/LiShenDMChallenge. I ended up ranking 12 in sub-challenge 1 and 10 in sub-challenge 2 out of more than 100 teams.

