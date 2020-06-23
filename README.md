# LAVSE
Python codes for [Lite Audio-Visual Speech Enhancement (LAVSE)](https://arxiv.org/abs/2005.11769).

## Introduction

This is the implementation of LAVSE in PyTorch. \
We have put one preprocessed test sample data (including enhanced results) in the result directory in this repository. \
The dataset of TMSV (Taiwan Mandarin speech with video) used in LAVSE is released [here](http://xxxxxxxx). \
Please cite the following paper if you use the codes from this repository.

[S.-Y. Chuang, Y. Tsao, C.-C. Lo, and H.-M. Wang, “Lite audio-visual speech enhancement,” in arXiv preprint arXiv:2005.11769, 2020.](https://arxiv.org/abs/2005.11769)

## Requirement

* python 3.6
* NVIDIA GPU + CUDA 10 + CuDNN

You can simply enter the command below and install the rest of the requirements.
```
bash env_setup.sh
```

## Usage

You can simply enter the command below and the average PESQ and STOI results will show on your terminal pane.
```
bash run.sh
```
Go check run.sh if you need further information about the command lines.

## Acknowledgment
* Bio-ASP Lab, CITI, Academia Sinica, Taipei, Taiwan
