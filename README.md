# Lite Audio-Visual Speech Enhancement (Interspeech 2020)

## Introduction

This is the PyTorch implementation of [Lite Audio-Visual Speech Enhancement (LAVSE)](https://arxiv.org/abs/2005.11769).

We have put one preprocessed test sample data (including enhanced results) in this repository.

<!-- The dataset of TMSV (Taiwan Mandarin speech with video) used in LAVSE is released [here](http://xxxxxxxx). -->

Please cite the following paper if you find the codes useful in your research.

```
@inproceedings{chuang2020lite,
  title={Lite Audio-Visual Speech Enhancement},
  author={Chuang, Shang-Yi and Tsao, Yu and Lo, Chen-Chou and Wang, Hsin-Min},
  booktitle={Proc. Interspeech 2020}
}
```

## Prerequisites

* python==3.6
* torch>=1.4.0
* torchaudio>=0.4.0
* torchsummaryX
* tqdm
* visdom
* pystoi
* soundfile

## Usage

You can simply enter the command below and the average PESQ and STOI results will show on your terminal pane.
```
bash run.sh
```
Go check run.sh if you need further information about the command lines.

## Acknowledgment
* Bio-ASP Lab, CITI, Academia Sinica, Taipei, Taiwan
