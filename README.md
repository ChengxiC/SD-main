# SD-main
## Introduction
This repository is for Scene-dependent anomaly detection: a benchmark and weakly supervised model


## Requirements
- Python 3.9
- CUDA 121
- numpy
- tqdm
- PyTorch (2.1.0)
- torchvision (0.16.0)
- mmcv
- einops 
- shutil
- ftfy

## Video Feature Extraction
if you want to use two-stage method, you can download the features from here (https://drive.google.com/file/d/1DzA4ec_y1VNeNZljziB3rPSkEpAoISIr/view?usp=sharing)
After downloading, organize the features directory as follows:
```features/
├── SHT/         # Features for ShanghaiTech dataset
└── TAD/         # Features for TAD dataset

## Our Dataset
the SDnormal dataset is available at (https://drive.google.com/file/d/1nmtenSv_4r8_DJQWe3IMEJ9jOHTc9jaN/view?usp=sharing) in the form of extracted frames. Alternatively, you can download the dataset in video format, available at ().

