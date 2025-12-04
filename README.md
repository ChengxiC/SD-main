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

## Extract Frames
use the extract_frames.py to extract frames firstly.

## Our Dataset
The SDnormal dataset is available on Google Drive:  
[Download SDnormal.zip] (https://drive.google.com/file/d/12TKdFuaUN-rybZ2rrZz-pQKdZ4-tVtsx/view?usp=drive_link)

We also provide pre-extracted I3D features of SDnormal, which can be downloaded from  
[here] (https://drive.google.com/file/d/1GUCn6wCpyZzpINGyxHJzxWfvaj2eceEg/view?usp=sharing).

The frame directory is organized as follows:
```
SDnormal/
└── frames/
├── normal/
│ ├── 0.jpg
│ ├── 1.jpg
│ └── …
└── abnormal/
├── 0.jpg
├── 1.jpg
└── …
```
We sincerely thank Lv Hui and the co-authors of UMIL for publicly releasing their code, which has greatly facilitated our implementation and experiments in this work.
