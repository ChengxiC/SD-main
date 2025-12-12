# SD-main
## Introduction
This repository is for Scene-dependent anomaly detection: a benchmark and weakly supervised model

SDNormal is a **synthetic, scene-dependent** video anomaly detection (VAD) benchmark created with **Unity 3D**. It contains **31 scenes** (2D backgrounds + 3D animations) and **7 scene-dependent anomaly types**:
- Dancing
- Fighting
- Knife holding
- Motorcycling
- Running
- Telephoning
- Vehicle movement

**Video specs**
- Resolution: 1280 × 720
- Frame rate: 30 FPS
- Average duration: ~15 seconds per video

The dataset provides **video-level** labels (Normal/Abnormal + context) and **frame-level** labels (0 for normal, 1 for abnormal) with **start/end** positions of abnormal segments. Two annotation formats are available:
- **Single-modal**: standard weakly-supervised VAD binary labels
- **Bimodal**: text labels describing activities (useful for video-text feature extractors, e.g., XCLIP-family)


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
    │   ├── video_001/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── …
    │   ├── video_002/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── …
    │   └── …
    └── abnormal/
        ├── video_101/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── …
        ├── video_102/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── …
        └── …
```
We sincerely thank Lv Hui and the co-authors of UMIL for publicly releasing their code, which has greatly facilitated our implementation and experiments in this work.
