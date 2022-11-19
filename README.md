# ViPLO: Vision Transformer based Pose-Conditioned Self-Loop Graph for Human-Object Interaction Detection

This anonymous GitHub repository contains the PyTorch implementation for CVPR 2023 paper (ID 10478) submission as a supplementary material.

This repository focuses on testing our trained ViPLO on the HICO-DET benchmark. As mentioned in the paper, this code is implemented based on the [official code of SCG](https://github.com/fredzzhang/spatially-conditioned-graphs) (Frederic Z. Zhang, Dylan Campbell and Stephen Gould. _Spatially Conditioned Graphs for Detecting Human-Object Interactions_. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13319-13327, October 2021).

(This repo is tested on Ubuntu 18.04 and Ubuntu 20.04 environment)

## Description
The two main contributions of our paper, MOA module and pose-conditioned graph, are implemented as follow.
- MOA module is implemented in the `CLIP` package, for example, in `./CLIP/clip/model.py`.
- pose-conditioned graph is implemented in `./models.py`, `./interaction_head.py`, etc.


## Prerequisites

1. Download the repository.
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket), as in the original SCG code.
3. Download HICO-DET dataset from its [homepage](http://www-personal.umich.edu/~ywchao/hico/), or it is also able to follow the instruction of the original SCG code. After the download, place the dataset folder under the 'hicodet' folder. (ex. `./hicodet/hico_20160224_det/images/test2015/...`)
4. Download [HICO-DET instance files](https://drive.google.com/drive/folders/1AdkHC_1HwKmbDY1uKCdFVHga2uPOKtmC?usp=sharing) used in our experiment and place them under the 'hicodet' folder. (ex. `./hicodet/instances_test2015_vitpose.json`)
5. Download [detection files](https://drive.google.com/drive/folders/1Jk3YTFSZou9CX0UB2wSrdms4oMdmEyN2?usp=share_link) used in our experiment and place them under the 'hicodet' folder. (ex. `./hicodet/detections/test2015_upt_vitpose`)
6. Install the CLIP ViT backbone with our proposed MOA module via `pip install ./CLIP`.
7. Download our [trained weight file](https://drive.google.com/drive/folders/1G64HJ8strFSCudZQysUWvAtvifnOkBFP?usp=sharing), and place it under the 'checkpoints' folder. (ex. `./checkpoints/clip_cls_117_final16_gamma03_ckpt_10983_07.pt`)

## Test on the HICO-DET

```bash
python test.py --detection-dir hicodet/detections/test2015_upt_vitpose \
    --model-path checkpoints/clip_cls_117_final16_gamma03/ckpt_10983_07.pt \
    --warp \
    --pose \
    --local_pose \
    --patch-size 16
```
