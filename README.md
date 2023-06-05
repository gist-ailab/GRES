# GRES: Generalized Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gres-generalized-referring-expression-1/generalized-referring-expression-segmentation)](https://paperswithcode.com/sota/generalized-referring-expression-segmentation?p=gres-generalized-referring-expression-1)

**[🏠[Project page]](https://henghuiding.github.io/GRES/)** &emsp; **[📄[Arxiv]](https://arxiv.org/abs/2306.00968)**  &emsp; **[🔥[New Dataset]](https://github.com/henghuiding/gRefCOCO)**

This repository contains code for paper [GRES: Generalized Referring Expression Segmentation](https://arxiv.org/abs/2306.00968).

<div align="center">
  <img src="https://github.com/henghuiding/ReLA/blob/main/imgs/fig1.png?raw=true" width="100%" height="100%"/>
</div><br/>

## Installation:

The code is tested under CUDA 11.8, Pytorch 1.11.0 and Detectron2 0.6.

1. Install [detectron2](https://github.com/facebookresearch/detectron2) following the [manual](https://detectron2.readthedocs.io/en/latest/)
2. Run `sh make.sh` under `gres_model/modeling/pixel_decoder/ops`
3. Install other required packages: `pip -r requirements.txt`
4. Prepare the dataset following `datasets/DATASET.md`

## Inference

```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```

## Training

Firstly, download the backbone weights (`swin_base_patch4_window12_384_22k.pkl`) and convert it into detectron2 format using the script:

```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl
```

Then start training:
```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_weights]
```

Add your configs subsquently to customize options. For example: 
```
SOLVER.IMS_PER_BATCH 48 
SOLVER.BASE_LR 0.00001 
```
For the full list of base configs, see `configs/referring_R50.yaml` and `configs/Base-COCO-InstanceSegmentation.yaml`


## Models

[Onedrive](https://entuedu-my.sharepoint.com/:u:/g/personal/liuc0058_e_ntu_edu_sg/Ed7MVRIoYjpFlYebJimfQUMBED9YTMhIe62VySCuyDQlJQ?e=KaX9Qd)
[Google](https://drive.google.com/file/d/1-LZdt1Dug9eEZKLCn9Wp5nlYy0v5D8Qz/view?usp=drive_link)

## Acknowledgement

This project is based on [refer](https://github.com/lichengunc/refer), [maskformer](https://github.com/facebookresearch/Mask2Former), [detectron2](https://github.com/facebookresearch/detectron2). Many thanks to the authors for their great works!

## BibTeX
Please consider to cite GRES if it helps your research.

```latex
@inproceedings{GRES,
  title={{GRES}: Generalized Referring Expression Segmentation},
  author={Liu, Chang and Ding, Henghui and Jiang, Xudong},
  booktitle={CVPR},
  year={2023}
}
```
