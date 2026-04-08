# [CVPR22] Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch

> Official PyTorch implementation of the CVPR 2022 paper "Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation".

  <p>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-3.6-blue.svg" alt="Python">
  </p>

## Authors

**Xingning Dong**<sup>1</sup>, **Tian Gan**<sup>1</sup>\*, **Xuemeng Song**<sup>1</sup>, **Jianlong Wu**<sup>1</sup>, **Yuan Cheng**<sup>2</sup>, **Liqiang Nie**<sup>1</sup>

<sup>1</sup> `Shandong University`  
<sup>2</sup> `Ant Group`  
\* Corresponding author / Contact: dongxingning1998@gmail.com

## Links

- **Paper**: [arXiv Link](http://arxiv.org/abs/2203.09811)
- **Code Repository**: [GitHub](https://github.com/dongxingning/SHA_GCL_for_SGG)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [03/2022] Initial release and paper accepted by CVPR 2022.

---

## Introduction

This repository is the official implementation of the paper **Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation**.

Our method addresses the unbiased Scene Graph Generation (SGG) problem by introducing Stacked Hybrid-Attention and a Group Collaborative Learning (GCL) strategy. This codebase provides the official implementation, pretrained object detector checkpoints, and evaluation scripts for the Visual Genome (VG) and GQA datasets.

---

## Highlights

- **Multi-Dataset Support**: Full compatibility with Visual Genome (VG) and GQA.
- **Task Coverage**: Supports PredCls (Predicate Classification), SGCls (Scene Graph Classification), and SGDet (Scene Graph Detection).
- **Flexible Predictors**: Includes various models such as TransLike, MotifsLike, and VCTree, alongside our GCL decoder versions.
- **Modular Encoders**: Easily switch between Self-Attention, Cross-Attention, Hybrid-Attention, Motifs, and VTransE.

---

## Method / Framework

![Framework](/assets/framework.png)
**Figure 1.** The framework of the common pipeline in SGG, which includes five key components. Notably, we improve three key components marked in red in the figure. Specifically, we propose the Stacked Hybrid-Attention network to enhance the object encoder and the relation encoder, and we also devise the Group Collaborative Learning strategy to guide the training of the relation decoder.

![Framework](/assets/gcl.png)
**Figure 2.** Illustration of the proposed Group Collaborative Learning (GCL) strategy, which includes five key steps. It is worth noting that we design two optimization mechanisms, namely Parallel Classifier Optimization (PCO) and Collaborative Knowledge Distillation (CKD), to jointly guide the training of the relation decoder.


---

## Project Structure

```text
.
├── configs/               # Configuration files (e.g., SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml)
├── datasets/              # Dataset directories (VG, GQA)
├── SHA_GCL_extra/         # Helper scripts for dataset paths and Group splits
├── tools/                 # Training and evaluation entry points
├── maskrcnn_benchmark/    # Core source code and configuration defaults
├── README.md
├── INSTALL.md             # Detailed installation guide
├── DATASET.md             # Detailed dataset preprocessing guide
└── LICENSE
```

---


## Installation

We recommend configuring the environment with **CUDA 10.1** & **PyTorch 1.6.0**.

### 1\. Clone the repository

```bash
git clone [https://github.com/dongxingning/SHA_GCL_for_SGG.git](https://github.com/dongxingning/SHA_GCL_for_SGG.git)
cd SHA_GCL_for_SGG
```

### 2\. Detailed Installation

Please check [INSTALL.md](https://www.google.com/search?q=INSTALL.md) for step-by-step installation instructions.

-----

## Checkpoints / Models

We provide pretrained object detectors and trained SGG models for quick reproduction:

### Object Detectors

  - **VG Dataset**: Pretrained object detector provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). [Download Link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR)
  - **GQA Dataset**: Pretrained object detector for GQA. [Download Link](https://1drv.ms/u/s!AjK8-t5JiDT1kxBfihou2smfXFV9?e=VtyoR7) *(Note: We recommend pretraining a new one on GQA for optimal region-level feature extraction).*

### Trained SGG Models

  - **SHA\_GCL\_VG\_PredCls**: [Download Link](https://1drv.ms/u/s!AjK8-t5JiDT1kxI8NkjiMUWBRnWd?e=w5zuBh)

If you want to get more trained models mentioned in our paper, please email `dongxingning1998@gmail.com`.

-----

## Dataset / Benchmark

Please check [DATASET.md](https://www.google.com/search?q=DATASET.md) for instructions on dataset preprocessing (VG & GQA).

First, please refer to `SHA_GCL_extra/dataset_path.py` and set the `datasets_path` to your dataset path. Organize all the files like this:

```text
datasets
  |-- vg
    |--detector_model
      |--pretrained_faster_rcnn
        |--model_final.pth
      |--GQA
        |--model_final_from_vg.pth       
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
  |--gqa
    |--images
      |--.... (images)
    |--GQA_200_ID_Info.json
    |--GQA_200_Train.json
    |--GQA_200_Test.json
```

-----

## Usage

### Configuration Setup

You can configure the training/testing behavior via command line parameters or in `configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml` (and `maskrcnn_benchmark/config/defaults.py`). The priority is `command > yaml > defaults.py`.

  - **Dataset**: `GLOBAL_SETTING.DATASET_CHOICE 'VG'` or `'GQA'`
  - **Task**:
      - **PredCls**: `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
      - **SGCls**: `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`
      - **SGDet**: `MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`
  - **Model / Predictor**: `GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL'`
      - Options: `"MotifsLikePredictor", "VCTreePredictor", "TransLikePredictor", "MotifsLike_GCL", "VCTree_GCL", "TransLike_GCL"`
  - **Encoder**: `GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention'`
      - Options for TransLike: `'Self-Attention', 'Cross-Attention', 'Hybrid-Attention'`
      - Options for MotifsLike: `'Motifs', 'VTransE'`
  - **Group Split (GCL only)**: `GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4'`
      - Options: `['divide4', 'divide3', 'divide5', 'average']`
  - **Knowledge Transfer (GCL only)**: `GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown'`
      - Options: `['None', 'KL_logit_Neighbor', 'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']`

### Training

**Example 1: (VG, TransLike, Hybrid-Attention, divide4, Topdown, PredCls)**

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/SHA_GCL_VG_PredCls_test
```

**Example 2: (GQA\_200, MotifsLike, Motifs, divide4, Topdown, SGCls)**

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'GQA_200' GLOBAL_SETTING.RELATION_PREDICTOR 'MotifsLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Motifs' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/Motifs_GCL_GQA_SGCls_test
```

### Evaluation

You can evaluate the trained model by running the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True TEST.IMS_PER_BATCH 8 DTYPE "float16" GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/SHA_GCL_VG_PredCls_test
```

-----


## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@inproceedings{dong2022stacked,
  title={Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation},
  author={Dong, Xingning and Gan, Tian and Song, Xuemeng and Wu, Jianlong and Cheng, Yuan and Nie, Liqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19427--19436},
  year={2022}
}
```

---

## Acknowledgement

  - Our codebase is built on top of [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). We sincerely thank them for their well-designed codebase.
  - We welcome you to submit an issue or contact us if you have any problems when reading the paper or reproducing the code.

---

## License

This project is released under the [MIT License](https://github.com/dongxingning/SHA_GCL_for_SGG/blob/master/LICENSE).

