# Flat’n’Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation

## Paper Abstract:
This repository is published along with a paper Flat’n’Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation.
We provided 1212 human and 887 robot demonstrations of flattening and folding 44 garments across 8 categories. Also, we establish two new benchmarks for grasping point prediction and subtask decomposition.

## Dataset:
* Dataset can be download at: https://gla-my.sharepoint.com/:f:/g/personal/2658047z_student_gla_ac_uk/Ekgx_o8q6ZZBtxusMwrP8zoBt2FkZL9vwq3hqe5c1CyHSQ. Some samples of data and data description are also provided.
* For each data sequences, rgbd images and action information are provided. Intrinsic parameters and external paramerter (origin is headset) of three cameras are provided. Pointclouds can be generated through preate_graspvisulize.ipynb in Pointcloud folder.
* Besides, 

## Grasping point prediction benchmark
* Code for two baselines: [Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Pointbert](https://github.com/Julie-tang00/Point-BERT) are provided in Grasping_point folder.

## Subtask decomposition benchmark
* Code of data processing, groundtruth extraction and metrics evaluation for one baseline: [Universal Visual Decomposer](https://zcczhang.github.io/UVD/) are provided in UVD folder.
