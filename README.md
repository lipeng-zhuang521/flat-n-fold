# Flat’n’Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation
<img src="https://github.com/user-attachments/assets/01531133-96ee-4625-9643-ad416733e4dc" width="700" height="500">

## Paper Abstract
This repository is published along with a paper Flat’n’Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation.
We provided 1212 human and 887 robot demonstrations of flattening and folding 44 garments across 8 categories. Also, we establish two new benchmarks for grasping point prediction and subtask decomposition.

## Harware
Intrinsic parameters and external paramerters (origin is headset) of three cameras, STL files of the grippers are provided in hardware folder.

## Dataset
* Dataset can be download at: https://gla-my.sharepoint.com/:f:/g/personal/2658047z_student_gla_ac_uk/Ekgx_o8q6ZZBtxusMwrP8zoBt2FkZL9vwq3hqe5c1CyHSQ. Some samples of data and data description are also provided.
* For each data sequences, rgbd images and action information are provided. Pointclouds can be generated through preate_graspvisulize.ipynb in Pointcloud folder. Merged pointclouds of three cameras and action visulization can be gernerated through merge_three_camera.py in Pointcloud folder.
* Besides, it's worth noting that the origin of action information of our robot demonstration is base of Baxter. The orgin of action information of our human demonstration is headset. The transformations of transition and rotation are also provided in Pointcloud folder.

## Grasping point prediction benchmark
* Code for two baselines: [Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Pointbert](https://github.com/Julie-tang00/Point-BERT) are provided in Grasping_point folder.

## Subtask decomposition benchmark
* Code of data processing, groundtruth extraction and metrics evaluation for one baseline: [Universal Visual Decomposer](https://zcczhang.github.io/UVD/) are provided in UVD folder.

## Acknowledgments
* [Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [Pointbert](https://github.com/Julie-tang00/Point-BERT)
* [Universal Visual Decomposer](https://zcczhang.github.io/UVD/)

## Citation
if you are using Flat'n'Fold dataset, please consider citing the follwing paper:

