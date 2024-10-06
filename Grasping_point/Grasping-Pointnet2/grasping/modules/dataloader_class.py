import os
import csv
import re
import numpy as np
import pandas as pd
import fpsample
import torch
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset
# from grasping.modules.rotations import quaternion_to_matrix, matrix_to_rotation_6d
from modules.rotations import quaternion_to_matrix, matrix_to_rotation_6d

def normalize(tensor, mean=None, std=None):
    """
    Args:
        tensor: (B, T, ...)

    Returns:
        normalized tensor with 0 mean and 1 standard deviation, std, mean
    """
    if mean is None or std is None:
        # std, mean = torch.std_mean(tensor, dim=0, unbiased=False, keepdim=True)
        std, mean = torch.std_mean(tensor, dim=(0, 1), unbiased=False, keepdim=True)
        std[std == 0.0] = 1.0

        return (tensor - mean) / std, mean, std

    return (tensor - mean) / std


def denormalize(tensor, mean, std):
    """
    Args:
        tensor: B x T x D
        mean:
        std:
    """
    return tensor * std + mean

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_point_sample(point, npoints):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    centroids = np.random.choice(N, npoints)
    point = point[centroids.astype(np.int32)].astype(np.float32)
    return point


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    new_point = point[centroids.astype(np.int32)].astype(np.float32)
    return new_point


def fast_fps(point, npoints):
    new_index = fpsample.bucket_fps_kdline_sampling(point, npoints, 5)
    new_point = point[new_index.astype(np.int32)].astype(np.float32)
    return new_point


def read_tracker_from_row(row):
    left_label = row['left_gripper']
    if left_label == 0:
        left_pos = np.zeros((3))
    else:
        left_pos_str = row['grasp_pos_left']
        left_pos = re.findall(r"[-+]?\d*\.\d+|\d+", left_pos_str)
        left_pos = np.array(list(map(float, left_pos)))

    right_label = row['right_gripper']
    if right_label == 0:
        right_pos = np.zeros((3))
    else:
        right_pos_str = row['grasp_pos_right']
        right_pos = re.findall(r"[-+]?\d*\.\d+|\d+", right_pos_str)
        right_pos = np.array(list(map(float, right_pos)))

    left_ori_str = row['original_ori_left']
    right_ori_str = row['original_ori_right']

    left_ori = re.findall(r"[-+]?\d*\.\d+|\d+", left_ori_str)
    left_ori = np.array(list(map(float, left_ori)))
    right_ori = re.findall(r"[-+]?\d*\.\d+|\d+", right_ori_str)
    right_ori = np.array(list(map(float, right_ori)))

    orient = np.stack((left_ori, right_ori), axis=0).astype(np.float32)
    pos = np.stack((left_pos, right_pos), axis=0).astype(np.float32)
    label = np.array([left_label, right_label]).astype(np.float32)

    return pos, orient, label

def read_tracker_from_index(file, index):
    result = file[file['time'] == str(index)]
    # result = file[file['pc_time'] == index]
    if result.empty:
        print('No such index in tracker.csv:', index)
        return None, None, None

    left_label = result.iloc[0]['left_gripper']

    if left_label == 0:
        left_pos = np.zeros((3))
    else:
        left_pos_str = result.iloc[0]['grasp_pos_left']
        left_pos = re.findall(r"[-+]?\d*\.\d+|\d+", left_pos_str)
        left_pos = np.array(list(map(float, left_pos)))

    right_label = result.iloc[0]['right_gripper']
    if right_label == 0:
        right_pos = np.zeros((3))
    else:
        right_pos_str = result.iloc[0]['grasp_pos_right']
        right_pos = re.findall(r"[-+]?\d*\.\d+|\d+", right_pos_str)
        right_pos = np.array(list(map(float, right_pos)))

    left_ori_str = result.iloc[0]['original_ori_left']
    right_ori_str = result.iloc[0]['original_ori_right']

    left_ori = re.findall(r"[-+]?\d*\.\d+|\d+", left_ori_str)
    left_ori = np.array(list(map(float, left_ori)))
    right_ori = re.findall(r"[-+]?\d*\.\d+|\d+", right_ori_str)
    right_ori = np.array(list(map(float, right_ori)))

    orient = np.concatenate((left_ori, right_ori), axis=0).astype(np.float32)
    pos = np.concatenate((left_pos, right_pos), axis=0).astype(np.float32)
    label = np.array([left_label, right_label]).astype(int)

    return pos, orient, label

class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, num_points=4096, from_csv=False, first_load=False):
        label_list = []
        poses_list = []
        orient_list = []
        point_list = []
        point_cloud_path = os.path.join(root_dir, 'grasp_point_merge_downsample')
        df = pd.read_csv(os.path.join(root_dir,'grasp_data_garment', csv_file))  # 替换为你的CSV文件名
        if from_csv:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                # print(row['pc_time'])
                subname = str(row['subname'])
                match = re.match(r"([^_]+_[^_]+)", subname)
                cloth_type = match.group(1) if match else None
                cloud_name = 'masked_' + str(row['pc_time']) + '_downsampled_0.0045.pcd'
                if not os.path.exists(os.path.join(point_cloud_path, cloth_type, cloud_name)):
                    print('No such file:', os.path.join(point_cloud_path, cloth_type, cloud_name))
                    continue

                pos, rot, label = read_tracker_from_row(row)
                pcd_data = o3d.io.read_point_cloud(os.path.join(point_cloud_path, cloth_type, cloud_name))
                point = np.asarray(pcd_data.points)
                if point.shape[0] < num_points:
                    # print('File:', os.path.join(point_cloud_path, cloth_type, cloud_name))
                    # print('Raw Points:', point.shape[0])
                    new_point = random_point_sample(point, num_points)
                else:
                    new_point = fast_fps(point, num_points)

                # new_point = fpsample.bucket_fps_kdline_sampling(point, num_points, 5)

                poses_list.append(pos)
                orient_list.append(rot)
                label_list.append(label)
                point_list.append(new_point)
        else:
            for file in tqdm(os.listdir(point_cloud_path)):
                pcd_data = o3d.io.read_point_cloud(os.path.join(point_cloud_path, file))
                point = np.asarray(pcd_data.points)
                new_point = farthest_point_sample(point, num_points)
                index = file.split('_')[1]
                pos, rot, label = read_tracker_from_index(df, index)
                if pos is None:
                    continue

                poses_list.append(pos)
                orient_list.append(rot)
                label_list.append(label)
                point_list.append(new_point)

        self.label = torch.from_numpy(np.array(label_list))
        self.pos = torch.from_numpy(np.array(poses_list))
        self.orient = torch.from_numpy(np.array(orient_list))
        self.point_cloud = torch.from_numpy(np.array(point_list))

        if first_load:

            rot6d_data = matrix_to_rotation_6d(quaternion_to_matrix(self.orient))
            rot6d_data = rot6d_data.view(-1, 6)
            pose_data = self.pos.view(-1, 3)

            _, rot_mean, rot_std = normalize(rot6d_data)
            _, pos_mean, pos_std = normalize(pose_data)
            np.savez(os.path.join(root_dir, 'mean_std.npz'), rot6d_mean=rot_mean, rot6d_std=rot_std, pos_mean=pos_mean,
                     pos_std=pos_std)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        point = self.point_cloud[idx]
        label = self.label[idx]
        pos = self.pos[idx]
        orient = self.orient[idx]

        return {'point': point, 'label': label, 'pos': pos, 'orient': orient}


class GraspDatasetCombined(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_list, num_points=4096, from_csv=False, first_load=False):
        label_list = []
        poses_list = []
        orient_list = []
        point_list = []
        point_cloud_path = os.path.join(root_dir, 'grasp_point_merge_downsample')
        for csv_file in csv_list:
            df = pd.read_csv(os.path.join(root_dir, 'grasp_data_garment', csv_file))  # 替换为你的CSV文件名
            if from_csv:
                for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                    # print(row['pc_time'])
                    subname = str(row['subname'])
                    match = re.match(r"([^_]+_[^_]+)", subname)
                    cloth_type = match.group(1) if match else None
                    cloud_name = 'masked_' + str(row['pc_time']) + '_downsampled_0.0045.pcd'
                    if not os.path.exists(os.path.join(point_cloud_path, cloth_type, cloud_name)):
                        print('No such file:', os.path.join(point_cloud_path, cloth_type, cloud_name))
                        continue

                    pos, rot, label = read_tracker_from_row(row)
                    pcd_data = o3d.io.read_point_cloud(os.path.join(point_cloud_path, cloth_type, cloud_name))
                    point = np.asarray(pcd_data.points)
                    if point.shape[0] < num_points:
                        # print('File:', os.path.join(point_cloud_path, cloth_type, cloud_name))
                        # print('Raw Points:', point.shape[0])
                        new_point = random_point_sample(point, num_points)
                    else:
                        new_point = fast_fps(point, num_points)

                    # new_point = fpsample.bucket_fps_kdline_sampling(point, num_points, 5)

                    poses_list.append(pos)
                    orient_list.append(rot)
                    label_list.append(label)
                    point_list.append(new_point)
            else:
                for file in tqdm(os.listdir(point_cloud_path)):
                    pcd_data = o3d.io.read_point_cloud(os.path.join(point_cloud_path, file))
                    point = np.asarray(pcd_data.points)
                    new_point = farthest_point_sample(point, num_points)
                    index = file.split('_')[1]
                    pos, rot, label = read_tracker_from_index(df, index)
                    if pos is None:
                        continue

                    poses_list.append(pos)
                    orient_list.append(rot)
                    label_list.append(label)
                    point_list.append(new_point)

        self.label = torch.from_numpy(np.array(label_list))
        self.pos = torch.from_numpy(np.array(poses_list))
        self.orient = torch.from_numpy(np.array(orient_list))
        self.point_cloud = torch.from_numpy(np.array(point_list))

        if first_load:

            rot6d_data = matrix_to_rotation_6d(quaternion_to_matrix(self.orient))
            rot6d_data = rot6d_data.view(-1, 6)
            pose_data = self.pos.view(-1, 3)

            _, rot_mean, rot_std = normalize(rot6d_data)
            _, pos_mean, pos_std = normalize(pose_data)
            np.savez(os.path.join(root_dir, 'mean_std.npz'), rot6d_mean=rot_mean, rot6d_std=rot_std, pos_mean=pos_mean,
                     pos_std=pos_std)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        point = self.point_cloud[idx]
        label = self.label[idx]
        pos = self.pos[idx]
        orient = self.orient[idx]

        return {'point': point, 'label': label, 'pos': pos, 'orient': orient}



if __name__ == '__main__':

    # left_ori = np.array([left_ori_dic['x'], left_ori_dic['y'], left_ori_dic['z'], left_ori_dic['w']])
    # print(left_ori)
    point_path = './grasping/modules/test.pcd'
    pcd = o3d.io.read_point_cloud(point_path)
    point = np.asarray(pcd.points)
    color = np.array(pcd.colors)
    # print(point.shape)
    new_point, new_color = farthest_point_sample(point, color, 8192)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_point)
    new_pcd.colors = o3d.utility.Vector3dVector(new_color)
    # o3d.visualization.draw_geometries([new_pcd])
    o3d.io.write_point_cloud('./grasping/modules/test_8192.pcd', new_pcd)

    # print(new_point.shape)