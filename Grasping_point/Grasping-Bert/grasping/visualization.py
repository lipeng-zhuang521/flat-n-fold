import os
import math
import tqdm
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from grasping.modules.dataloader import GraspDataset, GraspDatasetCombined
from grasping.modules.model import Grasp_Bert
from grasping.args import Arguments
from grasping.modules.dataloader import fast_fps
from grasping.modules.rotations import rotation_6d_to_matrix, matrix_to_quaternion
from grasping.metrics import cal_accuracy, pos_error, rot_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
original_pos_left, original_ori_left, original_pos_right, original_ori_right, grasp_pos_left, grasp_pos_right, left_gripper, right_gripper

("{'x': -0.1456676695834728, 'y': -1.2590139437643546, 'z': -0.9290434782925981}",
 "{'x': 0.5602054509566237, 'y': 0.5427648226666211, 'z': -0.22375376241026906, 'w': 0.5843889576313076}",
 "{'x': 0.30289243958529166, 'y': -1.222470178866121, 'z': -0.6237287812502067}",
 "{'x': 0.40094215171962727, 'y': 0.46764667440977237, 'z': -0.6794043793880544, 'w': 0.39869997260627976}",
 "{'x': -0.1456676695834728, 'y': -1.2590139437643546, 'z': -0.9290434782925981}",
 "{'x': 0.30289243958529166, 'y': -1.222470178866121, 'z': -0.6237287812502067}",######
 1,0)
'''

def visual():



    pcd_files = './grasping/data/robot/grasp_point_merge_downsample/lsshirt_bluedot/masked_1722266692102892760_downsampled_0.0045.pcd'
    pcd = o3d.io.read_point_cloud(pcd_files)
    ply_files = './grasping/visual.ply'
    o3d.io.write_point_cloud(ply_files, pcd)
    '''
    point = np.asarray(pcd.points)
    point = fast_fps(point, 4096)
    point = torch.from_numpy(point).float().to(device)
    point = point.unsqueeze(0)
    left_pos = torch.tensor([-0.1456676695834728, -1.2590139437643546, -0.9290434782925981]).to(device)
    left_ori = torch.tensor([0.5602054509566237, 0.5427648226666211, -0.22375376241026906, 0.5843889576313076]).to(device)
    right_pos = torch.tensor([0.30289243958529166, -1.222470178866121, -0.6237287812502067]).to(device)
    right_ori = torch.tensor([0.40094215171962727, 0.46764667440977237, -0.6794043793880544, 0.39869997260627976]).to(device)

    # pcd_data = o3d.io.read_point_cloud(os.path.join(point_cloud_path, cloth_type, cloud_name))
    args = Arguments('./cfgs/GraspNet', filename='model.yaml')
    model = Grasp_Bert(config=args, freeze_bert=True).to(device)
    model20_path = './grasping/ckpt/grasp_model_robot4096_20.pth'
    model.load_state_dict(torch.load(model20_path))
    model.eval()
    result = model(point)
    pred_label = result['label']
    pred_left_pos = result['pose_left']
    pred_left_ori = result['rot6d_left']
    pred_left_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_left_ori))
    pred_right_pos = result['pose_right']
    pred_right_ori = result['rot6d_right']
    pred_right_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_right_ori))
    print('20% model - pred_label:', pred_label, 'pred_left_pos:', pred_left_pos, 'pred_left_ori:', pred_left_ori,
          'pred_right_pos:', pred_right_pos, 'pred_right_ori:', pred_right_ori)

    model50_path = './grasping/ckpt/grasp_model_robot4096_50.pth'
    model.load_state_dict(torch.load(model50_path))
    model.eval()
    result = model(point)
    pred_label = result['label']
    pred_left_pos = result['pose_left']
    pred_left_ori = result['rot6d_left']
    pred_left_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_left_ori))
    pred_right_pos = result['pose_right']
    pred_right_ori = result['rot6d_right']
    pred_right_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_right_ori))
    print('50% model - pred_label:', pred_label, 'pred_left_pos:', pred_left_pos, 'pred_left_ori:', pred_left_ori,
          'pred_right_pos:', pred_right_pos, 'pred_right_ori:', pred_right_ori)

    model100_path = './grasping/ckpt/grasp_model_robot4096_100.pth'
    model.load_state_dict(torch.load(model100_path))
    model.eval()
    result = model(point)
    pred_label = result['label']
    pred_left_pos = result['pose_left']
    pred_left_ori = result['rot6d_left']
    pred_left_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_left_ori))
    pred_right_pos = result['pose_right']
    pred_right_ori = result['rot6d_right']
    pred_right_ori = matrix_to_quaternion(rotation_6d_to_matrix(pred_right_ori))
    print('100% model - pred_label:', pred_label, 'pred_left_pos:', pred_left_pos, 'pred_left_ori:', pred_left_ori,
          'pred_right_pos:', pred_right_pos, 'pred_right_ori:', pred_right_ori)
    '''
    return

if __name__ == '__main__':
    # pcd_files = './grasping/data/robot/grasp_point_merge_downsample/lsshirt_bluedot/masked_1722266692102892760_downsampled_0.0045.pcd'
    # visual()
    # stl_file = './grasping/Cloth_gripper_100.stl'
    # mesh = o3d.io.read_triangle_mesh(stl_file)
    # pcd = o3d.io.read_point_cloud(pcd_files)
    # o3d.visualization.draw_geometries([stl_file])

    for file in os.listdir('./grasping/point_cloud2'):
        print(file)
        pcd_files = os.path.join('./grasping/point_cloud2', file)
        new_file = file.replace('.pcd', '.ply')
        pcd = o3d.io.read_point_cloud(pcd_files)
        o3d.io.write_point_cloud(new_file, pcd)