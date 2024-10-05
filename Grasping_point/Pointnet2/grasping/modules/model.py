import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
# from grasping.modules.dataloader import denormalize, normalize
# from grasping.modules.loss import L_label, L_pos_mask, L_rot_mask
from modules.dataloader import denormalize, normalize
from modules.loss import L_label, L_pos_mask, L_rot_mask
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class get_model(nn.Module):
    def __init__(self,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, 128)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        # print("Shape after fc3:", x.shape) 

        return x


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss

class Grasp_PointNet2(nn.Module):
    def __init__(self, config, freeze_bert=True):
        super().__init__()
        self.point_trans = get_model(config)
        self.point_trans.load_state_dict(torch.load(config.pointnet2_ckpt), strict=False)
        self.lambda_label = config.lambda_label
        self.lambda_rot = config.lambda_rot
        self.lambda_pos = config.lambda_pos
        self.set_std(config.data_norm_path)

        if freeze_bert:
            for param in self.point_trans.parameters():
                param.requires_grad = False

        self.label_classifer = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        self.feature_predictor = nn.Sequential(
            # nn.Linear(config.trans_dim * 2, 256),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 18)
        )

    def set_std(self, load_path):
        data_norm = np.load(load_path)
        self.rot_std = torch.tensor(data_norm['rot6d_std'], dtype=torch.float32).to(device)
        self.rot_mean = torch.tensor(data_norm['rot6d_mean'], dtype=torch.float32).to(device)
        self.pos_std = torch.tensor(data_norm['pos_std'], dtype=torch.float32).to(device)
        self.pos_mean = torch.tensor(data_norm['pos_mean'], dtype=torch.float32).to(device)

    def forward(self, points):
        points = points.transpose(2, 1)
        x = self.point_trans(points)
        # print("Output shape from point_trans:", x.shape)
        label = self.label_classifer(x)
        results = self.feature_predictor(x)
        pose_left = results[..., :3]
        rot6d_left = results[..., 3:9]
        pose_right = results[..., 9:12]
        rot6d_right = results[..., 12:18]

        pose_right = denormalize(pose_right, self.pos_mean, self.pos_std)
        pose_left = denormalize(pose_left, self.pos_mean, self.pos_std)
        rot6d_left = denormalize(rot6d_left, self.rot_mean, self.rot_std)
        rot6d_right = denormalize(rot6d_right, self.rot_mean, self.rot_std)

        return {'label': label, 'pose_left': pose_left, 'rot6d_left': rot6d_left, 'pose_right': pose_right,
                'rot6d_right': rot6d_right}

    def get_loss(self, pred, gt):
        target_label = gt['label'].to(device)
        target_rotqut = gt['orient'].to(device)
        target_pos = gt['pos'].to(device)
        # print('output label:', pred['label'])
        rot6d = torch.cat((pred['rot6d_left'].unsqueeze(1), pred['rot6d_right'].unsqueeze(1)), dim=1)
        pos = torch.cat((pred['pose_left'].unsqueeze(1), pred['pose_right'].unsqueeze(1)), dim=1)

        bce_loss = L_label(target=target_label, output=pred['label'])
        rot_loss = L_rot_mask(target=target_rotqut, output=rot6d, mask=target_label)
        pos_loss = L_pos_mask(target=target_pos, output=pos, mask=target_label)
        total_loss = self.lambda_label*bce_loss + self.lambda_rot*rot_loss + self.lambda_pos*pos_loss
        return {'total_loss': total_loss, 'label_loss': bce_loss, 'rot_loss': rot_loss, 'pos_loss': pos_loss}