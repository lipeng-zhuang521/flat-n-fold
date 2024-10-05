import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.dvae import Group
from models.dvae import Encoder

from models.build import MODELS
from models.Point_BERT import TransformerEncoder
from grasping.modules.dataloader import denormalize, normalize
from grasping.modules.loss import L_label, L_pos_mask, L_rot_mask
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        '''
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )
        '''

        # self.build_loss_func()


    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        feature = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.cls_head_finetune(concat_f)
        return feature


class Grasp_Bert(nn.Module):
    def __init__(self, config, freeze_bert=True):
        super().__init__()
        self.point_trans = PointTransformer(config)
        self.point_trans.load_state_dict(torch.load(config.point_bert_ckpt), strict=False)
        self.lambda_label = config.lambda_label
        self.lambda_rot = config.lambda_rot
        self.lambda_pos = config.lambda_pos
        self.set_std(config.data_norm_path)

        if freeze_bert:
            for param in self.point_trans.parameters():
                param.requires_grad = False

        self.label_classifer = nn.Sequential(
            nn.Linear(config.trans_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

        self.feature_predictor = nn.Sequential(
            nn.Linear(config.trans_dim * 2, 256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 18)
        )

    def set_std(self, load_path):
        data_norm = np.load(load_path)
        self.rot_std = torch.tensor(data_norm['rot6d_std'], dtype=torch.float32).to(device)
        self.rot_mean = torch.tensor(data_norm['rot6d_mean'], dtype=torch.float32).to(device)
        self.pos_std = torch.tensor(data_norm['pos_std'], dtype=torch.float32).to(device)
        self.pos_mean = torch.tensor(data_norm['pos_mean'], dtype=torch.float32).to(device)

    def forward(self, points):
        x = self.point_trans(points)
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


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output