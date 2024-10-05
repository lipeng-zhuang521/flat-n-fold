import os
import math
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
# from timm.scheduler import CosineLRScheduler
# from grasping.modules.dataloader import GraspDataset
# from grasping.modules.model import Grasp_Bert
# from grasping.args import Arguments
# from grasping.metrics import cal_accuracy, pos_error, rot_error
from modules.dataloader import GraspDataset,GraspDatasetCombined
from modules.model import Grasp_PointNet2
from args import Arguments
from metrics import cal_accuracy, pos_error, rot_error
from torch import cat, no_grad, tensor
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def basic_train():
    args = Arguments('./cfgs/GraspNet', filename='model.yaml')
    train_files = ['lsshirt_train.csv', 'lstshirt_train.csv', 'napkin_train.csv', 'pant_train.csv', 'ssshirt_train.csv',
                   'sstshirt_train.csv', 'sweater_train.csv', 'towel_train.csv']
    dataset = GraspDatasetCombined(root_dir='./grasping/data', csv_list=train_files, from_csv=True, first_load=False,
                                   num_points=4096)
    dataset_size = len(dataset)
    train_dataset_size = int(0.9 * dataset_size)
    valid_dataset_size = dataset_size - train_dataset_size
    # train_dataset_size = dataset_length - test_dataset_size - valid_dataset_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_dataset_size, valid_dataset_size])
    train_data_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)
    valid_data_loader = DataLoader(valid_set, batch_size=32, shuffle=True, pin_memory=True)

    grasp_model = Grasp_PointNet2(config=args, freeze_bert=True).to(device)

    optim = torch.optim.Adam(grasp_model.parameters(), lr=0.001, betas=(0.9, 0.999),eps=1e-08,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.8)

    for epoch in range(200):
        print(f'epoch {epoch}')
        grasp_model.train()
        train_total_loss = 0
        trian_label_loss = 0
        trian_pos_loss = 0
        trian_rot_loss = 0
        for idx, (data) in enumerate(train_data_loader):
            point = data['point'].to(device)
            optim.zero_grad()
            result = grasp_model(point)
            loss = grasp_model.get_loss(pred=result, gt=data)
            loss['total_loss'].backward()
            optim.step()
            train_total_loss += loss['total_loss'].item()
            trian_label_loss += loss['label_loss'].item()
            trian_pos_loss += loss['pos_loss'].item()
            trian_rot_loss += loss['rot_loss'].item()

        avg_loss = train_total_loss / len(train_data_loader)
        avg_label_loss = trian_label_loss / len(train_data_loader)
        avg_pos_loss = trian_pos_loss / len(train_data_loader)
        avg_rot_loss = trian_rot_loss / len(train_data_loader)

        print(f'Train: Total Loss: {avg_loss}, Label Loss: {avg_label_loss}, Pos Loss: {avg_pos_loss}, '
              f'Rot Loss: {avg_rot_loss}')

        sched.step(epoch)

        grasp_model.eval()
        with torch.no_grad():
            val_total_loss = 0
            val_label_loss = 0
            val_pos_loss = 0
            val_rot_loss = 0
            for idx, (data) in enumerate(valid_data_loader):
                point = data['point'].to(device)
                result = grasp_model(point)
                loss = grasp_model.get_loss(pred=result, gt=data)
                val_total_loss += loss['total_loss'].item()
                val_label_loss += loss['label_loss'].item()
                val_pos_loss += loss['pos_loss'].item()
                val_rot_loss += loss['rot_loss'].item()

            avg_loss = val_total_loss / len(valid_data_loader)
            avg_label_loss = val_label_loss / len(valid_data_loader)
            avg_pos_loss = val_pos_loss / len(valid_data_loader)
            avg_rot_loss = val_rot_loss / len(valid_data_loader)
            print(f'Val: Total Loss: {avg_loss}, Label Loss: {avg_label_loss}, Pos Loss: {avg_pos_loss}, '
                  f'Rot Loss: {avg_rot_loss}')

    ### Evaluate the model ###

    test_files = ['lsshirt_test.csv', 'lstshirt_test.csv', 'napkin_test.csv', 'pant_test.csv', 'ssshirt_test.csv',
                   'sstshirt_test.csv', 'sweater_test.csv', 'towel_test.csv']

    for file in test_files:
        test_set = GraspDataset(root_dir='./grasping/data', from_csv=True, first_load=False, num_points=4096,
                                csv_file=file)
        test_dataset_size = len(test_set)
        test_data_loader = DataLoader(test_set, batch_size=32, shuffle=True, pin_memory=True)
        print(f'Evaluate on {file}')
        grasp_model.eval()
        with torch.no_grad():
            total_instance = test_dataset_size
            total_acc_num = 0
            total_acc_hand = 0
            total_pos_err = 0
            total_rot_err = 0
            for idx, (data) in enumerate(test_data_loader):
                point = data['point'].to(device)
                result = grasp_model(point)
                rot6d = torch.cat((result['rot6d_left'].unsqueeze(1), result['rot6d_right'].unsqueeze(1)), dim=1)
                pos = torch.cat((result['pose_left'].unsqueeze(1), result['pose_right'].unsqueeze(1)), dim=1)
                acc_num, acc_index, acc_hand = cal_accuracy(output=result['label'], target=data['label'].to(device))
                pos_err = pos_error(output=pos, target=data['pos'].to(device), label=data['label'].to(device),
                                    index=acc_index)
                rot_err = rot_error(output=rot6d, target=data['orient'].to(device),
                                    label=data['label'].to(device), index=acc_index)
                total_acc_num += acc_num
                total_acc_hand += acc_hand
                total_pos_err += pos_err
                total_rot_err += rot_err
            print('total_acc_num', total_acc_num)
            print('total_acc_hand', total_acc_hand)
            print('total_pos_err', total_pos_err)
            print('total_rot_err', total_rot_err)
            avg_acc = total_acc_num / total_instance
            avg_pos_err = total_pos_err / total_acc_hand
            avg_rot_erro = total_rot_err / total_acc_hand

            print(f'Test: Accuracy: {avg_acc}, Pos Error: {avg_pos_err}, Rot Error: {avg_rot_erro}')
    return


if __name__ == '__main__':
    basic_train()