import torch
import torch.nn as nn
import torch.nn.functional as F
# from grasping.modules.rotations import rotation_6d_to_matrix, quaternion_to_matrix
# from grasping.modules.loss import GeodesicLoss
from modules.rotations import rotation_6d_to_matrix, quaternion_to_matrix
from modules.loss import GeodesicLoss


def cal_accuracy(output, target):
    """
    Args:
        target: labels in the shape B x 2.
        output: labels in the shape B x 2.

    Returns:
        accuracy: accuracy of the output.
        correct_indices: indices of the correct predictions.
        hands_num: number of hands in the batch.
    """

    binary_output = (output > 0.5).int()
    matching_rows = (binary_output == target).all(dim=1)
    correct_index = torch.nonzero(matching_rows)
    hands_num = target[correct_index].sum()
    acc_num = correct_index.size(0)  # / output.size(0)
    return acc_num, correct_index, hands_num


def rot_error(output, target, label, index):
    """
    Args:
        target: joint local positions in the shape B x 2 x 3.
        output: joint local positions in the shape B x 2 x 3.
        mask: mask in the shape B x 2.
    Returns:
        total rotation loss evaluated on the hand rotation matrix.
    """
    output_selected = output[index]
    target_selected = target[index]
    output6d = rotation_6d_to_matrix(output_selected)  # (B, T, 3, 3)
    target6d = quaternion_to_matrix(target_selected)  # (B, T, 3, 3)
    label_selected = label[index]

    mask = label_selected.unsqueeze(-1).unsqueeze(-1)
    mask_bool = mask.to(torch.bool).to(output.device)

    output_masked = torch.masked_select(output6d, mask_bool)
    target_masked = torch.masked_select(target6d, mask_bool)
    criterion_geo = GeodesicLoss()
    loss = criterion_geo(output_masked.view(-1, 3, 3), target_masked.view(-1, 3, 3))
    total_rot_error = loss.sum()
    return total_rot_error


def pos_error(output, target, label, index):
    """
    Args:
        target: joint local positions in the shape B x 2 x 3.
        output: joint local positions in the shape B x 2 x 3.
        mask: mask in the shape B x 2.
    Returns:
        total position loss evaluated on the hand positions.
    """
    output_selected = output[index]
    target_selected = target[index]
    label_selected = label[index]
    mask = label_selected.unsqueeze(-1)
    criterion_loss = torch.nn.L1Loss(reduction='none')
    output_masked = output_selected * mask
    target_masked = target_selected * mask

    error = criterion_loss(output_masked, target_masked)
    total_pos_error = error.sum() / 3
    return total_pos_error