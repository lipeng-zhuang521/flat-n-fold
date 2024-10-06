import torch
import torch.nn as nn
from .rotations import rotation_6d_to_matrix, quaternion_to_matrix

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2, epsilon=1e-7):
        """ Compute the geodesic distance between two rotation matrices.
        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # cos = (m.diagonal(dim1=-2, dim2=-1).sum(-1) -1) /2
        # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
        cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')


def L_rot(target, output):
    """
    Args:
        target: rotation matrices in the shape B x 2 x 3 x 3.
        output: rotation matrices in the shape B x 2 x 3 x 3.

    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    output = rotation_6d_to_matrix(output)  # (B, T, 3, 3)
    target = rotation_6d_to_matrix(target)  # (B, T, 3, 3)

    criterion_geo = GeodesicLoss()
    loss = criterion_geo(output.view(-1, 3, 3), target.view(-1, 3, 3))

    return loss


def L_pos(target, output):
    """
    Args:
        target: joint local positions in the shape B x 2 x 3.
        output: joint local positions in the shape B x 2 x 3.

    Returns:
        reconstruction loss evaluated on the joint local positions.
    """
    criterion_rec = nn.L1Loss()
    loss = criterion_rec(output, target)

    return loss


def L_rot_mask(target, output, mask):
    """
    Args:
        target: rotation matrices in the shape B x 2 x 4.
        output: rotation matrices in the shape B x 2 x 6.
        mask: mask in the shape B x 2.
    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    hand_num = mask.sum()

    output = rotation_6d_to_matrix(output)  # (B, T, 3, 3)
    target = quaternion_to_matrix(target)  # (B, T, 3, 3)

    mask_bool = mask.to(torch.bool).to(output.device)
    output_masked = torch.masked_select(output, mask_bool)
    target_masked = torch.masked_select(target, mask_bool)

    criterion_geo = GeodesicLoss()
    loss = criterion_geo(output_masked.view(-1, 3, 3), target_masked.view(-1, 3, 3))

    average_rot_loss = loss.sum() / (hand_num)

    return average_rot_loss


def L_pos_mask(target, output, mask):
    """
    Args:
        target: joint local positions in the shape B x 2 x 3.
        output: joint local positions in the shape B x 2 x 3.
        mask: mask in the shape B x 2.
    Returns:
        reconstruction loss evaluated on the joint local positions.
    """
    mask = mask.unsqueeze(-1)
    hand_num = mask.sum()
    criterion_loss = torch.nn.L1Loss(reduction='none')

    output_masked = output * mask
    target_masked = target * mask
    loss = criterion_loss(output_masked, target_masked)
    average_pos_loss = loss.sum() / (hand_num * 3)

    return average_pos_loss


def L_label(target, output):
    """
    Args:
        target: labels in the shape B x 2.
        output: labels in the shape B x 2.

    Returns:
        classification loss evaluated on the labels.
    """
    # criterion_ce = nn.CrossEntropyLoss()
    criterion_ce = nn.BCELoss()
    left_loss = criterion_ce(output[..., 0], target[..., 0])
    right_loss = criterion_ce(output[..., 1], target[..., 1])
    total_loss = left_loss + right_loss
    return total_loss