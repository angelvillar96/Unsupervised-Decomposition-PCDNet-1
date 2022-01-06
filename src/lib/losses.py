"""
Loss functions and loss utils
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_type="mse"):
    """
    Setting up an object for computing the loss given the loss type
    """
    available_losses = ["mse", "l2", "mae", "l1", "cross_entropy", "ce"]
    assert loss_type in available_losses, f"""ERROR! Loss {loss_type} not available.
            Use one of the following: {available_losses}"""

    if loss_type in ["mse", "l2"]:
        loss = nn.MSELoss()
    elif loss_type in ["mae", "l1"]:
        loss = nn.L1Loss()
    if loss_type in ["cross_entropy", "ce"]:
        loss = nn.CrossEntropyLoss()

    return loss


class LossAdding(nn.Module):
    """ """

    def __init__(self, lambdas):
        """ """
        super().__init__()
        self.lambdas = torch.Tensor(lambdas)

    def forward(self, losses):
        """ """
        loss = sum([loss * lamb for loss, lamb in zip(losses, self.lambdas)])
        return loss


def proto_l1(proto):
    """ L1 Regulariziation on prototypes """
    l1_reg = proto.abs().mean()
    return l1_reg


def proto_l2(proto):
    """ L2 Regulariziation on prototypes """
    l2_reg = proto.pow(2).mean()
    return l2_reg


def total_variation(img):
    """ Computing the total variation regularization. Enforces smoothness in an image """
    img = F.pad(img, (3, 3, 3, 3))
    reg_rows = torch.mean(torch.abs(img[..., :-1] - img[..., 1:]))
    reg_cols = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    reg_loss = reg_rows + reg_cols
    return reg_loss


#
