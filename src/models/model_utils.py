"""
Model utils
"""
import numpy as np
from scipy import signal
import torch
import torch.nn as nn

def init_weights(m):
    """ Initializing the parameters of a nn.Module """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def identity_init(m):
    """ Initializing layers with zero-bias and identity filters """
    if type(m) == nn.Conv2d:
        K = torch.Tensor([
                [0 ,0, 0],
                [0, 1 ,0],
                [0, 0 ,0]
            ]).unsqueeze(0)
        out_k = m.out_channels
        in_k = m.in_channels
        m.weight = torch.nn.Parameter(K.view(1, 1, 3, 3).repeat(out_k, in_k, 1, 1))
        m.bias.data.fill_(0)
    print(m)
    return

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def freeze_params(model):
    """ Freezing model params to avoid updates in backward pass"""
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_params(model):
    """ Unfreezing model params to allow for updates during backward pass"""
    for param in model.parameters():
        param.requires_grad = True
    return model

def get_norm_layer(norm="batch"):
    """ Selecting norm layer by name """
    assert norm in ["batch", "instance", "group", "layer", "", None]
    if norm == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm == "instance":
        norm_layer = nn.InstanceNorm2d
    elif norm == "group":
        norm_layer = nn.GroupNorm
    elif norm == "layer":
        norm_layer = nn.LayerNorm
    elif norm == "" or norm is None:
        norm_layer = nn.Identity
    return norm_layer

def create_gaussian_weights(img_size, n_channels, std):
    """
    Creating Gaussina kernel with given std
    """
    g1d_h = signal.gaussian(img_size[0], std)
    g1d_w = signal.gaussian(img_size[1], std)
    g2d = np.outer(g1d_h, g1d_w)
    gauss = torch.from_numpy(g2d).unsqueeze(0).expand(n_channels, -1, -1).float()
    return gauss


class SoftClamp(nn.Module):
    """
    Module that clamps values to a range [0,1], but smoother to allow
    for a better gradient flow
    """
    def __init__(self, alpha=0.01):
        """ Module intializer"""
        super().__init__()
        self.alpha = alpha
        return

    def forward(self, x):
        """ Forward pass """
        x0 = torch.min(x, torch.zeros(x.shape, device=x.device))
        x1 = torch.max(x - 1, torch.zeros(x.shape, device=x.device))
        y = torch.clamp(x, 0, 1) + self.alpha * x0 + self.alpha * x1
        return y

#
