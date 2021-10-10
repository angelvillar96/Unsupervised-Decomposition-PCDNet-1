"""
Methods related to object prototypes and templates
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

import lib.metrics as metrics

PROTO_VALUES = ["blobs", "zeros", "zeros_", "shapes", "noise", "constant", "constant_"]

def get_proto_values():
    """ """
    return PROTO_VALUES

def init_prototypes(mode="noise", num_protos=2, requires_grad=True, proto_size=64, channels=1, **kwargs):
    """
    Initializing the object prototypes

    Args:
    -----
    mode: string
        mode used to intializer the object prototypes:
            blobs: Some blob-like shape centered in the fraame
            zeros: Empty frame
            shapes: Some geometrical priors: (ball, square, triangle)
            noise: Gaussian Random Noise N+(0,1)
    num_protos: integer
        Number of prototypes to intialize
    requires_grad: boolean
        If True, prototypes are meant to be learned
    proto_size: integer
        side of the prototype template
    """
    assert mode in PROTO_VALUES
    N = num_protos
    C = channels
    C = 1

    if(isinstance(proto_size,int)):
        proto_size = (proto_size,proto_size)

    if(mode == "blobs"):
        assert proto_size == 64
        sigma = 1.6
        x = torch.arange(40)
        yy, xx = torch.meshgrid(x, x)
        yy, xx = (yy - 20) / 20, (xx - 20) / 20
        blob = (xx.pow(2) + yy.pow(2)).pow(1/sigma)
        blob = 1 - blob/blob.max()
        protos = torch.zeros(N,1,proto_size[0],proto_size[1])
        protos[:,:,12:-12,12:-12] = blob
    elif(mode == "centers"):
        raise NotImplementedError()
    elif(mode == "zeros"):
        protos = torch.zeros(N,1,proto_size[0],proto_size[1])
        center0, center1 = proto_size[0] // 2, proto_size[1]//2
        protos[:,:,center0-1:center0+1, center1-1:center1+1] = 1
    elif(mode == "zeros_"):
        protos = torch.zeros(N,1,proto_size[0],proto_size[1])
        center0, center1 = proto_size[0] // 2, proto_size[1]//2
    elif(mode == "constant"):
        protos = torch.ones(N,1,proto_size[0],proto_size[1]) * 0.2
        center0, center1 = proto_size[0] // 2, proto_size[1]//2
        protos[:,:,center0:center0+1, center1:center1+1] = 1
    elif(mode == "constant_"):
        protos = torch.ones(N,1,proto_size[0],proto_size[1]) * 0.2
    elif(mode == "shapes"):
        assert proto_size == 64
        aux = np.zeros((64,64))
        # ball
        proto_ball = cv2.circle(aux, (32, 32), int(8), 1, -1)
        proto_ball = torch.Tensor(proto_ball)
        # rect
        proto_rect = cv2.rectangle(aux, (24,24), (40,40), 1, -1)
        proto_rect = torch.Tensor(proto_rect)
        # triangle
        aux = np.zeros((64,64))
        coords = np.array([[32,24], [24,40], [40,40]])
        coords = coords.reshape((-1, 1, 2))
        proto_tri = cv2.fillPoly(aux, [coords], 255, 1)
        proto_tri = torch.Tensor(proto_tri)
        protos = torch.stack([proto_ball, proto_rect, proto_tri])
        if(N < 3):
            protos = protos[:N]
        elif(N > 3):
            extra_balls = proto_ball.repeat(N-3,1,1)
            protos = torch.cat([protos, extra_balls], dim=0)
        protos = protos.unsqueeze(1)

    elif(mode == "noise"):
        proto_size = torch.Tensor([proto_size[0], proto_size[1]]).int()
        thr = (proto_size * 1 / 5).int()
        border = ((proto_size - thr) // 2).int()
        noise = torch.randn(N,C, thr[0].item(), thr[1].item()).clamp(0,1) / 3
        protos = torch.zeros(N,C,proto_size[0].item(),proto_size[1].item())
        try:
            protos[:,:,border[0]:-border[0],border[1]:-border[1]] = noise
        except:
            protos[:,:,border[0]:-border[0]-1,border[1]:-border[1]-1] = noise

    # we add some very small noise to avoid same initialization
    noise = (torch.rand(protos.shape) - 0.5) / 40
    protos = (protos + noise).clamp(0, 1)
    protos.requires_grad = requires_grad
    protos = nn.Parameter(protos)

    return protos


def init_proto_masks(num_protos=2, requires_grad=True, proto_size=64):
    """
    Initializing binary masks for object prototypes

    Args:
    -----
    num_protos: integer
        Number of masks to intialize
    requires_grad: boolean
        If True, masks are meant to be learned
    proto_size: integer
        side of the masks template
    """
    # initalizing masks with ones
    proto_masks = torch.ones(num_protos, 1, proto_size[0], proto_size[1])

    # add gradients for backpropagation
    proto_masks.requires_grad = requires_grad
    proto_masks = nn.Parameter(proto_masks)

    return proto_masks


def init_background(proto_size=64, channels=1, requires_grad=True):
    """
    Initialiizing a tensor for a learned background
    """

    if(isinstance(proto_size,int)):
        proto_size = (proto_size,proto_size)
    background = torch.ones(channels, proto_size[0], proto_size[1]) * 0  # 0.2
    background = nn.Parameter(background)
    background.requires_grad = requires_grad

    return background


def preload_background(background, dataset, n_imgs=1000):
    """ Initializing a background by averaging images"""

    new_background = torch.zeros(background.shape)
    for i in range(n_imgs):
        img = torch.Tensor(dataset[i])
        new_background += img
    new_background = new_background / n_imgs
    new_background.requires_grad = background.requires_grad
    new_background = nn.Parameter(new_background.to(background.device))
    return new_background


#
