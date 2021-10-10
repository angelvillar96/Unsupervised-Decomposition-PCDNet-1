"""
Transformer and Samples modules.
Currently only implements the Color Module of PCDNet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorTransformer(nn.Module):
    """
    Module that takes an image and a template and aims to apply
    the correct color to the protype
    """

    def __init__(self, channels=1):
        """ """
        super().__init__()
        self.channels = channels

        self.color_weight = None
        self.out = None
        self.logits = None

        self.eye = nn.Parameter(torch.eye(channels, channels))
        self.color_regressor = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(12, channels),
        )
        return

    def get_color_params(self):
        """ Getting the color parameters computed by the module """
        colors = self.color_weight.sum(2).clip(0, 1)
        return colors

    def forward(self, img, templates, masks):
        """ Forward pass """
        B, C, H, W = img.shape
        T = templates.shape[1]

        # masking relevant area and regressing color parameters
        img = img.unsqueeze(1)
        masked_img = (img * masks).clamp(0,1)
        self.masked_imgs = masked_img
        masked_img = masked_img.view(B*T, C, H, W)
        logits = self.color_regressor(masked_img)

        # color transformer: diagonal matrix * color channels + bias
        # weight, bias = logits[:, :C], logits[:, C:]
        weight = logits
        weight = weight.view(B, T, -1, C) * self.eye + self.eye
        # bias = bias.view(B,T, C, 1, 1).expand(-1, -1, -1, H, W)

        # output = torch.einsum('btij, btjkl -> btikl', weight, templates) + bias
        output = torch.einsum('btij, btjkl -> btikl', weight, templates)
        output = output.clamp(0, 1)

        self.logits = logits.view(B, T, -1, C)
        self.color_weight = weight
        self.out = output

        return output

#
