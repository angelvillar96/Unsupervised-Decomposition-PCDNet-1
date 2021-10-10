"""
Implementation of common non-differentiable operations so that the gradients
can be copied during the backward pass
"""

import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = ["TopK", "Argmax"]


class DifferentiableArgmax(Function):
    """
    Differentiable ArgMax function
    We copy the gradients during the backward pass
    """
    @staticmethod
    def forward(ctx, i, dim=-1):
        idx = torch.argmax(i, dim=dim)
        vals = torch.max(i, dim=dim)
        return idx, vals
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class DifferentiableTopK(Function):
    """
    Differentiable Top-K function
    We copy the gradients during the backward pass
    """
    @staticmethod
    def forward(ctx, i, k):
        vals, top_ids = torch.topk(i, k=k)
        return top_ids, vals
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Argmax(nn.Module):
    """ Argmax that allows for backpropagations """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, i):
        idx, vals = DifferentiableArgmax.apply(i, self.dim)
        return idx, vals


class TopK(nn.Module):
    """ Top-K that allows to copy gradients """
    def __init__(self, k):
        super().__init__()
        self.k= k

    def forward(self, i):
        top_ids, vals = DifferentiableTopK.apply(i, self.k)
        return top_ids, vals



#
