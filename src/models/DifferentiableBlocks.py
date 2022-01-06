"""
Implementation of common non-differentiable operations so that the gradients
can be copied during the backward pass
"""

import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = ["TopK"]


class DifferentiableTopK(Function):
    """
    Differentiable Top-K function
    We copy the gradients during the backward pass
    """
    @staticmethod
    def forward(ctx, i, k):
        """ Selecting top-K values and saving indices for backward pass """
        top_vals, top_ids = torch.topk(i, k=k)
        ctx.save_for_backward(i, top_ids)
        top_ids = top_ids.float()
        return top_ids.clone(), top_vals.clone()

    @staticmethod
    def backward(ctx, grad_output, other):
        """ Passing gradients back only on the TopK selected positions """
        i, top_ids = ctx.saved_tensors
        grads = torch.zeros(*i.shape).to(grad_output.device)
        for b, grad in enumerate(grad_output):
            cur_ids = top_ids[b]
            grads[b, cur_ids] = grad_output[b]
        return grads.clone(), None


class TopK(nn.Module):
    """
    Wrapper for the Top-K peak-picking functionality.

    Args:
    -----
    k: integer
        Number of peaks to take in each feature map
    diff: bool
        if True, gradients are copied to allow for backpropagation. Similar to Max-Pool
    """

    def __init__(self, k, diff=False):
        """ Module initializer """
        super().__init__()
        self.k = k
        self.diff = diff
        return

    def forward(self, i):
        """
        Peak picking

        Args:
        -----
        i: torch Tensor
            Tensor to find the maximum (correlation) peaks. Shape is (B, D)

        Returns:
        --------
        top_ids, top_vals: torh Tensor
            location and magnutes of the selected peaks. Shapes are (B, k) for both
        """
        if self.diff:
            top_ids, top_vals = DifferentiableTopK.apply(i, self.k)
        else:
            top_vals, top_ids = torch.topk(i, k=self.k)
        top_ids = top_ids.float()
        return top_ids, top_vals


#
