"""
Phase-Correlation (PC) Transformer for image decomposition and
future frame prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fftlib

import lib.freq_ops as freq_ops
import models.DifferentiableBlocks as DiffBlocks


class PCCell(nn.Module):
    """
    Implementation of the Phase Correltation Cell for affine transform parameter estimation

    Args:
    -----
    transform_type: string
        affine transformation this PC-Cell takes care off (e.g translate, rot-scale)
    use_cnn: boolean
        if True, a CNN is used to process the inputs
    L: integer
        number of objects to consider. Corresponds to the number of peaks to find in
        the correlation matrix.
            - Synthesis, L=1 to move an object to the next frame
            - Analysis: L=max_obj, finding a candidate position for every possible object
    """

    def __init__(self, transform_type="translate", use_cnn=False, L=1, **kwargs):
        """ Initializer of the Phase Correlation Cell """
        assert transform_type in ["translate", "rot-scale"]
        super().__init__()

        self.transform_type = transform_type
        self.L = L
        self.topK = DiffBlocks.TopK(k=L)
        self.argMax = DiffBlocks.Argmax(dim=1)
        self.params = kwargs
        self.grayscale = False if "grayscale" not in kwargs else kwargs["grayscale"]
        self.channels = 3 if "channels" not in kwargs else kwargs["channels"]

        return

    def forward(self, x, t, renorm=True):
        """ Computing the affine transformation parameters """

        n_temps = t.shape[0]
        n_cols = x.shape[-1]
        n_rows = x.shape[-2]
        b_size = x.shape[0]
        device = x.device

        # padding the pattern
        self.pad_size_y, self.pad_size_x = min(n_rows, 40), min(n_cols, 40)

        x_pad = F.pad(
                x, (self.pad_size_x, self.pad_size_x, self.pad_size_y, self.pad_size_y)
            )
        t_pad = F.pad(
                t, (self.pad_size_x, self.pad_size_x, self.pad_size_y, self.pad_size_y)
            )

        # Phase Correlation on the updated inputs to estimate a correlation matrix
        # where the peaks correspond to the most likeliy transforms
        f_corr = self.estimate_correlation(signal=x_pad, pattern=t_pad)

        # peak-picking
        peaks, vals = self.peak_picking(
            f_corr=f_corr, n_cols=n_cols + 2 * self.pad_size_x,
            n_rows=n_rows + 2 * self.pad_size_y, b_size=b_size,
            n_temps=n_temps, get_max=True
        )

        # reshaping/repat peaks and prototypes to compute templates
        t_ = torch.cat([torch.stack([p]*self.L) for p in t])
        t_ = t_.repeat(b_size, 1, 1, 1)
        peaks_ = peaks.view(b_size * self.L * n_temps, 2)

        # shifting prototype
        self.peaks = peaks_.clone()
        self.n_cols, self.n_rows = n_cols, n_rows
        self.b_size, self.device = b_size, device
        template = self.translate_pattern(
                pattern=t_, peaks=peaks_, device=device,
                n_cols=n_cols, n_rows=n_rows
            )

        return template, peaks

    def translate_pattern(self, pattern, peaks, device, n_cols, n_rows):
        """
        Translating the pattern using the estimated displacement peaks.

        We need to take special care of the circular simmetry of the FFT.
        Everything that goes out of bounds, returns through the opposite side.
        We address this issue by padding and cropping.
        """

        # padding the pattern
        pattern_pad = F.pad(
                pattern, (self.pad_size_x, self.pad_size_x,
                          self.pad_size_y, self.pad_size_y)
            )

        # phase matrix
        Rows = n_rows + self.pad_size_y * 2
        Cols = n_cols + self.pad_size_x * 2
        f_matrix_x = freq_ops.get_basis(size=Cols, second_size=Rows, device=device)
        f_matrix_x = f_matrix_x.view(1, 1, Rows, Cols)
        f_matrix_y = freq_ops.get_basis(size=Rows, second_size=Cols, device=device)
        f_matrix_y = f_matrix_y.unsqueeze(0).unsqueeze(0)
        self.f_matrix_x = f_matrix_x

        # shift matrix
        delta_y = peaks[..., 0].to(device).view(-1, 1, 1, 1)
        delta_x = peaks[..., 1].to(device).view(-1, 1, 1, 1)
        Translation_matrix_x = torch.exp(-1j * 2 * np.pi * f_matrix_x * delta_x)
        Translation_matrix_y = torch.exp(-1j * 2 * np.pi * f_matrix_y.transpose(-1,-2) * delta_y)

        # translating by phase adding: elementwise product in the frequency domain
        Template = fftlib.fft2(pattern_pad) * Translation_matrix_x * Translation_matrix_y
        template = fftlib.ifft2(Template).abs()

        # removing padded areas
        if(self.pad_size_x > 0 and self.pad_size_y > 0):
            template = template[..., self.pad_size_y:-self.pad_size_y,
                                self.pad_size_x:-self.pad_size_x]

        return template

    def translate_masks(self, masks):
        """
        Using last precomputed peaks in order to translate the corresponding masks
        """
        masks_ = torch.cat([torch.stack([m] * self.L) for m in masks])
        masks_ = masks_.repeat(self.b_size, 1, 1, 1)

        masks_ = self.translate_pattern(
                pattern=masks_, peaks=self.peaks, device=self.device,
                n_cols=self.n_cols, n_rows=self.n_rows
            )
        return masks_

    def estimate_correlation(self, signal, pattern):
        """
        Estimating the correlation between an image and a pattern using phase correlation

        Args:
        -----
        signal, pattern: torch Tensors
            torch tensors corresponding to the imput image and the corresponding normalizer pattern

        Returns:
        --------
        f_corr: torch Tensor
            inverse FFT from which we sample the displacement as the argmax
        """

        # using Phase Correlation to estimate displacement heatmap
        if(self.L > 1):  # Watch-Out with this, might be a future source of problems
            signal = signal.unsqueeze(1)

        Signal = fftlib.fft2(signal)
        Pattern = fftlib.fft2(pattern)
        F_corr = freq_ops.complex_prod(Signal, Pattern.conj()) / \
            ((freq_ops.complex_prod(Signal, Pattern.conj())).abs() + 1e-8)
        f_corr = fftlib.ifft2(F_corr).abs()

        return f_corr

    def peak_picking(self, f_corr, n_cols, n_rows, b_size, n_temps, get_max=True):
        """
        Selecting the positions with the maximum value using differentiable
        or custom operations
        """

        peaks = []

        if(self.L == 1):  # selecting the arg-max to estimate the translation
            idx, vals = self.argMax(f_corr.flatten(1))
            # removing dependency on color channel
            idx = idx % (n_cols * n_rows) if not self.grayscale else idx
            cur_peaks = torch.Tensor([[i // n_cols, i % n_cols] for i in idx])
            peaks.append(cur_peaks)
            peaks = torch.stack(peaks, dim=1)
        else:  # Case for detecting the TopK peaks for estimating the candidate templates
            temp = f_corr.view(b_size * n_temps, -1)
            idx, vals = self.topK(temp)
            # removing dependency on color channel
            idx = idx % (n_cols * n_rows) if not self.grayscale else idx
            peaks = torch.Tensor([[i // n_cols, i % n_cols]
                                 for cand_id in idx for i in cand_id])
            peaks = peaks.view(b_size, n_temps, self.L, 2)

        peaks[..., 0] = peaks[..., 0] / n_rows
        peaks[..., 1] = peaks[..., 1] / n_cols

        return peaks, vals

    def normalize_deltas(self, delta_x, delta_y):
        """
        Remapping the transformation parameters for reconstruction of the
        future frames
        """
        if self.transform_type == "translate":
            #  handling cyclic boundary conditions: [0, 1] -> [-0.5, 0.5]
            remap_x = delta_x / 2
            remap_y = delta_y / 2
            id_x = torch.where(delta_x < 0)
            delta_x[id_x] = 1 + delta_x[id_x]
            id_y = torch.where(delta_y < 0)
            delta_y[id_y] = 1 + delta_y[id_y]
            remap_x = delta_x
            remap_y = delta_y

        elif self.transform_type == "rot-scale":
            remap_x = delta_x
            remap_y = delta_y

        return remap_x, remap_y

    def renormalize_peaks(self, peaks):
        """
        Processing the computed displacement in order to invert transforms and
        understand pose parameters
        """

        if self.transform_type == "translate":
            #  handling cyclic boundary conditions: [0, 1] -> [-1, 1]
            ids = torch.where(peaks > 0.5)
            peaks[ids] = peaks[ids] - 1
            peaks = peaks * 2

        elif self.transform_type == "rot-scale":
            peaks = peaks

        return peaks

    def __repr__(self):
        """ Overriding the string so that we display all relevant elements"""
        list_mod = super().__repr__()[:-2]

        transform_string = f"PCCell(\n  (affine transform): {self.transform_type}, L={self.L}"
        list_mod = list_mod.replace("PCCell(", transform_string)
        list_mod += "\n)"
        return list_mod


#
