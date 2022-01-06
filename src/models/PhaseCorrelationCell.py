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
    L: integer
        Number of objects to consider. Corresponds to the number of peaks to find in the correlation matrix.
    """

    def __init__(self, L=1, **kwargs):
        """ Initializer of the Phase Correlation Cell """
        super().__init__()
        self.L = L
        self.topK = DiffBlocks.TopK(k=L)
        self.grayscale = kwargs.get("grayscale", False)
        return

    def forward(self, img, protos):
        """
        Computing the best transformation parameters, and aligning prototypes to objects with phase shift

        Args:
        -----
        img: torch Tensor
            image to extract the prototype candidates from. Shape is (B, C, H, W)
        protos: torch Tensor
            object prototypes to transform. Shape is (N_protos, 1, Proto_H, Proto_W)

        Returns:
        --------
        peaks: torch.Tensor
            Location of the correlation peaks used to transform prototypes. Shape is (B, N_protos, L, 2)
        """
        n_protos, proto_size = protos.shape[0], protos.shape[2:]
        (b_size, n_channels, n_rows, n_cols), device = img.shape, img.device
        self.img_size = (n_rows, n_cols)
        self.b_size, self.device = b_size, device

        # padding the pattern
        self.pad_size_y, self.pad_size_x = min(n_rows, 40), min(n_cols, 40)
        img_pad = F.pad(img, (self.pad_size_x, self.pad_size_x, self.pad_size_y, self.pad_size_y))
        protos_pad = F.pad(protos, (self.pad_size_x, self.pad_size_x, self.pad_size_y, self.pad_size_y))
        self.protos_pad = protos_pad

        # Phase Correlation on the updated inputs to estimate a correlation matrix
        # where the peaks correspond to the most likeliy transforms
        f_corr = self.estimate_correlation(signal=img_pad, pattern=protos_pad)
        f_corr = f_corr.view(b_size, n_protos, n_channels, img_pad.shape[-2], img_pad.shape[-1])
        # finding the peaks and their corresponding patch from the Fourier correlation matrix
        self.f_corr = f_corr
        peaks = self.peak_picking(f_corr=f_corr)
        peaks = peaks.view(b_size * self.L * n_protos, 2)
        self.peaks = peaks
        return peaks

    def translate_pattern(self, pattern, peaks):
        """
        Translating the pattern using the estimated displacement peaks.

        We need to take special care of the circular simmetry of the FFT.
        Everything that goes out of bounds, returns through the opposite side.
        We address this issue by padding and cropping.
        """
        # padding the pattern to same size as used for correlation
        pattern_rows, pattern_cols = pattern.shape[-2:]
        pattern_pad = F.pad(pattern, (self.pad_size_x, self.pad_size_x, self.pad_size_y, self.pad_size_y))

        # frequency basis matrix
        Rows = pattern_rows + self.pad_size_y * 2
        Cols = pattern_cols + self.pad_size_x * 2
        f_matrix_x = freq_ops.get_basis(size=Cols, second_size=Rows, device=self.device)
        f_matrix_x = f_matrix_x.view(1, 1, Rows, Cols)
        f_matrix_y = freq_ops.get_basis(size=Rows, second_size=Cols, device=self.device)
        f_matrix_y = f_matrix_y.view(1, 1, Cols, Rows)

        # phase shift matrices
        delta_y = peaks[..., 0].view(-1, 1, 1, 1)
        delta_x = peaks[..., 1].view(-1, 1, 1, 1)
        Translation_matrix_x = torch.exp(-1j * 2 * np.pi * f_matrix_x * delta_x)
        Translation_matrix_y = torch.exp(-1j * 2 * np.pi * f_matrix_y.transpose(-1, -2) * delta_y)

        # translating by phase adding: elementwise product in the frequency domain
        Template = fftlib.fft2(pattern_pad) * Translation_matrix_x * Translation_matrix_y
        template = fftlib.ifft2(Template).abs()
        self.temp = template

        # removing padded areas
        if(self.pad_size_x > 0 and self.pad_size_y > 0):
            template = template[..., self.pad_size_y:-self.pad_size_y, self.pad_size_x:-self.pad_size_x]
        return template

    def translate_masks_protos(self, pattern, peaks):
        """
        Using computed peaks to translate the corresponding pattern (prototypes or masks)
        """
        pattern_ = torch.cat([torch.stack([p] * self.L) for p in pattern])  # TODO: optimizable
        pattern_ = pattern_.repeat(self.b_size, 1, 1, 1)
        pattern_ = self.translate_pattern(pattern=pattern_, peaks=peaks)
        return pattern_

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
        signal = signal.unsqueeze(1)

        Signal = fftlib.fft2(signal)
        Pattern = fftlib.fft2(pattern)
        complex_prod = freq_ops.complex_prod(Signal, Pattern.conj())
        F_corr = complex_prod / (complex_prod.abs() + 1e-12)
        f_corr = fftlib.ifft2(F_corr).abs()
        return f_corr

    def peak_picking(self, f_corr):
        """
        Selecting the positions with the maximum value using differentiable
        or custom operations
        """
        B, n_protos, n_channels, H, W = f_corr.shape
        # finding location and magnitude of correlation peaks for each patch
        f_corr = f_corr.reshape(B * n_protos, n_channels * H * W)
        peaks, vals = self.topK(f_corr)
        peaks = peaks % (H * W) if not self.grayscale else peaks  # removing color dependency on locations

        # mapping back to (x, y) coords
        best_peaks = torch.stack([torch.floor(peaks / W), peaks % W], dim=-1)
        best_peaks = best_peaks.view(B, n_protos, self.L, 2).float()
        self.vals = vals.view(B, n_protos, self.L)

        best_peaks[..., 0] = best_peaks[..., 0] / H
        best_peaks[..., 1] = best_peaks[..., 1] / W

        return best_peaks

    def __repr__(self):
        """ Overriding the string so that we display all relevant elements"""
        list_mod = super().__repr__()[:-2]

        transform_string = f"PCCell(L={self.L})"
        list_mod = list_mod.replace("PCCell(", transform_string)
        return list_mod


#
