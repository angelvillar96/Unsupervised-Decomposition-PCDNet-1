"""
Methods for registering affine transforms using frequency domain methods
"""

import numpy as np
import torch
import torch.fft as fftlib
import torch.nn.functional as F

import lib.freq_ops as freq_ops
EPSILON = 1e-15


def estimate_translation(signal, pattern):
    """
    Estimating the translation between an image and a pattern using phase correlation

    Args:
    -----
    signal, pattern: torch Tensors
        torch tensors corresponding to the imput image and the corresponding normalizer pattern

    Returns:
    --------
    delta_y, delta_x: floats
        displacement (in norm pixels) between the norm pattern and the pattern in the image
    """

    Signal = fftlib.fft2(signal)
    Pattern = fftlib.fft2(pattern)
    F_corr = freq_ops.complex_prod(Signal, Pattern.conj())
    f_corr = fftlib.ifft2(F_corr).abs()

    rows, cols = signal.shape[-2], signal.shape[-1]
    delta_y, delta_x = freq_ops.compute_displacement(f_corr)
    delta_y, delta_x = delta_y / rows, delta_x / cols

    return delta_y, delta_x


def get_freq_norm(img, delta_x, delta_y):
    """
    Obtaining the freq-domain norm estimate to feed into the norm-model

    Args:
    -----
    img: torch Tensor
        Tensor containing the object to translate
    delta_x, delta_y: floats
        magnitude of the translation (normalized to the range [0,1])
        across the X and Y axis respectively

    Return:
    -------
    Norm: torch Tensor
        frequency domain estimate of the normalized pattern. Basically corresponds to
        the input signal shifted by the amount indicated in the symbolic pattern
    """

    device = img.device
    size = img.shape[-1]

    f_matrix = freq_ops.get_basis(size).unsqueeze(0).to(device)
    f_matrix_x = torch.stack([f_matrix * d for d in delta_x])
    f_matrix_y = torch.stack([f_matrix.transpose(-2, -1) * d for d in delta_y])

    translation_matrix_x = torch.exp(-1j * 2 * np.pi * f_matrix_x).conj().to(device)
    translation_matrix_y = torch.exp(-1j * 2 * np.pi * f_matrix_y).conj().to(device)
    # translation
    Img = fftlib.fft2(img)
    Norm = Img * translation_matrix_x * translation_matrix_y
    return Norm


def translate_pytorch_2d(img, delta_x, delta_y, **kwargs):
    """
    Translation for Torch Tensors across X & Y axis simultaneously

    Args:
    -----
    img: torch Tensor
        Tensor containing the object to translate
    delta_x, delta_y: torch Tensors
        magnitude of the translation (normalized to the range [0,1])
        across the X and Y axis respectively
    """

    if(len(img.shape) == 3):
        img = img.unsqueeze(0)
    batch_size, _, _, size = img.shape
    device = img.device

    # obtaining the translation matrices
    f_matrix = freq_ops.get_basis(size, device=device).view(1, 1, size, size)
    f_matrix = f_matrix.repeat(batch_size, 1, 1, 1)
    delta_x = delta_x.view(batch_size, 1, 1, 1)
    delta_y = delta_y.view(batch_size, 1, 1, 1)

    translation_matrix_x = torch.exp(-1j * 2 * np.pi * f_matrix * delta_x)
    translation_matrix_y = torch.exp(-1j * 2
                                     * np.pi * f_matrix.transpose(-1, -2) * delta_y)

    if("undo_transform" in kwargs and kwargs["undo_transform"] is True):
        translation_matrix_x = translation_matrix_x.conj()
        translation_matrix_y = translation_matrix_y.conj()

    # computing translation by phase correlation ==> freq-domain product
    Img = fftlib.fft2(img)
    Trans_Img = Img * translation_matrix_x * translation_matrix_y
    image_trans = fftlib.ifft2(Trans_Img)

    return image_trans.float()


#
