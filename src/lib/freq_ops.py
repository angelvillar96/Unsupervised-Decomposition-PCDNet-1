"""
Operations in the frquency domain
"""

import numpy as np
import torch
import torch.fft as fftlib

EPSILON = 1e-15


def add_imaginary(frame):
    """ Adding imaginary channel to the frames. Needed to call the FFT """
    # adding 0j to the frames: channel 0 (real), channel 1 (imaginary)
    frame_im = torch.stack((frame, torch.zeros(frame.shape)), dim=-1)
    return frame_im


def complex_modulus(x):
    """ Modulus of a complex signal """
    assert x.shape[-1] == 2
    abs = torch.sqrt(x[..., 0]*x[..., 0] + x[..., 1]*x[..., 1])
    return abs


def complex_prod(x, y):
    """
    Multiplication of two complex numbers in cartesian coords
        re(x*y) = re(x)*re(y) - im(x)*im(y)
        im(x*y) = im(x)*re(y) + re(x)*im(y)
    """
    x_re, x_im = x.real, x.imag
    y_re, y_im = y.real, y.imag

    prod_re = x_re * y_re - x_im * y_im
    prod_im = x_re * y_im + x_im * y_re
    prod = torch.stack([prod_re, prod_im], dim=-1)
    prod = torch.view_as_complex(prod)
    return prod


def compute_displacement(f_corr, type="torch", norm=True):
    """
    Measuring displacement by finding peaks in the Correlation matrix
    """
    shape = f_corr.shape

    # findig y-x coordinates of peak
    if(type == "numpy"):
        n_rows, n_cols = shape[-2], shape[-1]
        peak_id = np.argmax(f_corr)
        delta_y, delta_x = np.unravel_index(peak_id, shape)
    else:
        batch_size, n_rows, n_cols = shape[0], shape[-2], shape[-1]
        peak_id = torch.argmax(f_corr.reshape([batch_size, -1]), dim=-1)
        delta_x, delta_y = peak_id % n_cols, peak_id // n_cols

    return delta_y, delta_x


def fft(frames, type="torch"):
    """
    Computing the 2-dim FFT of the given frame(s)
    """
    if(type == "numpy"):
        freq_frames = np.fft.fft2(frames)
    else:
        # assert frames.shape[-1] == 2, "Frames must have dim-2 in last channel (re{} and im{})"
        freq_frames = fftlib.fft2(frames)
    return freq_frames


def freq_correlate(mat1, mat2, type="torch"):
    """
    Correlation between two images in the frequency domain, e.g., a (normalized) Hadamrd product
    """
    if(type == "numpy"):
        f_corr = (mat1 * mat2.conjugate()) / (np.abs(mat1) * np.abs(mat2) + EPSILON)
    else:
        assert mat1.shape[-1] == 2 and mat2.shape[-1] == 2
        conj_mat2 = torch.stack([mat2[...,0], -1 * mat2[...,1]], dim=-1)
        f_corr = complex_prod(mat1, conj_mat2)
        # f_corr = (mat1 * torch.conj(mat2)) / (torch.abs(mat1) * torch.abs(mat2) + EPSILON)
    return f_corr


def get_angles(d1, d2=None):
    """
    Obtainibg angular basis for log-polar transform

    Args:
    -----
    d1, d2: integer
        dimensionality of the matrix to generate

    Returns:
    --------
    angels: torch Tensor
        Tensor (d2*d1) containing a linspace of angles.
        This is used for the cartesian to log-polar mapping
    """
    d2 = d2 if d2 is not None else d1
    angles = torch.linspace(0, 2*np.pi, d1)
    angles = angles.repeat(d2, 1).T
    return angles


def get_basis(size, second_size=None, device=None):
    """
    Obtaining a torch Tensor matrix with coordnates for the fourier phase transform matrix
    """
    second_size= size if second_size is None else second_size
    coords = torch.linspace(-np.fix(size/2), np.ceil(size/2)-1, steps=size)

    axes = tuple(range(coords.ndim))
    shift = [dim // 2 for dim in coords.shape]
    f_bins = torch.roll(coords, shift, axes)
    f_matrix = f_bins.repeat(second_size, 1)

    if(device is not None):
        f_matrix = f_matrix.to(device)

    return f_matrix


def get_exp(phase):
    """
    Cartesian form of the complex explonentail that shifts by phase (delta in freq. domain)
    exp(2 pi j phi) = cos(2 pi phi) + j sin(2 pi phi)
    """
    return torch.stack([torch.cos(phase), torch.sin(phase)], -1)


def ifft(freq_frames, type="torch"):
    """
    Computing 2-dim inverse FFT of the given frequency-domain frames
    """
    if(type == "numpy"):
        image_frames = np.fft.ifft2(freq_frames)
    else:
        # assert freq_frames.shape[-1] == 2, "Frames must have dim-2 in last channel (re{} and im{})"
        image_frames = fftlib.ifft2(freq_frames)
    return image_frames


def outer(a, b):
    """
    Torch implementation of numpy's outer.
    """
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul*b_mul


def high_pass(shape):
    """
    Return high pass filter to be multiplied with fourier transform.
    Improves the registration performance for rotation and scale
    """
    x = outer(
        torch.cos(torch.linspace(-np.pi/3, np.pi/3, shape[-2])),
        torch.cos(torch.linspace(-np.pi/3, np.pi/3, shape[-1]))
    )
    return 2 * (1 - x)
    return (1.0 - x) * (2.0 - x)


#
