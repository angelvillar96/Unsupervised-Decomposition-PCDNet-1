"""
Utils methods for data visualization
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors


def make_gif(sequence, context, savepath, pad=2, add_title=True, interval=55, verbose=False):
    """
    Creating a GIF displaying the sequence

    Args:
    ------
    sequence: torch Tensor
        Tensor containing the sequence of images (T,C,H,W)
    context: integer
        number of frames used for context (they have green border)
    savepath: string
        path where GIF is stored
    pad: integer
        number of pixels to pad with color
    add_title: boolean
        if True, Frame number is indicated in the image title
    interval: integer
        number of milliseconds in between frames (55=18fps)
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    gif_frames = sequence.permute(0,2,3,1).detach().numpy()
    n_frames = len(gif_frames)
    colors = dict((i,"green") if i < context else (i,"red") for i in range(n_frames))
    update_ = lambda i: update(frame=gif_frames[i], pad=pad, color=colors[i], ax=ax,
                               title=f"Frame {i}", verbose=verbose, idx=i)

    anim = FuncAnimation(fig, update_, frames=np.arange(n_frames), interval=interval)
    ax.axis("off")
    anim.save(savepath, dpi=50, writer='imagemagick')
    return

def update(frame, color="green", pad=2, title="", ax=None, verbose=False, idx=0):
    """
    Auxiliar function to plot gif frames
    """
    if(verbose and idx % 50 == 0):
        print(f"Processing frame {idx+1}")  # displaying some processing progress
    disp_frame = add_border(frame, color=color, pad=pad)
    ax.imshow(disp_frame)
    ax.set_title(title)
    return ax

def add_border(x, color, pad=1):
    """
    Adding green/red border to gif-frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color: string
        'red' or 'green'. Color of the border
    pad: integer
        number of pixels to pad each side
    """
    w = x.shape[1]
    nc = x.shape[-1]
    px = np.zeros((w+2*pad, w+2*pad, 3))
    if color == 'red':
        px[:,:,0] = 0.7
    elif color == 'green':
        px[:,:,1] = 0.7
    if nc == 1:
        for c in range(3):
            px[pad:w+pad, pad:w+pad, c] = x
    else:
        px[pad:w+pad, pad:w+pad, :] = x
    return px


def visualize_sequence(sequence, savepath=None, add_title=True, add_axis=False, n_cols=10,
                       size=3, n_channels=3, **kwargs):
    """
    Visualizing a grid with all frames from a sequence
    """

    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])

    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1,2,0).cpu().detach()
        if(n_channels == 1):
            f = f[...,0]
        a.imshow(f, **kwargs)
        if(add_title):
            if("titles" in kwargs):
                a.set_title(kwargs["titles"][i])
            else:
                a.set_title(f"Frame {i}")


    # removing axis
    if(not add_axis):
        for col in range(n_cols):
            for row in range(n_rows):
                a = ax[row, col] if n_rows > 1 else ax[col]
                a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax


def visualize_prototypes(protos, savepath=None, add_title=True, add_axis=False,
                         n_cols=10, size=3, n_channels=3, **kwargs):
    """
    Displaying the learned object prototypes
    """

    n_protos = len(protos)
    n_rows = int(np.ceil(n_protos / n_cols))
    n_cols = min(n_protos, 10)

    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3*n_cols, 3*n_rows)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])

    for i in range(n_protos):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        proto = protos[i].permute(1,2,0).cpu().detach()
        if(n_channels == 1):
            proto = proto[...,0]
        a.imshow(proto, vmin=0, vmax=1, cmap="gray")
        if(add_title):
            if("titles" in kwargs):
                a.set_title(kwargs["titles"][i])
            else:
                a.set_title(f"Prototype {i+1}")
        if(not add_axis):
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax


def visualize_recons(recons, frames, savepath=None, add_title=True, add_axis=False, n_channels=3, **kwargs):
    """
    Displaying the learned object prototypes
    """

    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(3*3, 3*3)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])

    for i in range(3):
        ax[i,0].imshow(recons[i].cpu().permute(1,2,0))
        ax[i,1].imshow(frames[i].cpu().permute(1,2,0))
        ax[i,2].imshow((recons[i].cpu() - frames[i].cpu()).permute(1,2,0).abs().clamp(0,1), vmin=0, vmax=1)

        if(not add_axis):
            for a in ax[i]: a.axis("off")

    ax[0,0].set_title("Reconstruction")
    ax[0,1].set_title("Target")
    ax[0,2].set_title("Error")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax


def filter_curve(f, k=5):
    """ Using a 1D low-pass filter to smooth a loss-curve """
    kernel = np.ones(k)/k
    f = np.concatenate([f[:k//2], f, f[-k//2:]])
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[k//2:-k//2]
    return smooth_f


def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    h = x.shape[0]
    w = x.shape[1]
    nc = x.shape[-1]
    px = np.zeros((h+2*pad, w+2*pad, 3))
    color = colors.to_rgb(color_name)
    px[:,:,0] = color[0]
    px[:,:,1] = color[1]
    px[:,:,2] = color[2]
    if nc == 1:
        for c in range(3):
            px[pad:h+pad, pad:w+pad, c] = x
    else:
        px[pad:h+pad, pad:w+pad, :] = x
    return px

#
