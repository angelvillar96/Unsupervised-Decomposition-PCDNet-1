"""
Aux file to load the data from the cars dataset.
See the notebook Test_Real.ipynb for further details
"""

import os
import numpy as np
import torch
from CONFIG import CONFIG


FNAME = {
    "top": os.path.join(CONFIG["paths"]["data_path"], "cars", "cars.npy"),
    "front": os.path.join(CONFIG["paths"]["data_path"], "cars", "cars_front.npy"),
    "intersect": os.path.join(CONFIG["paths"]["data_path"], "cars", "cars_intersect.npy"),
    "side": os.path.join(CONFIG["paths"]["data_path"], "cars", "cars_side.npy")
}


class CarsDataset:
    """
    Dataset for loading the frames from the cars-db
    """

    def __init__(self, split="train", rgb=True, video="top", **kwargs):
        """ """
        assert video in ["top", "front", "intersect", "side"]
        frames = np.load(FNAME[video])
        frames = torch.Tensor(frames)
        self.frames = frames.unsqueeze(1)
        self.rgb = rgb
        if(split != "train"):
            self.frames = self.frames[:5]
        return

    def __len__(self):
        """ """
        return len(self.frames)

    def __getitem__(self, i):
        """ """
        frame = self.frames[i][0].permute(2,0,1)
        frame = frame / 255
        frame = torch.stack([frame[2], frame[1], frame[0]])
        if(not self.rgb):
            r, g, b = frame[0, :, :], frame[1, :, :], frame[2, :, :]
            frame = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(0)
        return frame

    @staticmethod
    def get_video():
        """ Getting the possible videos"""
        return FNAME
