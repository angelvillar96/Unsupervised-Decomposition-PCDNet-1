"""
Tetrominoes dataset from:
    Rishabh et al. "Multi-Object Datasets", 2019

1.000.000 images total  but...
 - 60.000 train
 - 320 test

17 unique protypes (different orientations = different prototypes)
"""

import os
import pickle
import torch


class Tetrominoes:
    """
    Loading Tetrominoes data from the Pickle file
    """

    def __init__(self, path, split="train", rgb=True, get_meta=False):
        """ Dataset initializer """
        # checking split is correct and data exists
        assert split in ["train", "val", "valid", "eval", "test"]
        split = "valid" if split in ["val", "valid"] else split
        split = "test" if split in ["test", "eval"] else split

        data_path = os.path.join(path, "Tetrominoes", f"{split}_data.pkl")
        assert os.path.exists(data_path)

        self.data_path = data_path
        self.split = split
        self.rgb = rgb
        self.get_meta = get_meta

        # loading data
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
        self.data = data

        return

    def __len__(self):
        """ Getting number of examples """
        return len(self.data)

    def __getitem__(self, i, get_mask=False):
        """ Sampling element i from the dataset """

        example = self.data[i]
        img = example["image"]
        meta = example

        if(not self.rgb):
            img = self.rgb2gray(img)
            img = torch.Tensor(img).unsqueeze(0)
        else:
            img = torch.Tensor(img).permute(2,0,1)

        if(get_mask or self.get_meta):
            return img / 255, meta
        else:
            return img / 255

    def rgb2gray(self, rgb):
        """"""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
