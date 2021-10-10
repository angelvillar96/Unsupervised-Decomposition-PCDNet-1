"""
Dataset for working with the attari games
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

from CONFIG import CONFIG

PATH = CONFIG["paths"]["data_path"]


GAMES = {
    "BeamRider": "BeamRiderNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "Enduro": "EnduroNoFrameskip-v4",
    "Pong": "PongNoFrameskip-v4",
    "QBert": "QbertNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4"
}
LIST_GAMES = list(GAMES.keys())


class AttariDataset(Dataset):
    """
    Dataset object for the Atari Dataset

    Args:
    -----
    path: string
        path to the directory containing the data
    mode: string
        Dataset split to use ["train", "val", "test", "valid", "eval"]
    game: string
        Atari game to load
        [BeamRider, Breakout, Enduro, Pong, QBert, Seaquest, SpaceInvaders]
    rgb: boolean
        If True, images are converted to grayscale
    """

    def __init__(self, path=None, mode="train", game="SpaceInvaders", rgb=False):
        """ Data loader initializer """
        assert mode in ["train", "val", "test", "valid", "eval"]
        assert game in LIST_GAMES
        mode = "test" if mode in ["test", "eval"] else mode
        mode = "val" if mode in ["val", "valid"] else mode
        path = path if path is not None else PATH

        data_path = os.path.join(path, "AAD", "clean", GAMES[game])
        assert os.path.exists(data_path)
        self.rgb = rgb

        all_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if "npy" in f])
        N = len(all_files)
        IDS = {
            "train": np.arange(0, int(round(0.7*N))),
            "val": np.arange(int(round(0.7*N)), int(round(0.85*N))),
            "test": np.arange(int(round(0.85*N)), N)
        }

        data_files = []
        for i in IDS[mode]:
            try:
                cur_data = np.load(all_files[i])
                data_files.append(cur_data)
            except:
                continue

        print(f"Loaded {len(data_files)} data files...")
        self.data_files = np.concatenate(data_files)

        return

    def __getitem__(self, index):
        """ Sampling image """
        x = self.data_files[index]
        x = x / 255
        x = torch.Tensor(x).permute(2,0,1)
        x = x[:, 25:]
        if(not self.rgb):
            r, g, b = x[0, :, :], x[1, :, :], x[2, :, :]
            x = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(0)
        return x

    def __len__(self):
        return len(self.data_files)

    @staticmethod
    def get_games():
        """ Getting the possible games """
        return LIST_GAMES

#
