"""
Script for generating some paper figures
"""

import os
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import torch

from lib.arguments import get_directory_argument
from lib.config import load_exp_config_file
from lib.logger import Logger, for_all_methods, log_function
import lib.metrics as metrics
import lib.setup_model as setup_model
import lib.utils as utils
import data

N_COLS = 8
SIZE = 3
COLORS = ["bisque", "aqua",  "red", "darkorange", "goldenrod", "forestgreen",
          "springgreen", "royalblue", "navy", "darkviolet", "plum", "magenta",
          "slategray", "yellow", "peachpuff", "red", "maroon", "silver",
          "aquamarine", "pink", "brown", "gold"]
THR = {
    "Tetrominoes": 0.8,
    "Cars": 0.4,
    "AtariSpace": 0.4
}


@for_all_methods(log_function)
class FigGenerator:
    """
    Class for evaluating a decomposition model
    """

    def __init__(self, exp_path, checkpoint):
        """
        Initializing the evaluator object
        """

        utils.set_random_seed()
        self.exp_path = exp_path
        self.exp_params = load_exp_config_file(exp_path)

        checkpoint_name = checkpoint.split(".")[0]
        self.plots_path = os.path.join(self.exp_path, "plots", f"Figs_{checkpoint_name}")
        utils.create_directory(self.plots_path)

        self.model_path = os.path.join(exp_path, "models", "PCDNet_models", checkpoint)

        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        self.dataset_name = self.exp_params["dataset"]["dataset_name"]
        self.test_set = data.load_data(exp_params=self.exp_params, split="test")
        return

    def setup_model(self):
        """
        Instanciating model and loading pretrained parameters
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = setup_model.setup_model(
                model_params=self.exp_params["model"],
                dataset_name=self.exp_params["dataset"]["dataset_name"]
            )
        model = setup_model.load_checkpoint(
                self.model_path, model=model, only_model=True
            )
        self.model = model.to(self.device).eval()

        self.mse = metrics.MSE()
        return

    @torch.no_grad()
    def generate_figs(self, i=0):
        """
        Generating nice figures for one image
        """
        global COLORS
        dif = self.model.num_objects - len(COLORS)
        if(dif > 0):
            extra_colors = [np.random.rand(3) for i in range(dif)]
            COLORS = COLORS + extra_colors

        # forward pass
        imgs = self.test_set[i]
        C, H, W = imgs.shape
        imgs = imgs.to(self.device).view(1, C, H, W).float()
        reconstruction, (objects, object_ids, protos) = self.model(imgs)
        num_objs = objects.shape[1]
        max_objs = self.model.max_objects
        n_cols = min(N_COLS, num_objs)
        n_rows = math.ceil(num_objs / n_cols)

        cur_path = os.path.join(self.plots_path, f"image_{i}")
        utils.create_directory(cur_path)

        ############################
        # generating image/recons/error
        ############################
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(3*SIZE, SIZE)
        ax[0].imshow(imgs[0].cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        ax[0].set_title("Original Img")
        ax[1].imshow(reconstruction[0].cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        ax[1].set_title("Reconstructed Img")
        err = (imgs[0].cpu() - reconstruction[0].cpu())
        im = ax[2].imshow(err.permute(1, 2, 0), vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Error")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(cur_path, "recons_img.png"))

        ############################
        # selected objects
        ############################
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_objs):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(objects[0, i].cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
            a.set_title(f"Object {i} = Template {object_ids[0,i].item()}")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(cur_path, "objects.png"))

        ############################
        # segmentation masks
        ############################
        # bin_mask = self.model.process_masks(self.model.final_masks, thr=THR[self.dataset_name])
        bin_mask = objects
        bin_mask[bin_mask > 0.1] = 1

        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_objs):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(bin_mask[0, i, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(cur_path, "segmentation_masks.png"))

        ############################
        # semantic segmentation
        ############################
        masks = []
        for i in range(num_objs):
            mask1 = bin_mask[0, i].repeat(3, 1, 1) if bin_mask.shape[2] == 1 else bin_mask[0, i]
            color = colors.to_rgb(COLORS[(object_ids[0, i]-1)//max_objs])
            color = torch.Tensor(color).to(mask1.device)
            mask1[0] = mask1[0] * color[0]
            mask1[1] = mask1[1] * color[1]
            mask1[2] = mask1[2] * color[2]
            masks.append(mask1)
        masks = torch.stack(masks)
        img = self.model.compose_templates(
                objects=masks.unsqueeze(0),
                masks=bin_mask,
                background=torch.zeros(3, H, W).cuda()
            )
        plt.figure(figsize=(SIZE, SIZE))
        plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
        plt.axis("off")
        plt.savefig(os.path.join(cur_path, "semantic_masks.png"))

        ############################
        # Instance Segmentation
        ############################
        masks = []
        for i in range(num_objs):
            mask1 = bin_mask[0, i].repeat(3, 1, 1) if bin_mask.shape[2] == 1 else bin_mask[0, i]
            color = colors.to_rgb(COLORS[i])
            color = torch.Tensor(color).to(mask1.device)
            mask1[0] = mask1[0] * color[0]
            mask1[1] = mask1[1] * color[1]
            mask1[2] = mask1[2] * color[2]
            masks.append(mask1)
        masks = torch.stack(masks)
        img = self.model.compose_templates(
                objects=masks.unsqueeze(0),
                masks=bin_mask,
                background=torch.zeros(3, H, W).cuda()
            )
        plt.figure(figsize=(SIZE, SIZE))
        plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
        plt.axis("off")
        plt.savefig(os.path.join(cur_path, "instance_masks.png"))

        return

    @torch.no_grad()
    def generate_global_figs(self):
        """
        Generating some global figures: prototypes, masks, combined imgs
        """
        protos = self.model.prototypes
        masks = self.model.masks
        proc_masks = self.model.process_masks(masks, thr=THR[self.dataset_name])
        proc_protos = protos * proc_masks

        num_protos = len(protos)
        n_cols = min(N_COLS, num_protos)
        n_rows = math.ceil(num_protos / n_cols)

        # Prototypes
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_protos):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(protos[i, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "prototypes.png"))

        # masks
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_protos):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(masks[i, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "masks.png"))

        # processed masks
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_protos):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(proc_masks[i, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "processed_masks.png"))

        # processed masks
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(n_cols*SIZE, n_rows*SIZE)
        for i in range(num_protos):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.imshow(proc_protos[i, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "processed_protos.png"))

        return


if __name__ == "__main__":
    utils.clear_cmd()
    utils.set_random_seed()
    exp_path, checkpoint = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting Figure generation procedure", message_type="new_exp")

    print("Initializing Fig Generator...")
    figGenerator = FigGenerator(exp_path=exp_path, checkpoint=checkpoint)
    print("Loading dataset...")
    figGenerator.load_data()
    print("Setting up model")
    figGenerator.setup_model()
    print("Generating some figures")
    ids = torch.randint(low=0, high=len(figGenerator.test_set), size=(5,))
    for i, id in enumerate(ids):
        print(f"  Generating figs {i+1}...")
        figGenerator.generate_figs(i=id)
    figGenerator.generate_global_figs()
