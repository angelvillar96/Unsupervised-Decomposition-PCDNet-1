"""
Training a Dissentangle model to learn a set of object prototypes to decompose
the scene into
"""

import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.arguments import get_directory_argument
from lib.config import load_exp_config_file
from lib.logger import Logger, print_, log_function, for_all_methods, log_info
from lib.losses import get_loss
import lib.setup_model as setup_model
import lib.utils as utils
from lib.visualizations import visualize_prototypes, visualize_recons
import data


@for_all_methods(log_function)
class Trainer:
    """
    Class for training a PredFormer model
    """

    def __init__(self, exp_path, checkpoint=None):
        """
        Initializing the trainer object
        """

        # utils.set_random_seed()
        self.exp_path = exp_path
        self.exp_params = load_exp_config_file(exp_path)

        if checkpoint is None:
            self.checkpoint_path = None
        else:
            self.checkpoint_path = os.path.join(exp_path, "models", "PCDNet_models", checkpoint)

        self.plots_path = os.path.join(self.exp_path, "plots", "dissentangle_valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models", "PCDNet_models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", "dissentangle_logs",
                                   f"Dissentangle_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.loss_types = ["total", "reconstruction", "template_regularization", "proto_l1"]
        self.training_losses = {}
        self.validation_losses = {}
        self.loss_iters = []
        for loss in self.loss_types:
            self.training_losses[loss] = []
            self.validation_losses[loss] = []
        self.writer = SummaryWriter(tboard_logs)

        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """

        # loading dataset and data loaders
        # utils.set_random_seed()
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        train_set = data.load_data(exp_params=self.exp_params, split="train")
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = data.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Examples in validation set: {len(valid_set)}")
        shuffle_train = False
        self.train_loader = data.build_data_loader(dataset=train_set,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle_train)
        self.valid_loader = data.build_data_loader(dataset=valid_set,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle_eval)
        return

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """

        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = setup_model.setup_model(
                model_params=self.exp_params["model"],
                dataset_name=self.exp_params["dataset"]["dataset_name"]
            )
        model_params = self.exp_params["model"]["PrototypeMatcher"]

        # initializing the background prototype
        if(model_params["init_background"] == True):
            model.init_background(db=self.train_loader.dataset)
            print(model.background.shape)

        self.model = model.to(self.device)
        utils.log_architecture(model=model,
                               exp_path=self.exp_path,
                               fname="architecture.txt")

        # loading optimizer, scheduler
        optimizer, scheduler = setup_model.setup_optimizer(exp_params=self.exp_params, model=model)

        # loading pretrained parameters if needed
        init_epoch = 0
        if self.checkpoint_path is not None:
            print_(f"Loading pretrained parameters from {self.checkpoint_path}...")
            model, optimizer, scheduler, init_epoch, meta = setup_model.load_checkpoint(
                    checkpoint_path=self.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
            self.training_losses = meta["train_loss"] if meta is not None else []
            self.validation_losses = meta["valid_loss"] if meta is not None else []
            self.loss_iters = meta["loss_iters"] if meta is not None else []

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.init_epoch = init_epoch

        # setting up loss function and multiplyiers
        lambda_recons = self.exp_params["loss"]["lambda_recons"]
        lambda_reg = self.exp_params["loss"]["lambda_reg"] if "lambda_reg" in self.exp_params["loss"] else 0
        lambda_l1 = self.exp_params["loss"]["lambda_l1"] if "lambda_l1" in self.exp_params["loss"] else 0
        self.loss_recons = get_loss(loss_type=self.exp_params["loss"]["loss_recons"])
        self.temp_reg = lambda : self.model.get_template_error() if self.exp_params["loss"]["template_reg"] else 0.
        self.l1_reg = lambda : self.model.prototypes.abs().sum() / len(self.model.prototypes)
        self.total_loss = lambda l1, l2, l3: lambda_recons * l1 + lambda_reg * l2 + lambda_l1 * l3

        return

    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the number of
        epoch specified in the exp_params file.
        """

        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        for epoch in range(self.init_epoch, num_epochs):
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing train vs eval losses
            for loss in self.loss_types:
                self.writer.add_scalars(f'loss/COMB_{loss}_loss', {
                    'train_loss': self.training_losses[loss][-1],
                    'eval_loss': self.validation_losses[loss][-1],
                }, epoch+1)

            # updating learning rate scheduler if loss increases or plateaus
            setup_model.update_scheduler(scheduler=self.scheduler,
                                         exp_params=self.exp_params,
                                         control_metric=self.validation_losses["total"][-1])

            # saving model checkpoint if reached saving frequency
            if(epoch % save_frequency == 0):
                print_(f"Saving model checkpoint")
                meta = {
                    "train_loss":self.training_losses,
                    "valid_loss":self.validation_losses,
                    "loss_iters": self.loss_iters
                }
                setup_model.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                            scheduler=self.scheduler, epoch=epoch, meta=meta,
                                            exp_path=self.exp_path, savedir="models/PCDNet_models")

        print_(f"Finished training procedure")
        print_(f"Saving final checkpoint")
        meta = {
            "train_loss": self.training_losses,
            "valid_loss": self.validation_losses,
            "loss_iters": self.loss_iters
        }

        setup_model.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, epoch=epoch,
                                    exp_path=self.exp_path, savedir="models/PCDNet_models",
                                    meta=meta, finished=True)
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        # utils.set_random_seed()
        epoch_losses = {}
        for type in self.loss_types:
            epoch_losses[type] = []
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, (frames) in progress_bar:
            iter_ = len(self.train_loader) * epoch + i
            self.model.randomness_shutdown(iter_=iter_)

            F, B, C, H, W = frames.shape if(len(frames.shape) == 5) else 1, *frames.shape
            frames = frames.to(self.device).view(F*B, C, H, W).float()

            reconstruction, (_, _, protos) = self.model(frames)

            # computing loss
            loss_recons = self.loss_recons(reconstruction, frames)
            template_reg = self.temp_reg()
            l1_reg = self.l1_reg()
            loss = self.total_loss(loss_recons, template_reg, l1_reg)
            self.loss_iters.append(loss.item())
            epoch_losses["reconstruction"].append(loss_recons.item())
            epoch_losses["template_regularization"].append(template_reg)
            epoch_losses["proto_l1"].append(l1_reg.item())
            epoch_losses["total"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # writinf Tensorboard
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                for type in self.loss_types:
                    self.writer.add_scalar(f'train_loss/{type}_loss',
                                           epoch_losses[type][-1],
                                           global_step=iter_)
                    log_data = f"""Log data train iteration {iter_}:
                                   loss={round(np.mean(epoch_losses[type]),5 )};"""
                    log_info(message=log_data)

            # visualizing the learned prototypes and a few reconstructions every epoch / few iterations
            if(i == 0 or iter_ % self.exp_params["training"]["log_frequency"] == 0):
                try:
                    # prototypes image
                    savepath = os.path.join(self.plots_path, f"Prototypes_Epoch_{epoch+1}_iter_{i+1}.png")
                    fig, ax = visualize_prototypes(protos=self.model.prototypes.clamp(0,1), savepath=savepath)
                    self.writer.add_figure(tag=f"Prototypes Epoch {epoch+1} iter {i+1}", figure=fig)
                    # reconstructions image
                    savepath = os.path.join(self.plots_path, f"Recons_Epoch_{epoch+1}_iter_{i+1}.png")
                    fig, ax = visualize_recons(
                            recons=reconstruction.detach().cpu().clamp(0, 1),
                            frames=frames, savepath=savepath
                        )
                    self.writer.add_figure(tag=f"Reconstructions Epoch {epoch+1} Iter {i+1}", figure=fig)
                    # masks image
                    savepath = os.path.join(self.plots_path, f"Masks_Epoch_{epoch+1}_iter_{i+1}.png")
                    fig, ax = visualize_prototypes(
                            protos=self.model.masks.detach().cpu().clamp(0, 1),
                            savepath=savepath
                        )
                    self.writer.add_figure(tag=f"Masks Epoch {epoch+1} Iter {i+1}", figure=fig)
                    # masked protos image
                    savepath = os.path.join(self.plots_path, f"MaskedProtos_Epoch_{epoch+1}_iter_{i+1}.png")
                    fig, ax = visualize_prototypes(
                            protos=(self.model.prototypes * self.model.masks).clamp(0, 1),
                            savepath=savepath
                        )
                    self.writer.add_figure(tag=f"Masked Prototypes Epoch {epoch+1} Iter {i+1}", figure=fig)
                    # processed masks image
                    savepath = os.path.join(self.plots_path, f"ProcessedMasks_Epoch_{epoch+1}_iter_{i+1}.png")
                    thr_masks = self.model.masks.clone()
                    thr_masks[thr_masks < 0.7] = 0
                    fig, ax = visualize_prototypes(
                            protos=thr_masks.detach().cpu().clamp(0, 1),
                            savepath=savepath
                        )
                    self.writer.add_figure(tag=f"Processed Masks Epoch {epoch+1} Iter {i+1}", figure=fig)
                except:
                    pass

                setup_model.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                            scheduler=self.scheduler, epoch=epoch, meta=None,
                                            exp_path=self.exp_path, savedir="models/PCDNet_models",
                                            savename=f"checkpoint_epoch_{epoch}_iter_{i}.pth")

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        for type in self.loss_types:
            train_loss = np.mean(epoch_losses[type])
            self.training_losses[type].append(train_loss)
            print_(f"    Train {type} Loss: {train_loss}")

        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """

        epoch_losses = {}
        for type in self.loss_types:
            epoch_losses[type] = []
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, (frames) in progress_bar:
            # not validating for more than 50 batches
            if(i >= 30):
                break

            F, B, C, H, W = frames.shape if(len(frames.shape) == 5) else 1, *frames.shape
            frames = frames.to(self.device).view(F*B, C, H, W).float()

            reconstruction, (objects, object_ids, protos) = self.model(frames)

            # computing loss
            loss_recons = self.loss_recons(reconstruction, frames)
            template_reg = self.temp_reg()
            l1_reg = self.l1_reg()
            loss = self.total_loss(loss_recons, template_reg, l1_reg)
            self.loss_iters.append(loss.item())
            epoch_losses["reconstruction"].append(loss_recons.item())
            epoch_losses["template_regularization"].append(template_reg)
            epoch_losses["proto_l1"].append(l1_reg.item())
            epoch_losses["total"].append(loss.item())

            # # visualizing the learned prototypes and a few reconstructions every epoch
            # if(i == 0 or i % self.exp_params["training"]["log_frequency"] == 0):
            #     savepath = os.path.join(self.plots_path, f"Prototypes_Epoch_{epoch+1}_iter_{i+1}.png")
            #     fig, ax = visualize_prototypes(protos=protos.clamp(0,1), savepath=savepath)
            #     self.writer.add_figure(tag=f"Prototypes Epoch {epoch+1} iter {i+1}", figure=fig)
            #     savepath = os.path.join(self.plots_path, f"Recons_Epoch_{epoch+1}_iter_{i+1}.png")
            #     fig, ax = visualize_recons(recons=reconstruction, frames=frames, savepath=savepath)
            #     self.writer.add_figure(tag=f"Reconstructions Epoch {epoch+1} Iter {i+1}", figure=fig)

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}.")

        # logging loss values
        for loss_type in self.loss_types:
            print(loss_type)
            valid_loss = np.mean(epoch_losses[loss_type])
            self.validation_losses[loss_type].append(valid_loss)
            self.writer.add_scalar(f'valid_loss/{loss_type}_loss',
                                   valid_loss,
                                   global_step=epoch+1)
            print_(f"    Valid {loss_type} Loss: {valid_loss}")

        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, checkpoint = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting DecompNet training procedure", message_type="new_exp")

    print("Initializing Trainer...")
    trainer = Trainer(exp_path=exp_path, checkpoint=checkpoint)
    print("Loading dataset...")
    trainer.load_data()
    print("Setting up model and optimizer")
    trainer.setup_model()
    print("Starting to train")
    trainer.training_loop()


#
