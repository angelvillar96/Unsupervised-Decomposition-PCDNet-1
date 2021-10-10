"""
Implementation of model for template-based frame decomposition.
Frames are decomposed on templates formed by a single transformed version of
an object prototype.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_
from lib.metrics import fill_mask, fix_borders
import lib.prototypes as protolib
import models.transformers as transformers
from models import PCCell, SoftClamp


class DecompModel(nn.Module):
    """
    Model for template-based image decomposition with
    Phase Correlation Transformer networks

    Args:
    -----
    num_protos: integer
        number of object prototypes to use. Additionally we incldue void
    num_objects: integer
        maximum number of objects in a frame
    mode: string
        mode used for intializing the object prototypes
    proto_size: integer/tuple
        size of the object prototypes. So far, they must be the same size as images
    colorTf: bool
        If True, the color Module is applied after the PC-Cell
    background: bool
        If True, a background is learned and used for image composition
    randomness: bool
        If True, uniform noise is added to the prototypes to break symmetry
    randomness_prob: float [0,1]
        probability of adding noise to the templates is (1-this)
    """

    def __init__(self, num_protos, num_objects, mode="blobs", proto_size=64,
                 randomness=False, randomness_prob=0.2, background_size=64,
                 colorTf=False, background=False, **kwargs):
        """ Model Initializer """
        assert mode in protolib.PROTO_VALUES
        super().__init__()

        if(isinstance(proto_size, int)):
            proto_size = (proto_size, proto_size)

        self.num_protos = num_protos
        self.num_objects = num_objects
        self.max_objects = kwargs.get("max_objects", num_objects)
        self.mode = mode
        self.proto_size = proto_size
        self.background_size = background_size
        self.use_bkg = background
        self.randomness = randomness
        self.randomness_prob = kwargs.get("randomness_prob", 0.2)
        self.random_iters = kwargs.get("randomness_iters", -1)
        self.channels = kwargs.get("channels", 1)
        self.template_reg = kwargs.get("template_reg", 0)
        self.use_empty = kwargs.get("template_reg", False)

        # defining main learned parameters
        self.prototypes = protolib.init_prototypes(
                mode=mode, num_protos=num_protos, requires_grad=True,
                proto_size=proto_size, channels=self.channels
            )
        self.background = protolib.init_background(
                proto_size=self.background_size, channels=self.channels, requires_grad=self.use_bkg
            )
        self.masks = nn.Parameter(
                torch.ones(num_protos, 1, *proto_size) * 0.5
            ).requires_grad_()

        # custom modules and neural networks
        self.softclamp = SoftClamp(alpha=0.01)
        self.pc_cell = PCCell(L=self.max_objects, **kwargs)
        if(colorTf):
            self.color_transformer = transformers.ColorTransformer(
                    channels=self.channels
                )
        else:
            self.color_transformer = None

        return

    def init_background(self, db):
        """ Initializing background with an average of images """
        n_imgs = np.minimum(30000, len(db))
        print(f"Initializing background with mean of {n_imgs} images...")
        self.background = protolib.preload_background(
                background=self.background,
                dataset=db,
                n_imgs=n_imgs
            )
        return

    def _greedy_selection(self, x, templates, masks):
        """
        Greedy algorithm for selection of prototypes. Selecting for each object the
        template that minimizes the reconstruction error

        Args:
        -----
        x: torch Tensor
            image/frame with some objects + background. Shape is (B,C,H,W)
        templates: torch Tensor
            Template candidates to be placed in the image. Correspond to the
            transformed object prototypes. Shape is (B,P,C,H,W), with P being
            num_protos * num_objects
        masks: torch Tensor

        Returns:
        --------
        objects: torch Tensor
            Selected templates in decreasing order of importance.
            Shape is (N-Obj, C, H, W)
        """

        device = x.device
        B = x.shape[0]
        C = self.channels
        L = self.num_objects
        T = templates.shape[1]
        spatial_dims = templates.shape[-2:]

        # adding empty template
        if(self.use_empty):
            empty_template = torch.zeros(B, 1, C, x.shape[-2], x.shape[-1]).to(device)
            empty_mask = torch.zeros(B, 1, 1, x.shape[-2], x.shape[-1]).to(device)
        else:  # to reuse code, we add a useless template and never select it
            empty_template = 5 * torch.ones(B, 1, C, x.shape[-2], x.shape[-1]).to(device)
            empty_mask = torch.ones(B, 1, 1, x.shape[-2], x.shape[-1]).to(device)
        templates = torch.cat([empty_template, templates], dim=1)
        masks = torch.cat([empty_mask, masks], dim=1)

        # auxiliar variables
        template_ids = -1 * torch.ones(B, L).to(device)
        used_ids = torch.zeros(B, templates.shape[1])
        aux_objects = torch.zeros(B, L, C, x.shape[-2], x.shape[-1]).to(device)
        aux_masks = torch.zeros(B, L, 1, x.shape[-2], x.shape[-1]).to(device)
        orig_templates = templates.clone()
        orig_masks = masks.clone()

        with torch.no_grad():
            x_rep = x.unsqueeze(1).repeat(1, T+1, 1, 1, 1)
            temps = templates.view(B*(T+1), 1, C, *spatial_dims)
            masks_temps = masks.view(B*(T+1), 1, 1, *spatial_dims)

            for l in range(L):
                aux_objects_ = aux_objects[:, :l].unsqueeze(1).repeat(1, T+1, 1, 1, 1, 1)
                aux_objects_ = aux_objects_.view(B*(T+1), l, C, *spatial_dims)

                aux_masks_ = aux_masks[:, :l].unsqueeze(1).repeat(1, T+1, 1, 1, 1, 1)
                aux_masks_ = aux_masks_.view(B*(T+1), l, 1, *spatial_dims)

                # computing candidate reconstructions
                cur_recons = self.compose_templates(
                            objects=torch.cat([aux_objects_, temps], dim=1),
                            masks=torch.cat([aux_masks_, masks_temps], dim=1),
                            background=self.background,
                            sum_composition=False
                        )
                cur_recons = cur_recons.view(B, T+1, C, *spatial_dims)

                # computing current reconstruction errors
                errors = (x_rep - cur_recons).pow(2).mean(dim=(2, 3, 4))
                # to avoid repiting the same template twice
                errors[used_ids > 0] = 1e8

                cur_template_id = torch.argmin(errors, dim=1)
                template_ids[:, l] = cur_template_id

                for b, id in enumerate(cur_template_id):
                    used_ids[b, id] = 1 if id != 0 else 0
                    aux_objects[b, l] = templates[b, id]
                    aux_masks[b, l] = masks[b, id]

        template_ids = template_ids.long()
        objects = torch.stack([orig_templates[b, ids] for b, ids in enumerate(template_ids)])
        masks = torch.stack([orig_masks[b, ids] for b, ids in enumerate(template_ids)])
        del orig_templates
        del orig_masks

        return objects, masks, template_ids

    def compose_templates(self, objects, masks, background=None, sum_composition=False):
        """
        Reconstructing the scene by overlaping the objects in a greedy manner
        as in the Dead Leaves model
        """

        B, N = objects.shape[0], objects.shape[1]
        background = self.background if background is None else background

        # overlapping objects in inverse depth order
        if(N > 0 and sum_composition is False):
            cur_recons = background.repeat(B, 1, 1, 1)
            # recons_new = obj * alpha + recons_old * (1 - alhpha)
            for i in range(N):
                cur_obj = objects[:, -1-i]
                cur_mask = masks[:, -1-i]
                cur_recons = cur_obj * cur_mask + cur_recons * (1 - cur_mask)
            recons = cur_recons
        else:  # just summing. Works when no overlap
            recons = (objects * masks).sum(1) + background * (1 - masks.sum(1))

        return recons

    def compose_masks(self):
        """
        Overlapping the masks in a dead leaves model fashion
        """
        masks = self.final_masks
        B, N = masks.shape[0], masks.shape[1]
        H, W = masks.shape[-2], masks.shape[-1]
        cur_recons = torch.zeros(B, 1, H, W).to(masks.device)
        for i in range(N):
            cur_mask = masks[:, -1-i]
            cur_recons = cur_mask + cur_recons * (1 - cur_mask)
        recons = cur_recons
        return recons

    def pad_prototypes(self, img, protos):
        """
        Padding prototypes to match image size
        """
        if(img.shape[-2:] != protos.shape[-2:]):
            pad_row = (img.shape[-2] - protos.shape[-2]) / 2
            pad_col = (img.shape[-1] - protos.shape[-1]) / 2
            padding = (
                    int(np.ceil(pad_col)), int(np.floor(pad_col)),
                    int(np.ceil(pad_row)), int(np.floor(pad_row))
                )
            pad_protos = F.pad(protos, padding)
        else:
            pad_protos = protos
        return pad_protos

    def forward(self, x):
        """
        Forward pass through PCDNet
            1. Padding prototypes to right size, and injecting noise if necessary
            2. Using PC-Cell for estimating candidate positions and translating
            3. Applying texture with color transformer
            4. Reconstructing image by using the greedy algorithm
        """

        B, C, H, W = x.shape
        L = self.max_objects
        P = self.num_protos

        if self.training:
            protos, masks = self.prototypes.clone(), self.masks.clone()
        else:
            masks = self.process_masks(self.masks)
            protos = self.prototypes * masks

        # adding noise to protos to avoid overselection due to greedy selection function
        for i in range(len(protos)):
            if(self.randomness and torch.rand(1) > self.randomness_prob and self.training):
                noise = (torch.rand(*protos[i].shape, device=protos[i].device) - 0.5)
            else:
                noise = torch.zeros(*protos[i].shape, device=protos[i].device)
            protos[i] = protos[i] + noise

        # adding some noise to masks
        if self.training:
            noise = (torch.rand(*masks.shape, device=masks.device) - 0.5)
            masks = masks + noise

        # soft clamp to range [0,1]
        masks = self.softclamp(masks)
        protos = self.softclamp(protos)

        # padding prototypes to match the correct size
        protos = self.pad_prototypes(img=x, protos=protos)
        masks = self.pad_prototypes(img=x, protos=masks)

        # estimating the candidate templates and shifting masks with the PC-Cell
        templates, peaks = self.pc_cell(x, protos)
        masks = self.pc_cell.translate_masks(masks)
        masks = masks.view(B, L * P, 1, H, W)
        templates = templates.view(B, L * P, 1, H, W)

        # applying color transformer to candidate templates
        if(self.color_transformer):
            templates = self.color_transformer(
                    img=x, templates=templates, masks=masks
                )
        self.templates = templates

        # greedy selection
        objects, masks, object_ids = self._greedy_selection(
                x=x, templates=templates, masks=masks
            )
        self.final_masks = masks

        # reconstructing by sum/overlap of the selected objects/masks
        reconstruction = self.compose_templates(
                objects=objects.clone(),
                masks=masks.clone(),
                background=self.background,
                sum_composition=False
            )

        return reconstruction, (objects, object_ids, protos)

    def randomness_shutdown(self, iter_, verbose=True):
        """
        Shutting down the prototype noise after the setup iterations
        """
        if(iter_ >= self.random_iters and self.random_iters != -1 and self.randomness is True):
            if(verbose):
                print_(f"Reached iteration {iter_}: Turning off prototype noise")
            self.randomness = False

        return

    def process_masks(self, masks, thr=0.8, fill_thr=0.2):
        """
        Processing prototype masks in order to fill gaps and fully binarize
        """
        thr_masks = masks.clone()
        # binarizing given threhold
        thr_masks[thr_masks < thr] = 0
        # thr_masks[thr_masks >= thr] = 1
        # filling gaps
        # thr_masks = fix_borders(thr_masks)
        return thr_masks

    def __repr__(self):
        """ Overriding the string representation of the object """
        list_mod = super().__repr__()[:-2]

        transform_string1 = f"DecompModel(\n  (prototypes): Prototypes(n_protos={self.num_protos}, mode={self.mode})"
        transform_string2 = f"  (background): Background(size=({self.proto_size}), learned={self.use_bkg})"
        transform_string = transform_string1 + "\n" + transform_string2
        list_mod = list_mod.replace("DecompModel(", transform_string)
        list_mod += "\n)"
        return list_mod


#
