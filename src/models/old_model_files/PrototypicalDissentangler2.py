"""
Implementation of model for template-based frame decomposition.
Frames are decomposed on templates formed by a single transformed version of
an object prototype.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fftlib

import lib.freq_ops as freq_ops
import lib.metrics as metrics
import lib.prototypes as protolib
import lib.transformations as tflib
import lib.utils as utils
import models.DifferentiableBlocks as DiffBlocks
import models.transformers as transformers
from models import PCCell


class DecompModel2(nn.Module):
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

    def __init__(self, num_protos, num_objects, mode="blobs", proto_size=64, background_size=64,
                 colorTf=False, background=False, randomness=False, randomness_prob=0.2,
                 use_empty=True, **kwargs):
        """ Model Initializer """
        assert mode in protolib.PROTO_VALUES
        super().__init__()

        if(isinstance(proto_size,int)):
            proto_size = (proto_size,proto_size)

        self.num_protos = num_protos
        self.num_objects = num_objects
        self.max_objects = num_objects if "max_obj" not in kwargs else kwargs["max_obj"]
        self.mode = mode
        self.proto_size = proto_size
        self.background_size = background_size
        self.use_bkg = background
        self.randomness = randomness
        self.randomness_prob = randomness_prob
        self.channels = 1 if "channels" not in kwargs else kwargs["channels"]
        self.template_reg = 0 if "template_reg" not in kwargs else kwargs["template_reg"]
        self.template_error = None
        self.use_empty = use_empty

        # defining building blocks
        self.prototypes = protolib.init_prototypes(
                mode=mode, num_protos=num_protos, requires_grad=True,
                proto_size=proto_size, channels=self.channels
            )
        self.background = protolib.init_background(
                proto_size=self.background_size, channels=self.channels, requires_grad=self.use_bkg
            )
        self.pc_cells = nn.ModuleList([PCCell(L=self.max_objects, **kwargs) for _ in range(num_protos)])
        self.transformer = transformers.Translation(pcCell=False)
        if(colorTf):
            self.color_transformer = transformers.ColorTransformer(channels=self.channels)
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

    def _greedy_selection(self, x, templates):
        """
        Greedy algorithm for selection of prototypes. Selecting for each object the
        template that minimizes the reconstruction error

        Args:
        -----
        x: torch Tenosr
            image/frame with some objects + background. Shape is (B,C,H,W)
        templates: torch Tensor
            Template candidates to be placed in the image. Correspond to the
            transformed object prototypes. Shape is (B,P,C,H,W), with P being
            num_protos * num_objects

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
        spatial_dims = templates.shape[-3:]

        # adding empty template
        if(self.use_empty):
            empty_template = torch.zeros(B, 1, C, x.shape[-2], x.shape[-1]).to(device)
        else:
            empty_template = 5*torch.ones(B, 1, C, x.shape[-2], x.shape[-1]).to(device)

        # templates = torch.cat([templates, empty_template], dim=1)
        templates = torch.cat([empty_template, templates], dim=1)

        # auxiliar variables
        template_ids = -1 * torch.ones(B,L).to(device)
        aux_objects = torch.zeros(B, L, C, x.shape[-2], x.shape[-1]).to(device)
        used_ids = torch.zeros(B, templates.shape[1])
        orig_templates = templates.clone()

        x_rep = x.unsqueeze(1).repeat(1, T+1, 1, 1, 1)
        temps = templates.view(B*(T+1), 1, *spatial_dims)

        with torch.no_grad():
            for l in range(L):
                aux_objects_ = aux_objects[:,:l].unsqueeze(1).repeat(1, T+1, 1, 1, 1, 1)
                aux_objects_ = aux_objects_.view(B*(T+1), l, *spatial_dims)

                # computing candidate reconstructions
                cur_recons = self.compose_templates(
                            objects=torch.cat([aux_objects_, temps], dim=1),
                            background=self.background,
                            sum_composition=False
                        )
                cur_recons = cur_recons.view(B, T+1, *spatial_dims)

                # computing current reconstruction errors
                errors = (x_rep - cur_recons).pow(2).mean(dim=(2,3,4))
                # errors[:, 1:] = errors[:, 1:] + self.template_reg  # small error to avoid overlap
                errors[:, 0] = errors[:, 0] #+ self.template_reg  # small error to avoid always selecting zeros
                # selecting template that minimizes reconstruction error
                errors[used_ids > 0] = 1e8  # to avoid repiting the same template twice

                cur_template_id = torch.argmin(errors, dim=1)
                template_ids[:,l] = cur_template_id

                for b, id in enumerate(cur_template_id):
                    used_ids[b, id] = 1 if id != 0 else 0
                    aux_objects[b,l] = templates[b,id]

        template_ids = template_ids.long()
        objects = [orig_templates[b,ids] for b, ids in enumerate(template_ids)]
        objects = torch.stack(objects)

        del orig_templates

        return objects, template_ids



    def compose_templates(self, objects, background=None, sum_composition=False):
        """
        Reconstructing the scene by overlaping the objects in a greedy manner
        """

        B, N = objects.shape[0], objects.shape[1]
        background = self.background if background is None else background
        # overlapping objects in inverse depth order
        if(N > 0 and sum_composition == False):
            cur_recons = background.repeat(B,1,1,1)
            obj_masks = metrics.binarize_masks(objects, thr=0.1)
            for i in range(N):
                # getting binary mask for object
                cur_obj = objects[:, -1-i]
                cur_mask = obj_masks[:,-1-i]
                if self.channels == 3:
                    cur_mask = cur_mask.unsqueeze(1)
                    ids = cur_mask > 0
                    ids = torch.cat([ids, ids, ids], dim=1)
                else:
                    ids = cur_mask > 0
                cur_recons[ids] = cur_obj[ids]
            recons = cur_recons
        # simple approach: just summing objects. Works worse, but its faster
        else:
            recons = objects.sum(1) + background
            recons = recons.clamp(0,1)

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
        """ Forward pass """

        device = x.device
        B,C,H,W = x.shape
        # L = self.num_objects
        L = self.max_objects
        P = self.num_protos

        # padding prototypes if necessary
        protos = self.pad_prototypes(img=x, protos=self.prototypes)
        self.pad_protos = protos

        # estimating the candidate template positions with the PC-Cell
        peaks = []
        for i, proto in enumerate(protos):
            # injecting some randomenss during training into the prototypes to break symmetry
            if(self.randomness and torch.rand(1) > self.randomness_prob and self.training):
                # proto = proto + torch.rand(*proto.shape, device=proto.device) * 0.2
                proto = proto + torch.rand(*proto.shape, device=proto.device) - 0.5
            cur_peaks, (_, _) = self.pc_cells[i](x, proto.unsqueeze(0))
            peaks.append(cur_peaks)
        peaks = torch.stack(peaks, dim=1)
        peaks = peaks.view(B * L * P, 2)
        self.peaks = peaks

        # transforming prototypes into templates
        translation_tf = self.transformer.transform_from_params( -peaks[:,1], -peaks[:,0] )
        self.translation_tf = translation_tf
        proto_tf = [p.repeat(L,1,1,1) for p in protos]
        proto_tf = torch.cat(proto_tf, dim=0).repeat(B,1,1,1)
        templates = tflib.apply_transform(imgs=proto_tf, transform=translation_tf)

        # applying color transformer
        templates = templates.view(B, L * P, 1, H, W)
        self.grey_templates = templates
        if(self.color_transformer):
            templates = self.color_transformer(img=x, templates=templates)

        # greedy selection
        self.templates = templates
        objects, object_ids = self._greedy_selection(x, templates)
        self.objects, self.object_ids = objects, object_ids
        self.compute_template_penalization(object_ids, B)

        # reconstructing by sum/overlap
        reconstruction = self.compose_templates(
                objects=objects.clone(),
                background=self.background,
                sum_composition=False
            )#.clamp(0,1)

        return reconstruction, (objects, object_ids, self.prototypes)


    @torch.no_grad()
    def compute_template_penalization(self, object_ids, b_size=1):
        """
        Computing regularization term that pernalizes with some small error
        the use of non-empty templates. This prevents correcting minor erros
        by overlapping different templates for just one object
        """

        num_empty = len(torch.where(object_ids.flatten() == 0)[0])
        num_non_empty = object_ids.numel() - num_empty
        self.template_error = num_non_empty / b_size
        return

    def get_template_error(self):
        """ Fetching the precompute template error for regularization """
        return self.template_error


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
