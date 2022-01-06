"""
Evaluation metrics for prediction and reconstruction

Several evaluation functions are taken from:
    https://github.com/ecker-lab/object-centric-representation-benchmark
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.cluster import adjusted_rand_score as ARI


class MSE(nn.Module):
    """
    Custom Mean Squared Error

    Args:
    -----
    mode: string
        wheter to compute the mean error for each pixel or for each frame
    """

    def __init__(self, mode="frame"):
        """ Initializer"""
        assert mode in ["pixel", "frame"]
        super().__init__()
        self.mode = mode

    def forward(self, pred, gt):
        """ Forward pass """
        B = pred.shape[0]
        error = (pred - gt).pow(2).sum()
        if(self.mode == "pixel"):
            loss = error.mean()
        elif(self.mode == "frame"):
            loss = error / B
        else:
            raise NotImplementedError()
        return loss


def segmentation_ari(pred, gt, use_bkg=False):
    """
    Computing the ARI metric for quantifying instance segmentation. We consider
    foreground pixel assignments as cluster assignemnts, and measure a clustering
    evaluation

    Args:
    -----
    pred, gt: torch Tensor / numpy array
        ground truth and predicted instance semgmentations of an image
    use_bkg: boolean
        If True, background is also considered as cluster for evaluation. Otherwise,
        we only use the instance segmentation masks
    """
    pred, gt = pred.flatten(), gt.flatten()
    if not use_bkg:   # removing background pixels
        foreground_px = gt > 0
        pred, gt = pred[foreground_px], gt[foreground_px]
    ari = ARI(labels_true=gt, labels_pred=pred)

    return ari


def binarize_masks(masks, thr=0.1):
    """
    Binarizing object templates for evaluation.
    Pixels are naively thresholded based on their value
    """
    if(len(masks.shape) == 3):
        masks = masks.unsqueeze(0)

    if(len(masks.shape) == 4 and masks.shape[-3]==3):
        r, g, b = masks[:, 0, :, :], masks[:, 1, :, :], masks[:, 2, :, :]
        gray_mask = 0.5 * r + 0.5 * g + 0.5 * b
    elif(len(masks.shape) == 5 and masks.shape[-3]==3):
        r, g, b = masks[:, :, 0, :, :], masks[:, :, 1, :, :], masks[:, :, 2, :, :]
        gray_mask = 0.5 * r + 0.5 * g + 0.5 * b
    else:
        gray_mask = masks

    binarized_masks = torch.zeros_like(gray_mask)
    binarized_masks[gray_mask > thr] = 1
    return binarized_masks


def fill_mask(mask, thr=0.8):
    """
    Filling some wholes in the binary masks using a voting apprached, which is
    implemented as a Average Pooling + Thresholding. Solves the common issue of
    small corners of the segmentation mask not
    """
    pooled_mask = F.avg_pool2d(mask, kernel_size=3, padding=1, stride=1)
    ids = (pooled_mask > thr)
    mask[ids] = 1
    return mask


def fix_borders(mask):
    """
    Filling some gaps in the mask edges as postprocessign step
    Makes masks qualitative worse, but optimizes for foreground ARI evaluation
    """
    N_ROWS, N_COLS = mask.shape[-2], mask.shape[-1]

    # filling gaps in a row-wise manner
    for i in range(1, N_COLS-1):
        mask[..., i] += torch.max(mask[..., :i], dim=-1)[0] * torch.max(mask[..., i+1:], dim=-1)[0]
        mask[..., i] = mask[..., i].clamp(0, 1)

    # filling gaps in a column-wise manner
    for i in range(1, N_ROWS-1):
        mask[..., i, :] += torch.max(mask[..., :i, :], dim=-2)[0] * torch.max(mask[..., i+1:, :], dim=-2)[0]
        mask[..., i, :] = mask[..., i, :].clamp(0, 1)

    return mask


def calculate_iou(mask1, mask2):
    '''
    Calculate IoU of two segmentation masks.
    https://github.com/ecker-lab/object-centric-representation-benchmark

    Args:
    -----
        mask1: HxW
        mask2: HxW
    '''
    eps = np.finfo(float).eps
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)
    union = ((np.sum(mask1) + np.sum(mask2) - np.sum(mask1*mask2)))
    iou = np.sum(mask1*mask2) / (union + eps)
    iou = 1. if union == 0. else iou
    return iou


def compute_dists_per_frame(gt_frame, pred_frame):
    """
    Computing a pairwise distance matrix between the object is a frame
    """
    H, W = 64,64
    MIN_PIX = 5
    IOU_THR = 0.5

    n_pred = len(pred_frame)
    n_gt = len(gt_frame["masks"])

    # accumulate pred masks for frame
    preds = []
    for j in range(n_pred):
        mask = binarize_masks(pred_frame[j]).numpy()
        # excluding duplicates
        cur_items = [p.tolist() for p in preds]
        if mask.tolist() in cur_items:
            continue
        # excluding background
        if mask.sum() > MIN_PIX:
            preds.append(mask)
    preds = np.array(preds)

    # accumulate gt masks for frame
    gts = []
    gt_ids = []
    for h in range(n_gt):
        # excluding gt-background mask
        cur_id = gt_frame["ids"][h]
        if(not isinstance(cur_id, int) and "bg" in cur_id):
            continue
        mask = decode_rle(gt_frame['masks'][h], (H, W))
        if mask.sum() > MIN_PIX:
            gts.append(mask)
            gt_ids.append(gt_frame['ids'][h])
    gts = np.array(gts)

    # compute pairwise distances
    dists = np.ones((len(gts), len(preds)))
    for h in range(len(gts)):
        for j in range(len(preds)):
            dists[h, j] = calculate_iou(gts[h], preds[j])

    dists = 1. - dists
    dists[dists > IOU_THR] = np.nan

    if(len(gt_ids) > 0):
        # pred_ids = np.arange(gt_ids[0], gt_ids[0] + len(preds)).tolist()
        pred_ids = np.arange(len(preds)).tolist()
    else:
        pred_ids = []
    return dists, gt_ids, pred_ids


def compute_mot_metrics(acc, summary):
    '''
    https://github.com/ecker-lab/object-centric-representation-benchmark
    Args:
    -----
        acc: motmetric accumulator
        summary: pandas dataframe with mometrics summary
    '''

    df = acc.mot_events
    df = df[(df.Type != 'RAW')
            & (df.Type != 'MIGRATE')
            & (df.Type != 'TRANSFER')
            & (df.Type != 'ASCEND')]
    obj_freq = df.OId.value_counts()
    n_objs = len(obj_freq)
    tracked = df[df.Type == 'MATCH']['OId'].value_counts()
    detected = df[df.Type != 'MISS']['OId'].value_counts()

    track_ratios = tracked.div(obj_freq).fillna(0.)
    detect_ratios = detected.div(obj_freq).fillna(0.)

    summary['mostly_tracked'] = track_ratios[track_ratios >= 0.8].count() / n_objs * 100
    summary['mostly_detected'] = detect_ratios[detect_ratios >= 0.8].count() / n_objs * 100

    n = summary['num_objects'][0]
    summary['num_matches'] = (summary['num_matches'][0] / n * 100)
    summary['num_false_positives'] = (summary['num_false_positives'][0] / n * 100)
    summary['num_switches'] = (summary['num_switches'][0] / n * 100)
    summary['num_misses'] = (summary['num_misses'][0] / n * 100)

    summary['mota'] = (summary['mota'][0] * 100)
    summary['motp'] = ((1. - summary['motp'][0]) * 100)

    return summary


def decode_rle(mask_rle, shape):
    '''
    from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
