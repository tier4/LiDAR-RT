#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import cv2
import matplotlib


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def colorize_depth(depth_np, vmin, vmax, mask=None, log_scale=False):
    """Colorize a depth map with cv2 JET colormap. Returns BGR uint8.

    Behavior is identical between training-time and offline visualization:
    map [vmin, vmax] → [0, 255], no internal renormalization. Masked pixels
    are set to black. Use cv2.cvtColor(..., cv2.COLOR_BGR2RGB) when feeding
    the result to rerun or wandb.

    log_scale: when True, apply log1p so 1-10 m occupies as much color range
    as 10-100 m. Useful for LiDAR which mixes near (~1 m) and far (~200 m)
    returns in the same frame.
    """
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np.squeeze(-1)
    if log_scale:
        d = np.log1p(np.clip(depth_np, 0.0, None))
        v_lo = np.log1p(max(vmin, 0.0))
        v_hi = np.log1p(max(vmax, vmin + 1e-3))
    else:
        d = depth_np
        v_lo, v_hi = vmin, vmax
    drange = max(v_hi - v_lo, 1e-6)
    norm = np.clip((d - v_lo) / drange, 0.0, 1.0)
    img = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_JET)
    if mask is not None:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        img[~mask.astype(bool)] = 0
    return img


def colorize_intensity(intensity_np, mask=None,
                       vmin=0.0, vmax=1.0, log_scale=False):
    """Colorize an intensity map with cv2 JET colormap. Returns BGR uint8.

    Defaults (vmin=0, vmax=1, log_scale=False) preserve the original
    behaviour of treating the input as already normalised. For raw LiDAR
    uint8 intensity (T4: p99 ≈ 64, retroreflectors at 255), pass
    vmin=0, vmax=64, log_scale=True to match the rerun 3D point-cloud
    ramp — log spreads the visible spectrum across the dense low-
    intensity band where ~99% of points live.
    """
    if intensity_np.ndim == 3 and intensity_np.shape[-1] == 1:
        intensity_np = intensity_np.squeeze(-1)
    if log_scale:
        v = np.log1p(np.maximum(intensity_np, 0.0))
        lo = np.log1p(max(vmin, 0.0))
        hi = np.log1p(max(vmax, vmin + 1e-3))
    else:
        v = intensity_np
        lo, hi = float(vmin), float(vmax)
    norm = np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_JET)
    if mask is not None:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        img[~mask.astype(bool)] = 0
    return img

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def color_mapping(tensor, colormap_, reversed=False):

    # To numpy
    tensor_type = ""
    if isinstance(tensor, torch.Tensor):
        tensor_type = "torch_cuda" if tensor.is_cuda else "torch_cpu"
        tensor = tensor.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        tensor_type = "numpy"
    # Normalize
    tensor = tensor.astype(np.float32)
    org_shape = tensor.shape
    tensor = tensor.reshape(-1)
    if tensor.min() != 0 or tensor.max() != 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if reversed:
        tensor = 1 - tensor
    # mapping

    if isinstance(colormap_, int):
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(tensor * 255), colormap_), cv2.COLOR_BGR2RGB)
        color =  np.array(color[:, 0, :]).astype(np.float32)  / 255.
    elif isinstance(colormap_, matplotlib.colors.Colormap):
        color = colormap_(tensor)
        color = color[:, 0:3] * color[:, 3:4] + (1 - color[:, 3:4])

    new_shape = list(org_shape)
    new_shape.append(3)
    color = color.reshape(new_shape)
    # Turn back
    if tensor_type == "torch_cuda":
        color = torch.from_numpy(color).cuda()
    elif tensor_type == "torch_cpu":
        color = torch.from_numpy(color)

    return color
