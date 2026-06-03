"""Render a checkpoint's depth comparison panel as a static PNG.

Reuses the same colormap / masks / log-scale that train.py logs to wandb,
so the output is what you'd see in viz/depth_compare for that checkpoint.

Usage:
  .venv/bin/python render_checkpoint.py \
      -ec configs/t4/exp_t4.yaml \
      -dc configs/t4/dynamic/example.yaml \
      -s /path/to/dataset/version_dir \
      -m output/.../models/ckpt_it_10000.pth \
      --out /tmp/depth_compare_10000.png \
      [--frame 5]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.scene import GaussianModel
from lib.utils.image_utils import colorize_depth
from vis_rerun import load_scene_fast, load_gaussians_fast


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", "--exp_config_path", required=True)
    p.add_argument("-dc", "--data_config_path", required=True)
    p.add_argument("-m", "--model", required=True, help="checkpoint .pth")
    p.add_argument("-s", "--source_dir", default="",
                   help="override source_dir from data config")
    p.add_argument("--out", default="depth_compare.png")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame id to render; default: first surviving train frame")
    launch = p.parse_args()
    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = ""
    args.eval_type = "all"
    args.rerun_save = ""
    if launch.source_dir:
        args.source_dir = launch.source_dir
    return args, launch


def main():
    args, launch = build_args()

    # Load scene + Gaussians (mirrors vis_rerun.main)
    lidars, bboxes = load_scene_fast(args)
    num_g = len(torch.load(args.model_path, weights_only=False)[0])
    gaussians, first_iter = load_gaussians_fast(args.model_path, num_g, args)

    # Pick frame
    first_lidar = next(iter(lidars.values()))
    if launch.frame is not None:
        viz_frame = launch.frame
    else:
        viz_frame = first_lidar.train_frames[0]
    sensor_name = next(iter(lidars.keys()))
    lidar = lidars[sensor_name]

    background = torch.tensor([0, 0, 1], device="cuda").float()
    print(f"Rendering frame {viz_frame} from {args.model_path} (iter {first_iter}) ...")

    t0 = time.time()
    render_pkg = raytracing(viz_frame, gaussians, lidar, background, args)
    print(f"  render took {time.time()-t0:.2f}s")

    rendered_depth = render_pkg["depth"].squeeze(-1).detach().cpu().numpy()
    gt_depth = lidar.get_depth(viz_frame).cpu().numpy()
    gt_mask = lidar.get_mask(viz_frame).cpu().numpy().astype(bool)

    viz_min_depth = float(getattr(args, "viz_min_depth", 0.0))
    dmax = max(float(gt_depth.max()), float(rendered_depth.max()))
    dmin = 0.0
    pred_hit = rendered_depth > 0
    pred_phantom = pred_hit & (rendered_depth <= viz_min_depth)

    gt_img = colorize_depth(gt_depth, dmin, dmax, mask=gt_mask, log_scale=True)
    pred_img = colorize_depth(rendered_depth, dmin, dmax, mask=pred_hit, log_scale=True)
    if pred_phantom.any():
        pred_img[pred_phantom] = (255, 0, 255)  # BGR magenta = phantom (depth<=viz_min)

    err_valid = gt_mask & pred_hit
    err_map = np.zeros_like(gt_depth, dtype=np.float32)
    err_map[err_valid] = np.abs(gt_depth[err_valid] - rendered_depth[err_valid])
    if err_valid.any():
        err_cap = float(np.quantile(err_map[err_valid], 0.99))
    else:
        err_cap = 1.0
    err_cap = max(err_cap, 1e-3)
    err_img = colorize_depth(err_map, 0.0, err_cap, mask=err_valid)

    stacked = np.concatenate([gt_img, pred_img, err_img], axis=0)

    # Add caption strip at top
    H, W, _ = stacked.shape
    label = np.full((24, W, 3), 30, dtype=np.uint8)
    cv2.putText(
        label,
        f"iter {first_iter} | frame {viz_frame} | top->bot: depth_gt / depth_rendered / error(cap={err_cap:.2f}m)",
        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    stacked = np.concatenate([label, stacked], axis=0)

    cv2.imwrite(launch.out, stacked)
    print(f"Saved: {launch.out}  ({stacked.shape[1]}x{stacked.shape[0]})")


if __name__ == "__main__":
    main()
