"""Visualize the depth-edge mask used by edge_loss_boost / lambda_stack /
front_acc at multiple `edge_depth_grad_thresh` values, to pick a good
default by eye. CPU-only so it doesn't fight a running training run.

Usage:
    .venv/bin/python visualize_edge_thresholds.py \
        -ec configs/t4/exp_t4.yaml \
        -dc configs/t4/dynamic/example.yaml \
        -s ~/.webauto/data/data/annotation_dataset/<UUID>/<VER>

Output: output/edge_threshold_vis/<sensor>_frame<NNNN>.png
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.arguments import parse
from vis_rerun import load_scene_fast


def compute_edge_mask(gt_depth, gt_mask, thresh):
    """Mirror of train.py:309-330 (CPU)."""
    gt_d = gt_depth.unsqueeze(0).unsqueeze(0).float()
    sx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
                      dtype=gt_d.dtype)
    sy = sx.transpose(-1, -2).contiguous()
    gx = F.conv2d(gt_d, sx, padding=1).squeeze()
    gy = F.conv2d(gt_d, sy, padding=1).squeeze()
    edge_mag = torch.sqrt(gx * gx + gy * gy)
    invalid = (~gt_mask).float().unsqueeze(0).unsqueeze(0)
    any_invalid = F.max_pool2d(invalid, 3, stride=1, padding=1).squeeze() > 0.5
    edge_mask = (edge_mag > thresh) & ~any_invalid & gt_mask
    return edge_mask, edge_mag


def compute_sky_boundary_mask(gt_mask):
    """Valid pixels with at least one invalid (no-return) neighbour in
    their 3x3 window. Captures the horizon ring / building-sky / tree-sky
    boundary where phantoms collect — independent of any depth threshold.
    """
    invalid = (~gt_mask).float().unsqueeze(0).unsqueeze(0)
    any_invalid = F.max_pool2d(invalid, 3, stride=1, padding=1).squeeze() > 0.5
    return any_invalid & gt_mask


def colorize_depth(depth, mask, vmin=0.5, vmax=80.0):
    """Return uint8 (H, W, 3) RGB with no-return pixels darkened."""
    d_clamped = depth.clamp(vmin, vmax)
    d_norm = ((d_clamped - vmin) / (vmax - vmin)).cpu().numpy()
    cmap = plt.get_cmap("turbo")
    rgb = (cmap(d_norm)[..., :3] * 255).astype(np.uint8)
    rgb[~mask.cpu().numpy()] = 25  # no-return → near-black
    return rgb


def overlay_edge(rgb, edge_mask, color=(255, 0, 0)):
    """Paint edge pixels in solid red on top of colored depth."""
    em = edge_mask.cpu().numpy()
    out = rgb.copy()
    out[em] = color
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", "--exp_config_path", required=True)
    p.add_argument("-dc", "--data_config_path", required=True)
    p.add_argument("-s", "--source_dir", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="Frame IDs to visualize. Default: 2 frames near the "
                        "middle of the train sequence.")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[5, 10, 15, 20, 30, 50, 80, 120])
    p.add_argument("--out_dir", default="output/edge_threshold_vis")
    p.add_argument("--sensor", default=None,
                   help="Sensor name; default = all sensors")
    launch = p.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.source_dir = launch.source_dir
    args.unet = ""
    args.eval_type = "all"
    args.rerun_save = ""

    lidars, _ = load_scene_fast(args)
    if launch.sensor:
        use_sensors = [launch.sensor]
    else:
        use_sensors = sorted(lidars.keys())

    os.makedirs(launch.out_dir, exist_ok=True)
    summary = []  # (sensor, frame, thresh, n_edge, ratio_pct)

    for sname in use_sensors:
        if sname not in lidars:
            print(f"skip: sensor {sname!r} not in scene")
            continue
        lidar = lidars[sname]
        train_frames = list(lidar.train_frames)
        if not train_frames:
            print(f"skip: sensor {sname!r} has no train frames")
            continue
        if launch.frames:
            frames = [f for f in launch.frames
                      if f in lidar.range_image_return1]
        else:
            mid = len(train_frames) // 2
            frames = train_frames[mid:mid + 2]

        for fid in frames:
            gt_depth = lidar.get_depth(fid).cpu().float()
            gt_mask = lidar.get_mask(fid).cpu()
            H, W = gt_depth.shape
            total_px = H * W

            panels = []
            base_rgb = colorize_depth(gt_depth, gt_mask)
            valid_px = int(gt_mask.sum().item())
            panels.append((f"GT depth   ({sname}, frame {fid}, "
                           f"{valid_px}/{total_px} valid)", base_rgb))

            # Sky-boundary mask (no depth threshold; pure gt_mask boundary).
            # Shown in BLUE so it's distinguishable from the Sobel edges
            # (red) below. This is the band CURRENTLY excluded by train.py.
            sky_bd = compute_sky_boundary_mask(gt_mask)
            n_sky = int(sky_bd.sum().item())
            sky_ratio = n_sky / max(valid_px, 1) * 100.0
            sky_overlay = overlay_edge(base_rgb, sky_bd, color=(0, 120, 255))
            panels.append((f"sky-boundary only (BLUE, no thr)   "
                           f"edge_pixels={n_sky:>6,}   "
                           f"{sky_ratio:>4.1f}% of valid   "
                           f"-- currently EXCLUDED by train.py",
                           sky_overlay))
            summary.append((sname, fid, "sky-bd", n_sky, sky_ratio))

            for t in launch.thresholds:
                em, _ = compute_edge_mask(gt_depth, gt_mask, t)
                n_edge = int(em.sum().item())
                ratio = n_edge / total_px * 100.0
                ratio_valid = n_edge / max(valid_px, 1) * 100.0
                # Show Sobel edges in red AND sky-boundary in blue together,
                # so the user can see what the combined mask would look like.
                overlay = overlay_edge(base_rgb, sky_bd, color=(0, 120, 255))
                overlay = overlay_edge(overlay, em, color=(255, 0, 0))
                combined_count = int((em | sky_bd).sum().item())
                combined_ratio = combined_count / max(valid_px, 1) * 100.0
                title = (f"thr={t:>5.1f}   sobel(red)={n_edge:>6,} "
                         f"({ratio_valid:>4.1f}%)   "
                         f"sobel|sky={combined_count:>6,} "
                         f"({combined_ratio:>4.1f}% of valid)")
                panels.append((title, overlay))
                summary.append((sname, fid, t, n_edge, ratio_valid))

            # Native-resolution stacked PNG: each panel is the full 128×1800
            # range image at 1:1 pixel scale, with a small label band above.
            # No matplotlib axes — keeps text and pixels both legible.
            label_h = 22
            pad = 2
            cells = []
            for label, rgb in panels:
                band = np.full((label_h, rgb.shape[1], 3), 255, dtype=np.uint8)
                cv2.putText(band, label, (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cell = np.concatenate([band, rgb], axis=0)
                cells.append(cell)
                cells.append(np.full((pad, rgb.shape[1], 3), 220,
                                     dtype=np.uint8))
            stacked = np.concatenate(cells[:-1], axis=0)
            out_path = os.path.join(launch.out_dir,
                                    f"{sname}_frame{fid:04d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
            print(f"[saved] {out_path}  ({stacked.shape[1]}×{stacked.shape[0]})")

    # Console summary table
    if summary:
        print("\n=== Summary: edge pixel counts ===")
        print(f"{'sensor':<24}{'frame':>6}{'thresh':>9}{'edges':>10}"
              f"{'%valid':>10}")
        for sname, fid, t, n_edge, ratio in summary:
            t_str = f"{t:.1f}" if isinstance(t, (int, float)) else str(t)
            print(f"{sname:<24}{fid:>6}{t_str:>9}{n_edge:>10,}{ratio:>9.1f}%")


if __name__ == "__main__":
    main()
