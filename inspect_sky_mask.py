"""Render the GT sky mask (= ~gt_mask) as a video for visual inspection.

Use this to verify that the "no-return" pixels are distributed where we
expect physically (sky / above horizon / beyond max range) rather than
clustering in a fixed azimuth band, which would indicate a range-image
construction bug (e.g., missing rosbag packets, wrong beam table, off-by-
one in azimuth indexing).

For each frame we stack 4 horizontal panels (top→bottom):
  1. gt_depth          colorized 0..max
  2. gt_mask           white = ray hit, black = no return
  3. sky_mask          inverse of gt_mask (so the suspected "phantom-prone"
                       region is what's WHITE here)
  4. azimuth ruler     0°, 90°, 180°, 270° tick marks so the user can read
                       which compass direction each column corresponds to

Usage:
  .venv/bin/python inspect_sky_mask.py \
      --cache_dir /home/masaya/.webauto/data/.../cache_v5/hesai_top \
      --output sky_mask.mp4 \
      [--fps 5]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def colorize_depth(d, vmax):
    norm = np.clip(d / max(vmax, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_JET)


def build_azimuth_ruler(W, H_strip=16):
    """White strip with vertical tick marks + degree labels for 0°/90°/180°/270°.

    Mirrors the convention used in range2point / _pointcloud_to_range_image:
    column 0 corresponds to azimuth +π (back of vehicle), column W/2 to
    azimuth 0 (front), column W to azimuth -π (back again).
    """
    strip = np.full((H_strip, W, 3), 240, dtype=np.uint8)
    # azimuth(col) = π - 2π*col/W  → col(azimuth) = (π - azimuth) * W / (2π)
    def col(azimuth_rad):
        return int(round((np.pi - azimuth_rad) * W / (2 * np.pi)))
    ticks = [
        (np.pi, "180° (back)"),
        (np.pi / 2, "90° (left)"),
        (0.0, "0° (front)"),
        (-np.pi / 2, "270° (right)"),
    ]
    for az, label in ticks:
        c = col(az) % W
        cv2.line(strip, (c, 0), (c, H_strip - 1), (0, 0, 0), 1)
        cv2.putText(
            strip, label, (max(0, c - 38), H_strip - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return strip


def compute_weak_col_frac(depth, weak_hit_rate=0.5, below_row_start=32):
    """Same heuristic as LiDARSensor.detect_dropped_frames.

    Returns (weak_col_fraction, weak_col_mask) where weak_col_mask is a
    bool array (W,) marking azimuth columns whose downward-beam hit rate
    is below `weak_hit_rate`.
    """
    H, W = depth.shape
    row_start = min(below_row_start, max(H - 1, 0))
    below = depth[row_start:]
    if below.size == 0:
        return 0.0, np.zeros(W, dtype=bool)
    col_hit_rate = (below > 0).mean(axis=0)
    weak_mask = col_hit_rate < weak_hit_rate
    return float(weak_mask.mean()), weak_mask


def render_frame(depth, vmax, weak_col_threshold=0.01, weak_hit_rate=0.5):
    """Stack the four panels for one frame."""
    H, W = depth.shape
    gt_mask = depth > 0
    sky_mask = ~gt_mask
    weak_frac, weak_mask = compute_weak_col_frac(depth, weak_hit_rate)
    is_dropped = weak_frac > weak_col_threshold

    panel_depth = colorize_depth(depth, vmax)
    panel_depth[~gt_mask] = 0  # mask depth==0 to black so structure is clearer

    panel_hit = np.zeros((H, W, 3), dtype=np.uint8)
    panel_hit[gt_mask] = 255

    panel_sky = np.zeros((H, W, 3), dtype=np.uint8)
    panel_sky[sky_mask] = 255  # WHITE = "no GT return" = where phantoms can hide

    # Highlight columns whose downward-beam hit rate fell below threshold —
    # these are the azimuth bands the drop detector flags as packet-loss.
    if weak_mask.any():
        red_overlay = weak_mask[None, :].repeat(H, axis=0)
        panel_sky[red_overlay] = (0, 0, 255)  # BGR red

    ruler = build_azimuth_ruler(W, H_strip=16)

    out = np.concatenate([panel_depth, panel_hit, panel_sky, ruler], axis=0)

    # Overlay frame-level stats
    sky_pct = 100 * sky_mask.mean()
    cv2.putText(
        out, f"sky_mask = {sky_pct:.1f}%   weak_col = {weak_frac*100:.1f}%",
        (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
    )

    if is_dropped:
        # Thick red border to make dropped frames obvious in the timeline
        border = 6
        cv2.rectangle(
            out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1),
            (0, 0, 255), border,
        )
        # Big "DROPPED" stamp
        cv2.putText(
            out, "DROPPED", (W // 2 - 80, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True,
                   help="Path to .../cache_v5/<sensor_name>/")
    p.add_argument("--output", default="sky_mask.mp4")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_end", type=int, default=None,
                   help="Exclusive end; defaults to all available frames")
    p.add_argument("--weak_col_threshold", type=float, default=0.01,
                   help="Fraction of weak columns above which a frame is "
                        "flagged as dropped (default: 0.01)")
    p.add_argument("--weak_hit_rate", type=float, default=0.5,
                   help="Per-column downward hit-rate threshold below which "
                        "a column is considered 'weak' (default: 0.5)")
    args = p.parse_args()

    cache = Path(args.cache_dir)
    pt_files = sorted(
        cache.glob("range_image_frame_*.pt"),
        key=lambda f: int(f.stem.split("_")[-1]),
    )
    if args.frame_end is None:
        args.frame_end = len(pt_files)
    pt_files = pt_files[args.frame_start:args.frame_end]
    if not pt_files:
        print(f"No .pt files found under {cache}", file=sys.stderr)
        sys.exit(1)

    # First pass: find global vmax for stable colormap
    print("Scanning frames for global vmax ...")
    vmax = 0.0
    for f in pt_files:
        d = torch.load(f, weights_only=True)["r1"][..., 0].numpy()
        vmax = max(vmax, float(d.max()))
    print(f"  vmax = {vmax:.2f} m")

    # Sanity: print per-frame stats + drop detection
    print("\nFrame  | gt_hit %  | sky_mask % | weak_col %  | status")
    drop_list = []
    for f in pt_files:
        d = torch.load(f, weights_only=True)["r1"][..., 0].numpy()
        hit = (d > 0).mean() * 100
        weak_frac, _ = compute_weak_col_frac(d, args.weak_hit_rate)
        idx = int(f.stem.split("_")[-1])
        dropped = weak_frac > args.weak_col_threshold
        tag = " ⚠️ DROPPED" if dropped else ""
        if dropped:
            drop_list.append(idx)
        print(f"  {idx:>3}  |  {hit:5.1f}%   |  {100-hit:5.1f}%    |  {weak_frac*100:5.1f}%    |{tag}")
    print(f"\nDropped frames: {drop_list}")

    # Render frames
    first = torch.load(pt_files[0], weights_only=True)["r1"][..., 0].numpy()
    sample = render_frame(first, vmax, args.weak_col_threshold, args.weak_hit_rate)
    H_out, W_out, _ = sample.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (W_out, H_out))
    if not writer.isOpened():
        print(f"Failed to open VideoWriter for {args.output}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWriting {len(pt_files)} frames to {args.output} ({W_out}x{H_out} @ {args.fps}fps) ...")
    for f in pt_files:
        d = torch.load(f, weights_only=True)["r1"][..., 0].numpy()
        frame = render_frame(d, vmax, args.weak_col_threshold, args.weak_hit_rate)
        # Stamp frame number
        idx = int(f.stem.split("_")[-1])
        cv2.putText(
            frame, f"frame {idx}", (W_out - 110, 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
