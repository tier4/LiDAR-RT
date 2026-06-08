"""Quantify whether high-error ("ghost") range-image pixels concentrate on
depth-image edges.

Re-runs the same fast loader / raytrace path as vis_rerun.py for a given
checkpoint, builds the training-time edge mask (Sobel-magnitude > thr OR
adjacent to no-return pixel), and reports how much of the rendering error
sits inside that mask vs. how much sits well away from edges.

Outputs (under <out_dir>/):
  - per_frame.csv             — one row per (sensor, frame) with aggregate stats
  - per_bin.csv               — error stratified by distance-to-edge (pixels)
  - frame_<F>_<sensor>.png    — per-frame visual: edges, error heatmap, phantoms
  - summary.txt               — pooled stats + headline numbers

Usage:
  uv run python analyze_ghost_vs_edge.py \
      -ec configs/t4/exp_t4.yaml -dc configs/t4/dynamic/example.yaml \
      -s ~/.webauto/data/data/annotation_dataset/<UUID>/<VER> \
      -m output/t4_tuned/test/scene_t4d1/models/ckpt_it_8000.pth \
      --eval-type test --out_dir output/ghost_edge_8000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.scene.unet import UNet
from vis_rerun import load_gaussians_fast, load_scene_fast


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", "--exp_config_path", required=True)
    p.add_argument("-dc", "--data_config_path", required=True)
    p.add_argument("-s", "--source_dir", default="")
    p.add_argument("-m", "--model", required=True, help="checkpoint .pth")
    p.add_argument("-un", "--unet", default="",
                   help="optional UNet weights (file or dir). If empty, raw "
                        "rasterizer raydrop is used.")
    p.add_argument("--eval-type", default="test",
                   choices=["train", "test", "all"])
    p.add_argument("--edge-thresh", type=float, default=15.0,
                   help="Sobel-magnitude threshold for depth edges (matches "
                        "opt.edge_depth_grad_thresh, default 15.0).")
    p.add_argument("--error-thresh", type=float, default=1.0,
                   help="A point is called a 'high-error / ghost' point when "
                        "|gt - rd| > this many meters.")
    p.add_argument("--raydrop-ratio", type=float, default=0.5,
                   help="Rendered raydrop > this means 'predicted no-return'.")
    p.add_argument("--max-bin-px", type=int, default=20,
                   help="Cap distance-to-edge in pixels for the histogram.")
    p.add_argument("--out_dir", default="output/ghost_edge_analysis")
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="Subset of frame ids (otherwise: all per eval-type).")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip saving per-frame PNG visualisations.")
    launch = p.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = launch.unet
    args.eval_type = launch.eval_type
    args.rerun_save = ""
    if launch.source_dir:
        args.source_dir = launch.source_dir
    return launch, args


# ---------- edge mask (mirror train.py / visualize_edge_thresholds.py) ----------

_SX = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
_SY = _SX.transpose(-1, -2).contiguous()


def compute_edges(gt_depth: torch.Tensor, gt_mask: torch.Tensor,
                  thresh: float) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]:
    """Return (sobel_edge, sky_boundary, combined_edge) — all bool (H, W).

    sobel_edge   : |∇gt_depth| > thresh, on rows/cols where both gt sides
                    are valid (no-return neighbours excluded).
    sky_boundary : valid pixel with ≥1 invalid neighbour in 3x3.
    combined     : (sobel_edge | sky_boundary)  — what the training loss
                    treats as the "edge band".
    """
    g = gt_depth.unsqueeze(0).unsqueeze(0).float()
    sx = _SX.to(g.device, g.dtype)
    sy = _SY.to(g.device, g.dtype)
    gx = F.conv2d(g, sx, padding=1).squeeze()
    gy = F.conv2d(g, sy, padding=1).squeeze()
    edge_mag = torch.sqrt(gx * gx + gy * gy)

    invalid = (~gt_mask).float().unsqueeze(0).unsqueeze(0)
    any_invalid = F.max_pool2d(invalid, 3, stride=1, padding=1).squeeze() > 0.5

    sky_bd = any_invalid & gt_mask
    sobel_edge = (edge_mag > thresh) & ~any_invalid & gt_mask
    combined = sobel_edge | sky_bd
    return sobel_edge, sky_bd, combined


def distance_to_edge_px(combined_edge: np.ndarray) -> np.ndarray:
    """For each pixel, pixel distance to the nearest edge pixel. 0 on the
    edge. Edges == True. Returns float32 (H, W)."""
    if not combined_edge.any():
        return np.full(combined_edge.shape, 1e6, dtype=np.float32)
    non_edge = (~combined_edge).astype(np.uint8)
    # distanceTransform: distance from each pixel to the nearest 0-pixel.
    # We want distance from each pixel to the nearest True-edge pixel, so
    # set the edges to 0 and the background to 1.
    return cv2.distanceTransform(non_edge, cv2.DIST_L2, 3)


# ---------- per-frame render ----------

def render_one_frame(frame_id, gaussians, lidar, unet, args, background,
                     use_spatial: bool):
    pkg = raytracing(frame_id, gaussians, lidar, background, args)
    rd_depth = pkg["depth"].detach()
    rd_intensity = pkg["intensity"].detach()
    rd_raydrop = pkg["raydrop"].detach()

    if unet is not None:
        H, W = rd_depth.shape[0], rd_depth.shape[1]
        inp = torch.cat([
            rd_raydrop.reshape(1, H, W),
            rd_intensity.reshape(1, H, W),
            rd_depth.reshape(1, H, W),
        ], dim=0)
        if use_spatial:
            ray_o, ray_d = lidar.get_range_rays(frame_id)
            inp = torch.cat([inp, ray_o.permute(2, 0, 1),
                             ray_d.permute(2, 0, 1)], dim=0)
        rd_raydrop = unet(inp.unsqueeze(0)).detach().reshape(H, W, 1)

    gt_mask = lidar.get_mask(frame_id)        # (H, W) bool
    gt_depth = lidar.get_depth(frame_id)      # (H, W) float
    return gt_depth, gt_mask, rd_depth.squeeze(-1), rd_raydrop.squeeze(-1)


# ---------- visualisation ----------

def colourise_depth(depth_np, mask_np, vmin=0.5, vmax=80.0):
    norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgb = (plt.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~mask_np] = 25
    return rgb


def colourise_error(err_np, valid_np, vmin=0.0, vmax=5.0):
    norm = np.clip((err_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgb = (plt.get_cmap("magma")(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid_np] = 25
    return rgb


def save_frame_panel(out_path, gt_depth_np, gt_mask_np, rd_depth_np,
                     rd_hit_np, both_valid_np, err_np, combined_edge_np,
                     phantom_np, high_err_np, frame_id, sensor_name,
                     edge_thresh, err_thresh):
    panels = []
    H, W = gt_depth_np.shape

    base = colourise_depth(gt_depth_np, gt_mask_np)
    panels.append((f"GT depth ({sensor_name}, frame {frame_id})", base))

    edge_overlay = base.copy()
    edge_overlay[combined_edge_np] = (255, 0, 0)
    panels.append((f"edge mask (sobel>{edge_thresh:g} | sky-bd) "
                   f"= {int(combined_edge_np.sum()):,} px", edge_overlay))

    err_rgb = colourise_error(err_np, both_valid_np, vmin=0.0, vmax=5.0)
    panels.append(("|gt-rd| (m), 0..5 magma", err_rgb))

    high_rgb = base.copy()
    high_rgb[high_err_np] = (255, 255, 0)        # yellow: |err|>thr
    high_rgb[phantom_np] = (255, 0, 255)         # magenta: phantom hits (gt no-return)
    panels.append((f"high-err (yellow, >{err_thresh:g} m) + "
                   f"phantom hit on no-return (magenta)", high_rgb))

    label_h = 22
    pad = 2
    cells = []
    for label, rgb in panels:
        band = np.full((label_h, rgb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(band, label, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cell = np.concatenate([band, rgb], axis=0)
        cells.append(cell)
        cells.append(np.full((pad, rgb.shape[1], 3), 220, dtype=np.uint8))
    stacked = np.concatenate(cells[:-1], axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))


# ---------- main ----------

def main():
    launch, args = build_args()
    os.makedirs(launch.out_dir, exist_ok=True)

    lidars, _ = load_scene_fast(args)
    sensor_names = list(lidars.keys())

    # Determine number of gaussian models from checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    num_gaussians = len(ckpt[0])
    del ckpt
    gaussians, first_iter = load_gaussians_fast(args.model_path, num_gaussians,
                                                args)

    unets = {}
    if args.unet:
        in_channels = 9 if args.refine.use_spatial else 3
        if os.path.isfile(args.unet):
            unet = UNet(in_channels=in_channels, out_channels=1).cuda()
            unet.load_state_dict(torch.load(args.unet))
            unets[sensor_names[0]] = unet
        else:
            unet_dir = args.unet if os.path.isdir(args.unet) \
                else os.path.dirname(args.model_path)
            for sname in sensor_names:
                p = os.path.join(unet_dir, f"unet_{sname}.pth")
                if os.path.exists(p):
                    unet = UNet(in_channels=in_channels, out_channels=1).cuda()
                    unet.load_state_dict(torch.load(p))
                    unets[sname] = unet

    background = torch.tensor([0, 0, 1], device="cuda").float()

    first_lidar = next(iter(lidars.values()))
    train_frames = list(first_lidar.train_frames)
    eval_frames = list(first_lidar.eval_frames)
    if args.eval_type == "train":
        all_frames = train_frames
    elif args.eval_type == "test":
        all_frames = eval_frames
    else:
        all_frames = sorted(set(train_frames) | set(eval_frames))
    if launch.frames:
        all_frames = [f for f in all_frames if f in launch.frames]

    print(f"[setup] iter {first_iter}, "
          f"{len(sensor_names)} sensor(s), {len(all_frames)} frame(s), "
          f"eval-type={args.eval_type}, edge-thresh={launch.edge_thresh}, "
          f"error-thresh={launch.error_thresh}", flush=True)

    per_frame_rows = []
    bin_agg = defaultdict(lambda: {  # key: dist_px (int, capped)
        "n_pixels": 0,
        "sum_err": 0.0,
        "n_high_err": 0,
        "n_phantom": 0,
    })
    pooled = {
        "valid_both": 0,
        "valid_either": 0,
        "edge_px": 0,
        "edge_high_err": 0,
        "non_edge_high_err": 0,
        "total_high_err": 0,
        "edge_phantom": 0,
        "non_edge_phantom": 0,
        "total_phantom": 0,
        "sum_err_edge": 0.0,
        "sum_err_non_edge": 0.0,
        "n_err_edge": 0,
        "n_err_non_edge": 0,
    }

    for frame_id in all_frames:
        for sname in sensor_names:
            lidar = lidars[sname]
            unet = unets.get(sname)

            gt_depth, gt_mask, rd_depth, rd_raydrop = render_one_frame(
                frame_id, gaussians, lidar, unet, args, background,
                args.refine.use_spatial,
            )

            gt_depth_np = gt_depth.cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy().astype(bool)
            rd_depth_np = rd_depth.cpu().numpy()
            rd_raydrop_np = rd_raydrop.cpu().numpy()

            rd_hit_np = (rd_raydrop_np < launch.raydrop_ratio) & (rd_depth_np > 0)

            sobel_edge, sky_bd, combined = compute_edges(
                gt_depth, gt_mask, launch.edge_thresh,
            )
            combined_np = combined.cpu().numpy()

            dist_px = distance_to_edge_px(combined_np)  # float32, pixels

            both_valid = gt_mask_np & rd_hit_np
            err_np = np.zeros_like(gt_depth_np, dtype=np.float32)
            err_np[both_valid] = np.abs(
                gt_depth_np[both_valid] - rd_depth_np[both_valid]
            )
            high_err = both_valid & (err_np > launch.error_thresh)

            # phantom hit: rendered says HIT but GT had no return at all
            phantom = rd_hit_np & (~gt_mask_np)

            # --- aggregate (pooled) ---
            pooled["valid_both"] += int(both_valid.sum())
            pooled["valid_either"] += int((gt_mask_np | rd_hit_np).sum())
            pooled["edge_px"] += int(combined_np.sum())
            pooled["total_high_err"] += int(high_err.sum())
            pooled["edge_high_err"] += int((high_err & combined_np).sum())
            pooled["non_edge_high_err"] += int((high_err & ~combined_np).sum())
            pooled["total_phantom"] += int(phantom.sum())
            pooled["edge_phantom"] += int((phantom & combined_np).sum())
            pooled["non_edge_phantom"] += int((phantom & ~combined_np).sum())

            err_on_edge_mask = both_valid & combined_np
            err_off_edge_mask = both_valid & ~combined_np
            pooled["sum_err_edge"] += float(err_np[err_on_edge_mask].sum())
            pooled["sum_err_non_edge"] += float(err_np[err_off_edge_mask].sum())
            pooled["n_err_edge"] += int(err_on_edge_mask.sum())
            pooled["n_err_non_edge"] += int(err_off_edge_mask.sum())

            # --- aggregate (per bin) ---
            cap = launch.max_bin_px
            d_int = np.minimum(np.round(dist_px).astype(np.int32), cap)
            # For distance bins we care about (a) every pixel where we have
            # an error sample (both_valid) and (b) phantom counts (rd hit on
            # no-return). Treat them as separate populations.
            for d in range(cap + 1):
                m = (d_int == d)
                bv_m = m & both_valid
                ph_m = m & phantom
                he_m = m & high_err
                if bv_m.any() or ph_m.any():
                    bin_agg[d]["n_pixels"] += int(bv_m.sum())
                    bin_agg[d]["sum_err"] += float(err_np[bv_m].sum())
                    bin_agg[d]["n_high_err"] += int(he_m.sum())
                    bin_agg[d]["n_phantom"] += int(ph_m.sum())

            # --- per-frame row ---
            row = {
                "frame": frame_id,
                "sensor": sname,
                "H": gt_depth_np.shape[0],
                "W": gt_depth_np.shape[1],
                "valid_gt": int(gt_mask_np.sum()),
                "valid_rd_hit": int(rd_hit_np.sum()),
                "valid_both": int(both_valid.sum()),
                "edge_px": int(combined_np.sum()),
                "sobel_only_px": int(sobel_edge.cpu().numpy().sum()),
                "sky_bd_px": int(sky_bd.cpu().numpy().sum()),
                "mae_all": float(err_np[both_valid].mean()) if both_valid.any() else 0.0,
                "mae_edge": (float(err_np[err_on_edge_mask].mean())
                             if err_on_edge_mask.any() else 0.0),
                "mae_non_edge": (float(err_np[err_off_edge_mask].mean())
                                 if err_off_edge_mask.any() else 0.0),
                "n_high_err": int(high_err.sum()),
                "n_high_err_on_edge": int((high_err & combined_np).sum()),
                "n_phantom": int(phantom.sum()),
                "n_phantom_on_edge": int((phantom & combined_np).sum()),
            }
            per_frame_rows.append(row)

            print(f"[frame {frame_id:>4} {sname}] "
                  f"MAE all={row['mae_all']:.3f} m, "
                  f"edge={row['mae_edge']:.3f} m, "
                  f"non-edge={row['mae_non_edge']:.3f} m | "
                  f"high-err {row['n_high_err']:,} "
                  f"({row['n_high_err_on_edge']/max(row['n_high_err'],1)*100:.1f}% on edge) | "
                  f"phantom {row['n_phantom']:,} "
                  f"({row['n_phantom_on_edge']/max(row['n_phantom'],1)*100:.1f}% on edge)",
                  flush=True)

            if not launch.no_viz:
                out_png = os.path.join(
                    launch.out_dir,
                    f"frame_{frame_id:04d}_{sname}.png",
                )
                save_frame_panel(
                    out_png, gt_depth_np, gt_mask_np, rd_depth_np, rd_hit_np,
                    both_valid, err_np, combined_np, phantom, high_err,
                    frame_id, sname, launch.edge_thresh, launch.error_thresh,
                )

    # --- write per_frame.csv ---
    fp = os.path.join(launch.out_dir, "per_frame.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        w.writeheader()
        for r in per_frame_rows:
            w.writerow(r)
    print(f"[saved] {fp}", flush=True)

    # --- write per_bin.csv ---
    fp = os.path.join(launch.out_dir, "per_bin.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dist_px", "n_valid_both", "mean_err",
                    "n_high_err", "n_phantom",
                    "frac_high_err_in_bin", "frac_phantom_in_bin"])
        total_he = sum(b["n_high_err"] for b in bin_agg.values()) or 1
        total_ph = sum(b["n_phantom"] for b in bin_agg.values()) or 1
        for d in sorted(bin_agg):
            b = bin_agg[d]
            mean_err = b["sum_err"] / max(b["n_pixels"], 1)
            w.writerow([d, b["n_pixels"], f"{mean_err:.4f}",
                        b["n_high_err"], b["n_phantom"],
                        f"{b['n_high_err']/total_he:.4f}",
                        f"{b['n_phantom']/total_ph:.4f}"])
    print(f"[saved] {fp}", flush=True)

    # --- pooled summary ---
    edge_ratio = (pooled["edge_px"] / max(pooled["valid_both"], 1)) * 100.0
    he_on_edge = (pooled["edge_high_err"] / max(pooled["total_high_err"], 1)) * 100.0
    ph_on_edge = (pooled["edge_phantom"] / max(pooled["total_phantom"], 1)) * 100.0
    mae_edge = pooled["sum_err_edge"] / max(pooled["n_err_edge"], 1)
    mae_non = pooled["sum_err_non_edge"] / max(pooled["n_err_non_edge"], 1)

    # density: per-pixel rate
    he_density_edge = pooled["edge_high_err"] / max(pooled["n_err_edge"], 1)
    he_density_non = pooled["non_edge_high_err"] / max(pooled["n_err_non_edge"], 1)
    he_lift = he_density_edge / max(he_density_non, 1e-12)

    summary_path = os.path.join(launch.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        def w(line=""):
            print(line, file=f)
            print(line)
        w(f"# Ghost-vs-edge analysis  (iter {first_iter})")
        w(f"model:        {args.model_path}")
        w(f"unet:         {args.unet or '(none, raw rasterizer raydrop)'}")
        w(f"eval-type:    {args.eval_type}")
        w(f"edge thresh:  sobel>{launch.edge_thresh}  | sky boundary")
        w(f"err thresh:   |gt-rd| > {launch.error_thresh} m  (= 'high-err / ghost')")
        w(f"raydrop thr:  rd_raydrop < {launch.raydrop_ratio}  (= 'predicted hit')")
        w(f"frames:       {len(all_frames)}   sensors: {sensor_names}")
        w()
        w(f"edge area:         {edge_ratio:5.1f}% of valid_both pixels")
        w()
        w("--- error magnitude ---")
        w(f"MAE on edge band : {mae_edge:.3f} m  ({pooled['n_err_edge']:,} samples)")
        w(f"MAE off edge band: {mae_non:.3f} m  ({pooled['n_err_non_edge']:,} samples)")
        w(f"  ratio edge/non : {mae_edge/max(mae_non,1e-12):5.2f}x")
        w()
        w(f"--- high-error pixels (|err| > {launch.error_thresh:g} m) ---")
        w(f"total high-err: {pooled['total_high_err']:,}")
        w(f"  on edge band: {pooled['edge_high_err']:,}  ({he_on_edge:5.1f}%)")
        w(f"  off edge   :  {pooled['non_edge_high_err']:,}  "
          f"({100 - he_on_edge:5.1f}%)")
        w(f"  density on edge band : {he_density_edge*100:6.3f}% of edge pixels")
        w(f"  density off edge band: {he_density_non*100:6.3f}% of non-edge pixels")
        w(f"  edge / non-edge lift : {he_lift:5.2f}x")
        w()
        w("--- phantom hits (rendered hit on no-return GT pixel) ---")
        w(f"total phantom:  {pooled['total_phantom']:,}")
        w(f"  on edge band: {pooled['edge_phantom']:,}  ({ph_on_edge:5.1f}%)")
        w(f"  off edge   :  {pooled['non_edge_phantom']:,}  "
          f"({100 - ph_on_edge:5.1f}%)")
        w()
        w("Headline (does the hypothesis hold?):")
        if he_lift >= 2.0 or he_on_edge >= 50.0:
            w(f"  YES — high-error pixels are {he_lift:.1f}x denser on the "
              f"edge band and the edge band (≈{edge_ratio:.0f}% of valid area) "
              f"holds {he_on_edge:.0f}% of all high-error pixels.")
        else:
            w(f"  WEAK — only {he_lift:.1f}x lift, edge band holds "
              f"{he_on_edge:.0f}% of high-err pixels vs its {edge_ratio:.0f}% "
              f"area share — concentration is real but modest.")
    print(f"[saved] {summary_path}")

    # --- per-bin plot ---
    if not launch.no_viz:
        ds = sorted(bin_agg)
        n_pix = np.array([bin_agg[d]["n_pixels"] for d in ds], dtype=np.float64)
        m_err = np.array([bin_agg[d]["sum_err"] / max(n, 1)
                          for d, n in zip(ds, n_pix)])
        n_he = np.array([bin_agg[d]["n_high_err"] for d in ds], dtype=np.float64)
        rate_he = n_he / np.maximum(n_pix, 1)
        n_ph = np.array([bin_agg[d]["n_phantom"] for d in ds], dtype=np.float64)

        fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        ax[0].plot(ds, m_err, "o-", color="C0")
        ax[0].set_ylabel("mean |gt-rd| (m)")
        ax[0].grid(alpha=0.3)
        ax[0].set_title("Error vs. distance-to-edge (pixels in range image)")

        ax[1].plot(ds, rate_he * 100, "o-", color="C3")
        ax[1].set_ylabel(f"high-err rate (>{launch.error_thresh:g}m) [%]")
        ax[1].grid(alpha=0.3)

        ax[2].bar(ds, n_ph, color="C1", alpha=0.7, label="phantom on no-return")
        ax[2].bar(ds, n_he, color="C3", alpha=0.5, label="high-err")
        ax[2].set_yscale("log")
        ax[2].set_xlabel("distance to nearest edge pixel "
                         f"(capped at {launch.max_bin_px})")
        ax[2].set_ylabel("# pixels (log)")
        ax[2].legend()
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        out_plot = os.path.join(launch.out_dir, "per_bin.png")
        fig.savefig(out_plot, dpi=130)
        plt.close(fig)
        print(f"[saved] {out_plot}")


if __name__ == "__main__":
    main()
