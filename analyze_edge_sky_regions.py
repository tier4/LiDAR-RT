"""Verify the user's hypothesis: ghosts cluster at "edge AND sky_mask"
pixels (= no-return side of depth discontinuities) while non-sky pixels
are well-fitted.

Splits all valid+rendered pixels into 4 mutually-exclusive regions:

    region                 | gt_mask | edge_band | typical interp
    ---------------------- | ------- | --------- | --------------
    A. interior_valid      |   True  |   False   | flat surface inside object
    B. edge_valid          |   True  |   True    | edge but on object side
    C. edge_sky            |  False  |   True    | edge but on sky side  ← HYPOTHESIS
    D. interior_sky        |  False  |   False   | open sky

For each region report:
  - pixel count
  - rendered-hit count (rd_raydrop < 0.5)
  - if there's a GT depth: MAE(rd_depth, gt_depth)
  - if there's no GT depth: "phantom rate" = rendered hits / total pixels

Usage:
  .venv/bin/python analyze_edge_sky_regions.py \
      -ec configs/t4/exp_t4.yaml -dc configs/t4/dynamic/example.yaml \
      -s ~/.webauto/data/data/annotation_dataset/<UUID>/<VER> \
      -m output/t4_tuned/test/scene_t4d1/models/model_it_25000.pth
"""

import argparse
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from vis_rerun import load_gaussians_fast, load_scene_fast


_SX = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
_SY = _SX.transpose(-1, -2).contiguous()


def compute_edge_band(gt_depth, gt_mask, sobel_thr=15.0, dilate_px=1):
    """Mirror of train.py edge_mask construction (with dilation)."""
    g = gt_depth.unsqueeze(0).unsqueeze(0).float()
    sx, sy = _SX.to(g.device, g.dtype), _SY.to(g.device, g.dtype)
    gx = F.conv2d(g, sx, padding=1).squeeze()
    gy = F.conv2d(g, sy, padding=1).squeeze()
    edge_mag = torch.sqrt(gx * gx + gy * gy)
    invalid = (~gt_mask).float().unsqueeze(0).unsqueeze(0)
    any_invalid = F.max_pool2d(invalid, 3, stride=1, padding=1).squeeze() > 0.5
    sobel_edge = (edge_mag > sobel_thr) & ~any_invalid & gt_mask
    sky_boundary = any_invalid & gt_mask
    edge = sobel_edge | sky_boundary
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        em_f = edge.float().unsqueeze(0).unsqueeze(0)
        edge = F.max_pool2d(em_f, k, stride=1, padding=dilate_px).squeeze() > 0.5
    return edge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", required=True, dest="exp_config_path")
    p.add_argument("-dc", required=True, dest="data_config_path")
    p.add_argument("-s", default="", dest="source_dir")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-un", "--unet", default="")
    p.add_argument("--frames", type=int, nargs="+", default=None)
    p.add_argument("--eval-type", default="test", choices=["train","test","all"])
    p.add_argument("--sobel-thr", type=float, default=15.0)
    p.add_argument("--dilate-px", type=int, default=1)
    p.add_argument("--raydrop-ratio", type=float, default=0.5)
    p.add_argument("--out-dir", default="output/edge_sky_regions")
    launch = p.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = launch.unet
    args.eval_type = launch.eval_type
    args.rerun_save = ""
    if launch.source_dir:
        args.source_dir = launch.source_dir

    os.makedirs(launch.out_dir, exist_ok=True)

    lidars, _ = load_scene_fast(args)
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    num_g = len(ckpt[0])
    del ckpt
    gs_list, first_iter = load_gaussians_fast(args.model_path, num_g, args)
    bg = torch.tensor([0, 0, 1], device="cuda").float()
    lidar = next(iter(lidars.values()))
    sname = next(iter(lidars))

    if launch.eval_type == "test":
        frames = list(lidar.eval_frames)
    elif launch.eval_type == "train":
        frames = list(lidar.train_frames)
    else:
        frames = sorted(set(lidar.eval_frames) | set(lidar.train_frames))
    if launch.frames:
        frames = [f for f in frames if f in launch.frames]

    print(f"\n# ckpt: {args.model_path} (iter {first_iter})")
    print(f"# frames: {frames}  sensor: {sname}")
    print(f"# edge: sobel>{launch.sobel_thr} | sky_boundary, dilate={launch.dilate_px}px")

    agg = defaultdict(lambda: {
        "n_pixels": 0, "n_rd_hit": 0,
        "sum_err": 0.0, "n_err_samples": 0,
        "sum_rd_depth_at_hit": 0.0, "n_rd_depth_at_hit": 0,
    })

    for fid in frames:
        pkg = raytracing(fid, gs_list, lidar, bg, args)
        rd_depth = pkg["depth"].detach().squeeze(-1)
        rd_raydrop = pkg["raydrop"].detach().squeeze(-1)
        gt_depth = lidar.get_depth(fid).cuda()
        gt_mask = lidar.get_mask(fid).cuda()

        rd_hit = (rd_raydrop < launch.raydrop_ratio) & (rd_depth > 0)
        edge = compute_edge_band(gt_depth, gt_mask, launch.sobel_thr,
                                  launch.dilate_px)
        # Sky-side band: edge-band ∩ ~gt_mask (= the "no-return side near edge")
        # Note: our edge_band by construction includes ONLY gt_mask=True
        # pixels in the sobel half. The sky_boundary half also requires
        # gt_mask=True. So `edge & ~gt_mask` = 0 in the undilated version.
        # With dilate_px≥1, dilation spreads into ~gt_mask side too — this
        # IS the "edge_sky" region the user is hypothesising about.

        # 4 regions
        A_interior_valid = gt_mask & ~edge
        B_edge_valid     = gt_mask &  edge
        C_edge_sky       = ~gt_mask &  edge
        D_interior_sky   = ~gt_mask & ~edge

        gt_np = gt_depth.cpu().numpy()
        rd_np = rd_depth.cpu().numpy()
        rd_hit_np = rd_hit.cpu().numpy()
        regs = {
            "A_interior_valid": A_interior_valid.cpu().numpy(),
            "B_edge_valid"   : B_edge_valid.cpu().numpy(),
            "C_edge_sky"     : C_edge_sky.cpu().numpy(),
            "D_interior_sky" : D_interior_sky.cpu().numpy(),
        }
        for name, mask in regs.items():
            n = int(mask.sum())
            n_hit = int((mask & rd_hit_np).sum())
            agg[name]["n_pixels"] += n
            agg[name]["n_rd_hit"] += n_hit
            # For gt_mask=True regions, compute |gt - rd| where both valid
            if name.endswith("valid"):
                ok = mask & rd_hit_np
                if ok.any():
                    err = np.abs(gt_np[ok] - rd_np[ok])
                    agg[name]["sum_err"] += float(err.sum())
                    agg[name]["n_err_samples"] += int(ok.sum())
            # For ~gt_mask regions, the rd_depth value AT the phantom hit
            # tells us "how close to sensor the phantom landed"
            if name.endswith("sky"):
                ok = mask & rd_hit_np
                if ok.any():
                    agg[name]["sum_rd_depth_at_hit"] += float(rd_np[ok].sum())
                    agg[name]["n_rd_depth_at_hit"] += int(ok.sum())

        # Save a per-frame visualisation
        H, W = gt_np.shape
        region_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        region_rgb[A_interior_valid.cpu().numpy()] = (40, 40, 40)     # dark gray
        region_rgb[B_edge_valid.cpu().numpy()]     = (0, 200, 255)    # cyan (BGR)
        region_rgb[C_edge_sky.cpu().numpy()]       = (0, 0, 255)      # red
        region_rgb[D_interior_sky.cpu().numpy()]   = (60, 0, 0)       # dark blue

        # Overlay rendered hits as bright dots
        rd_hit_in_C = rd_hit_np & regs["C_edge_sky"]
        rd_hit_in_D = rd_hit_np & regs["D_interior_sky"]
        region_rgb[rd_hit_in_C] = (0, 255, 255)   # YELLOW = phantom on edge_sky
        region_rgb[rd_hit_in_D] = (255, 0, 255)   # MAGENTA = phantom on interior_sky

        # Stamp counts
        cv2.putText(region_rgb,
                     f"frame {fid}  C(edge_sky)={int(C_edge_sky.sum()):,}  "
                     f"phantom_in_C={int(rd_hit_in_C.sum()):,}  "
                     f"phantom_in_D={int(rd_hit_in_D.sum()):,}",
                     (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                     cv2.LINE_AA)
        cv2.imwrite(os.path.join(launch.out_dir, f"frame_{fid:04d}_regions.png"),
                    region_rgb)

    # --- summary ---
    total_pix = sum(agg[r]["n_pixels"] for r in agg)
    total_hit = sum(agg[r]["n_rd_hit"] for r in agg)
    print(f"\n# Summary over {len(frames)} frame(s)")
    print(f"# total pixels = {total_pix:,}  total rd_hit = {total_hit:,}")
    print()
    print(f"{'region':<22} {'pixels':>10} {'%area':>6} "
          f"{'rd_hit':>10} {'hit_rate':>9} {'MAE(m)':>8} {'mean rd_d':>10}")
    for name in ["A_interior_valid","B_edge_valid","C_edge_sky","D_interior_sky"]:
        a = agg[name]
        n = a["n_pixels"]
        hr = a["n_rd_hit"]/max(n,1)*100
        mae = a["sum_err"]/max(a["n_err_samples"],1) if a["n_err_samples"] else float("nan")
        mrd = (a["sum_rd_depth_at_hit"]/max(a["n_rd_depth_at_hit"],1)
               if a["n_rd_depth_at_hit"] else float("nan"))
        print(f"{name:<22} {n:>10,} {n/max(total_pix,1)*100:>5.1f}% "
              f"{a['n_rd_hit']:>10,} {hr:>8.1f}% "
              f"{mae:>8.3f} {mrd:>10.3f}")

    # Key derived ratios
    C, D = agg["C_edge_sky"], agg["D_interior_sky"]
    print()
    print("=== Phantom hit rate (= rd_hit on no-return) ===")
    print(f"  C_edge_sky    : {C['n_rd_hit']:>7,} / {C['n_pixels']:>7,} "
          f"= {C['n_rd_hit']/max(C['n_pixels'],1)*100:.2f}%")
    print(f"  D_interior_sky: {D['n_rd_hit']:>7,} / {D['n_pixels']:>7,} "
          f"= {D['n_rd_hit']/max(D['n_pixels'],1)*100:.2f}%")
    if C['n_pixels'] > 0 and D['n_pixels'] > 0:
        rate_C = C['n_rd_hit']/C['n_pixels']
        rate_D = D['n_rd_hit']/D['n_pixels']
        print(f"  density ratio C/D = {rate_C/max(rate_D, 1e-9):.1f}x  "
              f"(higher = more concentration at edge_sky)")
    A, B = agg["A_interior_valid"], agg["B_edge_valid"]
    mae_A = A["sum_err"]/max(A["n_err_samples"],1)
    mae_B = B["sum_err"]/max(B["n_err_samples"],1)
    print()
    print("=== Depth fit at valid pixels ===")
    print(f"  A_interior_valid MAE: {mae_A:.3f} m  ({A['n_err_samples']:,} samples)")
    print(f"  B_edge_valid     MAE: {mae_B:.3f} m  ({B['n_err_samples']:,} samples)")
    print(f"  ratio edge/interior: {mae_B/max(mae_A,1e-6):.1f}x")


if __name__ == "__main__":
    main()
