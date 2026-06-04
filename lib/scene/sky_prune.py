"""Shared sweep-and-prune logic for sky-mask hard pruning.

Used both as the final post-training pass invoked from train.py, and as the
standalone path in sky_prune_checkpoint.py. The in-training densify-time
prune lives in GaussianModel.densify_and_prune — this module is for the
deterministic full-view sweep applied to a settled model.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_sky_mask(gt_mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel > 1:
        gt_f = gt_mask.float().unsqueeze(0).unsqueeze(0)
        pad = kernel // 2
        dilated = F.max_pool2d(gt_f, kernel, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel, stride=1, padding=pad)
        gt_mask_closed = closed.squeeze(0).squeeze(0) > 0.5
        return ~gt_mask_closed
    return ~gt_mask


def sweep_and_sky_prune(
    gaussians_assets,
    train_lidars: Dict,
    args,
    ratio_threshold: float = 0.8,
    view_consistency: float = 0.8,
    min_views: int = 3,
    min_total_contrib: float = 1e-3,
    sky_morph_kernel: int | None = None,
    progress_desc: str = "Sky-prune sweep",
):
    """Sweep every training (sensor, frame) once and hard-prune background
    Gaussians whose sky-pixel contribution concentrates across enough views.

    Only assets with ``bounding_box is None`` (background) are eligible —
    object Gaussians live inside tracking boxes and the sky mask is not a
    valid criterion for them.

    Mutates ``gaussians_assets`` in place via ``prune_points``. Returns a
    dict of per-asset stats (Gaussians before, observed, pruned).
    """
    # Local import to avoid a hard dep at module load (raytracing pulls
    # CUDA/OptiX which is heavy and not always desired by importers).
    from lib.gaussian_renderer import raytracing

    if sky_morph_kernel is None:
        sky_morph_kernel = int(getattr(args.opt, "sky_morph_kernel", 3))

    bg_indices = [i for i, gs in enumerate(gaussians_assets)
                  if gs.bounding_box is None]
    if not bg_indices:
        return {"bg_indices": [], "pruned": 0, "per_asset": {}}

    sky_view_counts = {
        i: torch.zeros(gaussians_assets[i].get_local_xyz.shape[0], device="cuda")
        for i in bg_indices
    }
    view_counts = {
        i: torch.zeros(gaussians_assets[i].get_local_xyz.shape[0], device="cuda")
        for i in bg_indices
    }

    background = torch.tensor([0, 0, 1], device="cuda").float()

    pairs = [
        (sensor_name, frame_id)
        for sensor_name, lidar in train_lidars.items()
        for frame_id in lidar.train_frames
    ]

    with torch.no_grad():
        for sensor_name, frame_id in tqdm(pairs, desc=progress_desc):
            lidar = train_lidars[sensor_name]
            gt_mask = lidar.get_mask(frame_id).cuda()
            sky_mask = compute_sky_mask(gt_mask, sky_morph_kernel)
            pixel_weight = sky_mask.float()

            render_pkg = raytracing(
                frame_id, gaussians_assets, lidar, background, args,
                pixel_weight=pixel_weight,
            )
            acc_wet = render_pkg["accum_gaussian_weight"].view(-1)
            acc_sky_wet = render_pkg["accum_gaussian_sky_weight"].view(-1)

            begin = 0
            for i, gs in enumerate(gaussians_assets):
                n = gs.get_local_xyz.shape[0]
                if i in sky_view_counts:
                    tw = acc_wet[begin:begin + n]
                    sw = acc_sky_wet[begin:begin + n]
                    observed = tw > min_total_contrib
                    if observed.any():
                        ratio = sw / tw.clamp_min(1e-12)
                        sky_judged = observed & (ratio > ratio_threshold)
                        view_counts[i][observed] += 1
                        sky_view_counts[i][sky_judged] += 1
                begin += n

    per_asset = {}
    total_pruned = 0
    for i in bg_indices:
        gs = gaussians_assets[i]
        vc = view_counts[i]
        svc = sky_view_counts[i]
        n_before = gs.get_local_xyz.shape[0]
        enough_views = vc >= min_views
        sky_ratio = svc / vc.clamp_min(1.0)
        mostly_sky = sky_ratio > view_consistency
        prune_mask = enough_views & mostly_sky
        n_prune = int(prune_mask.sum().item())
        observed_any = int((vc > 0).sum().item())
        observed_min = int(enough_views.sum().item())
        per_asset[i] = dict(
            n_before=n_before,
            observed_any=observed_any,
            observed_min=observed_min,
            n_prune=n_prune,
        )
        if 0 < n_prune < n_before:
            gs.prune_points(prune_mask)
            total_pruned += n_prune

    return {"bg_indices": bg_indices, "pruned": total_pruned, "per_asset": per_asset}
