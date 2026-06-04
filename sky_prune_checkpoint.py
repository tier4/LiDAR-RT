"""
Post-training sky-mask hard prune (standalone).

Runs the deterministic full-view sweep over every training (sensor, frame),
accumulates per-Gaussian sky-vs-total contribution, then hard-prunes background
Gaussians whose sky-pixel concentration meets the multi-view consistency
criterion. Saves the pruned model as a new checkpoint alongside the input one.

The same sweep is invoked automatically at the end of train.py when
sky_prune_enabled is true — this script is for re-running the prune offline
against an existing checkpoint with different thresholds.

Usage:
    python sky_prune_checkpoint.py \
        -ec configs/t4/exp_t4.yaml \
        -dc configs/t4/dynamic/example.yaml \
        -m output/.../models/ckpt_it_30000_good.pth \
        [--ratio-threshold 0.8] [--view-consistency 0.8] [--min-views 3] \
        [--min-total-contrib 1e-3] [--sky-morph-kernel 5]
"""
import argparse
import os

import torch

from lib import dataloader
from lib.arguments import parse
from lib.scene.sky_prune import sweep_and_sky_prune


def main():
    parser = argparse.ArgumentParser(description="Post-training sky-mask hard prune")
    parser.add_argument("-ec", "--exp_config_path", type=str, required=True)
    parser.add_argument("-dc", "--data_config_path", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Input checkpoint path (.pth)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output checkpoint path. Defaults to <input>_skyprune.pth")
    # Defaults come from the config (args.opt.sky_prune_*) so a config edit
    # propagates without rewriting the CLI. Pass an explicit flag to override
    # the config for a one-off run.
    parser.add_argument("--ratio-threshold", type=float, default=None)
    parser.add_argument("--view-consistency", type=float, default=None)
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--min-total-contrib", type=float, default=None)
    parser.add_argument("--sky-morph-kernel", type=int, default=None,
                        help="Override sky_morph_kernel from config")
    launch_args = parser.parse_args()

    args = parse(launch_args.exp_config_path)
    args = parse(launch_args.data_config_path, args)
    args.model_path = launch_args.model

    ratio_threshold = (launch_args.ratio_threshold if launch_args.ratio_threshold is not None
                       else float(getattr(args.opt, "sky_prune_pixel_ratio_threshold", 0.8)))
    view_consistency = (launch_args.view_consistency if launch_args.view_consistency is not None
                        else float(getattr(args.opt, "sky_prune_view_consistency_threshold", 0.8)))
    min_views = (launch_args.min_views if launch_args.min_views is not None
                 else int(getattr(args.opt, "sky_prune_min_views", 3)))
    min_total_contrib = (launch_args.min_total_contrib if launch_args.min_total_contrib is not None
                         else float(getattr(args.opt, "sky_prune_min_total_contrib", 1e-3)))

    out_path = launch_args.output
    if out_path is None:
        base, ext = os.path.splitext(launch_args.model)
        out_path = base + "_skyprune" + ext

    scene = dataloader.load_scene(args.source_dir, args, test=False)
    gaussians_assets = scene.gaussians_assets

    model_params, ckpt_iter = torch.load(launch_args.model)
    scene.restore(model_params, args.opt)
    print(f"[Loaded] checkpoint iter={ckpt_iter} from {launch_args.model}")

    print(f"[Params] ratio>{ratio_threshold}  view_consistency>{view_consistency}  "
          f"min_views={min_views}  min_total_contrib={min_total_contrib}")
    stats = sweep_and_sky_prune(
        gaussians_assets,
        scene.train_lidars,
        args,
        ratio_threshold=ratio_threshold,
        view_consistency=view_consistency,
        min_views=min_views,
        min_total_contrib=min_total_contrib,
        sky_morph_kernel=launch_args.sky_morph_kernel,
    )

    for i, st in stats["per_asset"].items():
        print(f"  asset[{i}] (bg) Gaussians={st['n_before']}  "
              f"observed_any={st['observed_any']}  "
              f"observed>={min_views}={st['observed_min']}  "
              f"to_prune={st['n_prune']}")
    print(f"[Done] total hard-pruned: {stats['pruned']}")

    model_params_out = [gs.capture() for gs in gaussians_assets]
    torch.save((model_params_out, ckpt_iter), out_path)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
