"""Inspect a LiDAR-RT checkpoint for "useless" Gaussians.

Loads the checkpoint via the same fast path vis_rerun.py uses, then
reports the joint distribution of σ (scale) and α (opacity) for the bg
Gaussian model, plus per-object summaries. Aimed at answering "how many
Gaussians are too small / too transparent to contribute meaningfully?"

Usage:
    .venv/bin/python analyze_gaussian_stats.py \
        -ec configs/t4/exp_t4.yaml -dc configs/t4/dynamic/example.yaml \
        -s ~/.webauto/data/data/annotation_dataset/<UUID>/<VER> \
        -m output/t4_tuned/test/scene_t4d1/models/ckpt_it_21000_good.pth
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.arguments import parse
from vis_rerun import load_gaussians_fast, load_scene_fast


def fmt(n):
    return f"{n:,}"


def percentile(x, q):
    return float(np.percentile(x, q))


def report_gaussian(name, gs, eff_alpha_thr=0.05,
                    tiny_sigma_max_thr=0.01,
                    needle_aniso_thr=10.0):
    """Print one block of stats for a single GaussianModel."""
    sigma = torch.exp(gs._scaling.detach()).cpu().numpy()        # (N, 2) — 2D surfel
    opa = torch.sigmoid(gs._opacity.detach()).cpu().numpy().squeeze(-1)

    N = sigma.shape[0]
    sig_min = sigma.min(axis=1)
    sig_max = sigma.max(axis=1)
    aniso = sig_max / np.maximum(sig_min, 1e-12)

    print(f"\n=== {name}   N = {fmt(N)} ===")

    # σ distribution
    print(f"σ_min  (per-Gaussian min axis)  p1/p25/p50/p75/p99 = "
          f"{percentile(sig_min,1):.4f} / {percentile(sig_min,25):.4f} / "
          f"{percentile(sig_min,50):.4f} / {percentile(sig_min,75):.4f} / "
          f"{percentile(sig_min,99):.4f}   "
          f"min/max = {sig_min.min():.2e} / {sig_min.max():.2e}")
    print(f"σ_max  (per-Gaussian max axis)  p1/p25/p50/p75/p99 = "
          f"{percentile(sig_max,1):.4f} / {percentile(sig_max,25):.4f} / "
          f"{percentile(sig_max,50):.4f} / {percentile(sig_max,75):.4f} / "
          f"{percentile(sig_max,99):.4f}   "
          f"min/max = {sig_max.min():.2e} / {sig_max.max():.2e}")
    print(f"opacity                         p1/p25/p50/p75/p99 = "
          f"{percentile(opa,1):.4f} / {percentile(opa,25):.4f} / "
          f"{percentile(opa,50):.4f} / {percentile(opa,75):.4f} / "
          f"{percentile(opa,99):.4f}   "
          f"min/max = {opa.min():.4f} / {opa.max():.4f}")
    print(f"aniso  (σ_max / σ_min)          p1/p25/p50/p75/p99 = "
          f"{percentile(aniso,1):.2f} / {percentile(aniso,25):.2f} / "
          f"{percentile(aniso,50):.2f} / {percentile(aniso,75):.2f} / "
          f"{percentile(aniso,99):.2f}   max = {aniso.max():.2f}")

    # "Useless" categories
    cats = [
        ("σ_max <  1e-4   (hard-floor near min_scale=1e-6)",
         sig_max < 1e-4),
        ("σ_max <  1e-3   (sub-mm, invisible at LiDAR range)",
         sig_max < 1e-3),
        ("σ_max <  1e-2 = 1cm (below scale_reg_min)",
         sig_max < 1e-2),
        ("σ_max <  3cm    (smaller than typical voxel 15cm)",
         sig_max < 0.03),
        ("σ_max <  10cm",  sig_max < 0.10),
        (f"α      <  {eff_alpha_thr}    (below thresh_opa_prune=0.02)",
         opa < eff_alpha_thr),
        (f"α      <  0.10",  opa < 0.10),
        (f"aniso  > {needle_aniso_thr}  (needle-like)",
         aniso > needle_aniso_thr),
        (f"σ_max < {tiny_sigma_max_thr} AND α < {eff_alpha_thr}  (BOTH tiny + faint)",
         (sig_max < tiny_sigma_max_thr) & (opa < eff_alpha_thr)),
        (f"σ_max < {tiny_sigma_max_thr} OR  α < {eff_alpha_thr}  (EITHER tiny or faint)",
         (sig_max < tiny_sigma_max_thr) | (opa < eff_alpha_thr)),
    ]
    print(f"\n  {'category':56s}  {'count':>10s}  {'%':>6s}")
    for label, mask in cats:
        n = int(mask.sum())
        pct = n / max(N, 1) * 100
        print(f"  {label:56s}  {fmt(n):>10s}  {pct:>5.1f}%")

    return {
        "name": name, "N": N,
        "sig_min": sig_min, "sig_max": sig_max,
        "opa": opa, "aniso": aniso,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", required=True, dest="exp_config_path")
    p.add_argument("-dc", required=True, dest="data_config_path")
    p.add_argument("-s", default="", dest="source_dir")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--out-dir", default="output/gaussian_stats")
    p.add_argument("--no-plot", action="store_true")
    launch = p.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = ""
    args.eval_type = "all"
    args.rerun_save = ""
    if launch.source_dir:
        args.source_dir = launch.source_dir

    # Need lidars for num_gaussians (= 1 bg + N objects)
    lidars, bboxes = load_scene_fast(args)
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    num_g = len(ckpt[0])
    del ckpt
    gs_list, first_iter = load_gaussians_fast(args.model_path, num_g, args)

    print(f"\n# Checkpoint: {args.model_path}")
    print(f"# Iteration : {first_iter}")
    print(f"# Models    : 1 bg + {num_g - 1} objects = {num_g} GaussianModel")

    stats = []
    stats.append(report_gaussian("bg (gs[0])", gs_list[0]))

    # Aggregate object stats
    obj_sig_max, obj_opa, obj_aniso = [], [], []
    obj_total = 0
    for i, gs in enumerate(gs_list[1:], start=1):
        n = gs._xyz.shape[0]
        obj_total += n
        sigma = torch.exp(gs._scaling.detach()).cpu().numpy()
        opa = torch.sigmoid(gs._opacity.detach()).cpu().numpy().squeeze(-1)
        sig_max = sigma.max(axis=1)
        sig_min = sigma.min(axis=1)
        aniso = sig_max / np.maximum(sig_min, 1e-12)
        obj_sig_max.append(sig_max)
        obj_opa.append(opa)
        obj_aniso.append(aniso)
    print(f"\n=== objects (gs[1..{num_g-1}], pooled)   "
          f"N = {fmt(obj_total)} ===")
    if obj_total > 0:
        all_sig_max = np.concatenate(obj_sig_max)
        all_opa = np.concatenate(obj_opa)
        all_aniso = np.concatenate(obj_aniso)
        print(f"σ_max p50 = {np.percentile(all_sig_max, 50):.4f}   "
              f"α p50 = {np.percentile(all_opa, 50):.4f}   "
              f"aniso p50 = {np.percentile(all_aniso, 50):.2f}")
        tiny = (all_sig_max < 1e-2) | (all_opa < 0.05)
        print(f"  EITHER σ_max<1cm OR α<0.05: {fmt(int(tiny.sum()))} "
              f"({tiny.sum()/max(all_sig_max.size,1)*100:.1f}%)")

    # --- joint σ_max × α plot for bg ---
    if not launch.no_plot:
        import os
        os.makedirs(launch.out_dir, exist_ok=True)
        bg = stats[0]
        sig_max = bg["sig_max"]
        opa = bg["opa"]
        # Log σ for the heatmap so the dynamic range is readable
        sig_log = np.log10(np.maximum(sig_max, 1e-7))
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        h = ax[0].hist2d(sig_log, opa, bins=(80, 60),
                          cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
        ax[0].set_xlabel("log10(σ_max)  [m]")
        ax[0].set_ylabel("opacity (sigmoid)")
        ax[0].set_title(f"bg Gaussians joint distribution "
                         f"(N={fmt(bg['N'])})")
        fig.colorbar(h[3], ax=ax[0], label="count")
        for x, lab in [(-4, "1e-4"), (-3, "1e-3"), (-2, "1cm"),
                       (-1, "10cm"), (0, "1m")]:
            ax[0].axvline(x, color="white", lw=0.5, alpha=0.5)
            ax[0].text(x + 0.05, 0.95, lab, color="white", fontsize=8,
                       transform=ax[0].get_xaxis_transform())
        ax[0].axhline(0.05, color="red", lw=0.8, alpha=0.7)
        ax[0].text(sig_log.min() + 0.1, 0.06, "α=0.05 (low-impact)",
                   color="red", fontsize=8)

        # σ_max histogram alone (log x)
        ax[1].hist(np.maximum(sig_max, 1e-7), bins=np.logspace(-7, 1, 80),
                   color="C0", alpha=0.8)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("σ_max  [m]  (log)")
        ax[1].set_ylabel("count (log)")
        ax[1].set_title("bg σ_max histogram")
        for x, lab, c in [(0.01, "scale_reg_min", "orange"),
                          (1.0, "scale_reg_max", "green"),
                          (1.5, "oversized_split", "red")]:
            ax[1].axvline(x, color=c, lw=0.8, label=lab)
        ax[1].legend(loc="upper right")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out = f"{launch.out_dir}/bg_size_opacity_iter{first_iter}.png"
        fig.savefig(out, dpi=130)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
