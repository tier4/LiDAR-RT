"""Post-training phantom hard-prune (standalone, scene-config-free).

Loads a checkpoint, applies opacity/anisotropy filters per asset, and writes
a pruned checkpoint back. Operates on the raw tensors so it works even if
the current data config no longer matches the one used to train the ckpt
(different frame_length, different dynamic object set, etc.). Use the
flag combinations to target the bg distribution analysis findings:

  - opacity < --opacity         (default 0.02; bimodal gap floor)
  - sigma_min < --sigma-min     (default 0.005; needle-like)
  - sigma_max / sigma_min > --max-aniso (default 10; edge-tracking ellipsoid)

The optimizer state is dropped because the param-shape change would
otherwise mismatch the saved Adam state; the produced ckpt is intended
for inference / visualisation, not training resume.

Usage:
    python phantom_prune_checkpoint.py \
        -m output/.../models/ckpt_it_20000_good.pth \
        [--opacity 0.02] [--sigma-min 0.005] [--max-aniso 10] \
        [--bg-only] [--dry-run]
"""
import argparse
import os

import torch


def _maybe_slice(t, keep):
    """Slice tensor t along dim 0 by bool mask `keep` if shape matches."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == keep.shape[0]:
        return t[keep]
    return t


def _prune_asset(p, opacity_thr, sigma_min_thr, max_aniso, drop_optimizer):
    """Return (pruned_tuple, n_before, n_after, breakdown_dict)."""
    (sh_degree, _xyz, _features_dc, _features_rest, _scaling, _rotation,
     _opacity, max_radii2D, xyz_gradient_accum, denom, opt_state,
     spatial_lr_scale) = p

    n_before = _xyz.shape[0]
    if n_before == 0:
        return p, 0, 0, dict(opa=0, smin=0, aniso=0, union=0)

    opacity = torch.sigmoid(_opacity.detach()).view(-1)
    sigma = torch.exp(_scaling.detach())
    sig_min = sigma.min(dim=-1).values
    sig_max = sigma.max(dim=-1).values
    aniso = sig_max / sig_min.clamp_min(1e-12)

    bad_opa = opacity < opacity_thr
    bad_smin = sig_min < sigma_min_thr
    bad_aniso = aniso > max_aniso
    bad = bad_opa | bad_smin | bad_aniso
    keep = ~bad
    n_after = int(keep.sum().item())

    breakdown = dict(
        opa=int(bad_opa.sum().item()),
        smin=int(bad_smin.sum().item()),
        aniso=int(bad_aniso.sum().item()),
        union=n_before - n_after,
    )

    # If nothing to prune or asset would be emptied, return unchanged.
    if breakdown["union"] == 0 or n_after == 0:
        return p, n_before, n_before, breakdown

    pruned = (
        sh_degree,
        _xyz[keep],
        _features_dc[keep],
        _features_rest[keep],
        _scaling[keep],
        _rotation[keep],
        _opacity[keep],
        _maybe_slice(max_radii2D, keep),
        _maybe_slice(xyz_gradient_accum, keep),
        _maybe_slice(denom, keep),
        None if drop_optimizer else opt_state,
        spatial_lr_scale,
    )
    return pruned, n_before, n_after, breakdown


def main():
    parser = argparse.ArgumentParser(description="Hard-prune phantom Gaussians from a checkpoint")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Input checkpoint path (.pth)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path. Defaults to <input>_phantomprune.pth")
    parser.add_argument("--opacity", type=float, default=0.02,
                        help="Prune if post-sigmoid opacity < this value")
    parser.add_argument("--sigma-min", dest="sigma_min", type=float, default=0.005,
                        help="Prune if min(sigma) over axes < this value")
    parser.add_argument("--max-aniso", dest="max_aniso", type=float, default=10.0,
                        help="Prune if sigma_max/sigma_min > this value")
    parser.add_argument("--bg-only", action="store_true",
                        help="Only prune asset[0] (the bg); leave objects alone")
    parser.add_argument("--keep-optimizer", action="store_true",
                        help="Keep the saved optimizer state (will mismatch the "
                             "pruned param shapes; only safe if you don't resume)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be pruned without writing output")
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        base, ext = os.path.splitext(args.model)
        out_path = base + "_phantomprune" + ext

    model_params, ckpt_iter = torch.load(args.model, weights_only=False)
    print(f"[Loaded] iter={ckpt_iter}  n_assets={len(model_params)}  from {args.model}")
    print(f"[Params] opacity<{args.opacity}  sigma_min<{args.sigma_min}  "
          f"max_aniso>{args.max_aniso}  bg_only={args.bg_only}  "
          f"drop_optimizer={not args.keep_optimizer}")
    print()

    print(f"{'asset':>6} {'N_before':>11} {'bad_opa':>10} {'bad_smin':>10} "
          f"{'bad_aniso':>10} {'union':>10} {'after':>11} {'%pruned':>8}")
    print("-" * 90)

    out_params = []
    total_before = total_after = 0
    for i, p in enumerate(model_params):
        n_before = p[1].shape[0]
        total_before += n_before
        if args.bg_only and i > 0:
            out_params.append(p)
            total_after += n_before
            print(f"  [{i:>2}]  {n_before:>9,}   (skipped via --bg-only)")
            continue
        pruned_p, n_b, n_a, bd = _prune_asset(
            p, args.opacity, args.sigma_min, args.max_aniso,
            drop_optimizer=not args.keep_optimizer,
        )
        out_params.append(p if args.dry_run else pruned_p)
        total_after += n_a
        pct = 100.0 * (n_b - n_a) / max(n_b, 1)
        print(f"  [{i:>2}]  {n_b:>9,}  {bd['opa']:>10,} {bd['smin']:>10,} "
              f"{bd['aniso']:>10,} {bd['union']:>10,}  {n_a:>9,}  {pct:>7.1f}%")

    print("-" * 90)
    print(f"TOTAL before={total_before:,}  after={total_after:,}  "
          f"pruned={total_before-total_after:,} "
          f"({100*(total_before-total_after)/max(total_before,1):.1f}%)")

    if args.dry_run:
        print("\n[dry-run] no checkpoint written")
        return

    torch.save((out_params, ckpt_iter), out_path)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
