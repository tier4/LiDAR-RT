"""Post-training phantom hard-prune (standalone, scene-config-free).

Loads a checkpoint, applies opacity/anisotropy/height-band filters per
asset, and writes a pruned checkpoint back. Operates on the raw tensors
so it works even if the current data config no longer matches the one
used to train the ckpt (different frame_length, different dynamic
object set, etc.).

Filters:
  - opacity < --opacity         (default 0.02; bimodal gap floor)
  - sigma_min < --sigma-min     (default 0.005; needle-like)
  - sigma_max / sigma_min > --max-aniso (default 10; edge-tracking ellipsoid)
  - bg z-band phantom: bg Gaussians whose world-z falls within ±--z-band-hw
    of any --z-band-center (default 0.4, 2.18 — empirical bimodal peaks
    near road and sensor height) AND whose aniso > --z-band-aniso AND
    whose opacity < --z-band-opa. Opt in via --z-band-prune.

The bg z-distribution and per-band phantom breakdown are always printed
(diagnostic info, doesn't affect output unless --z-band-prune is set).

The optimizer state is dropped because the param-shape change would
otherwise mismatch the saved Adam state; the produced ckpt is intended
for inference / visualisation, not training resume.

Usage:
    python phantom_prune_checkpoint.py \
        -m output/.../models/ckpt_it_20000_good.pth \
        [--opacity 0.02] [--sigma-min 0.005] [--max-aniso 10] \
        [--z-band-prune] [--bg-only] [--dry-run]
"""
import argparse
import os

import numpy as np
import torch


def _maybe_slice(t, keep):
    """Slice tensor t along dim 0 by bool mask `keep` if shape matches."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == keep.shape[0]:
        return t[keep]
    return t


def _zband_phantom_mask(z, opa, aniso, centers, halfwidth, aniso_min, opa_max):
    """True where z is within ±halfwidth of any band center AND the Gaussian
    matches the in-band phantom signature (high aniso, low opacity)."""
    mask = torch.zeros_like(z, dtype=torch.bool)
    for c in centers:
        in_band = (z - c).abs() < halfwidth
        is_phantom = in_band & (aniso > aniso_min) & (opa < opa_max)
        mask |= is_phantom
    return mask


def _zband_diagnostic(p, centers, halfwidth, aniso_min, opa_max):
    """Print z-histogram + per-band breakdown for the bg asset and return
    a phantom mask. Heights interpreted in world frame (only meaningful
    for bg, where _xyz is world coordinates)."""
    _xyz = p[1].detach().to(torch.float32)
    _sc  = p[4].detach().to(torch.float32)
    _op  = p[6].detach().to(torch.float32)
    N = _xyz.shape[0]
    z = _xyz[:, 2]
    opa = torch.sigmoid(_op).view(-1)
    sigma = torch.exp(_sc)
    aniso = sigma.max(dim=-1).values / sigma.min(dim=-1).values.clamp_min(1e-12)

    z_np = z.cpu().numpy()
    z_lo = float(min(z_np.min(), -2.0))
    z_hi = float(max(z_np.max(), 6.0))
    z_bins = np.arange(np.floor(z_lo * 2) / 2, np.ceil(z_hi * 2) / 2 + 0.5, 0.5)
    counts, _ = np.histogram(z_np, bins=z_bins)
    cmax = max(int(counts.max()), 1)

    print('\n=== bg z-distribution (world frame, 0.5 m bins) ===')
    for i in range(len(counts)):
        pct = 100 * counts[i] / N
        bar = '#' * int(40 * counts[i] / cmax)
        marker = ''
        for c in centers:
            if z_bins[i] <= c < z_bins[i + 1]:
                marker = f'  ← band center {c:+.2f}'
                break
        print(f'  [{z_bins[i]:+5.2f}, {z_bins[i+1]:+5.2f}) m: '
              f'{counts[i]:>10,} ({pct:5.2f}%)  {bar}{marker}')

    print(f'\n=== Per-band phantom breakdown '
          f'(±{halfwidth:.2f} m, aniso>{aniso_min}, opa<{opa_max}) ===')
    union_mask = torch.zeros_like(z, dtype=torch.bool)
    for c in centers:
        in_band = (z - c).abs() < halfwidth
        is_phantom = in_band & (aniso > aniso_min) & (opa < opa_max)
        is_legit_flat = in_band & (aniso < 2.0)  # isotropic-ish, likely real
        union_mask |= is_phantom
        print(f'  z ∈ [{c - halfwidth:+.2f}, {c + halfwidth:+.2f}] m:')
        print(f'    total in band:                {int(in_band.sum()):>10,}')
        print(f'    aniso < 2 (likely legit):     {int(is_legit_flat.sum()):>10,}')
        print(f'    aniso > {aniso_min} & opa < {opa_max} (PHANTOM): '
              f'{int(is_phantom.sum()):>10,}  ← prune candidates')
    print(f'\n  Union of phantom bands:          {int(union_mask.sum()):>10,}  '
          f'({100 * union_mask.float().mean():.2f}% of bg)')

    return union_mask


def _prune_asset(p, opacity_thr, sigma_min_thr, max_aniso, drop_optimizer,
                 extra_bad=None):
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
    if extra_bad is not None:
        bad = bad | extra_bad
    keep = ~bad
    n_after = int(keep.sum().item())

    breakdown = dict(
        opa=int(bad_opa.sum().item()),
        smin=int(bad_smin.sum().item()),
        aniso=int(bad_aniso.sum().item()),
        zband=int(extra_bad.sum().item()) if extra_bad is not None else 0,
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
    # Height-band phantom controls. Always diagnoses; pruning is opt-in.
    parser.add_argument("--z-band-centers", dest="z_band_centers", type=float,
                        nargs="+", default=[0.4, 2.18],
                        help="World-z centers of phantom bands (default: 0.4 = "
                             "road-edge band, 2.18 = sensor-height ring)")
    parser.add_argument("--z-band-hw", dest="z_band_hw", type=float, default=0.15,
                        help="Halfwidth of each band in metres (default 0.15)")
    parser.add_argument("--z-band-aniso", dest="z_band_aniso", type=float, default=5.0,
                        help="Min aniso to qualify as phantom inside a band")
    parser.add_argument("--z-band-opa", dest="z_band_opa", type=float, default=0.5,
                        help="Max opacity to qualify as phantom inside a band")
    parser.add_argument("--z-band-prune", action="store_true",
                        help="Actually prune the z-band phantom mask in addition "
                             "to the opacity/aniso/sigma_min filters")
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

    # bg z-band diagnostic + optional phantom mask (asset[0] only — _xyz is
    # world frame there; for object assets it's bbox-local and z is meaningless).
    zband_mask = _zband_diagnostic(
        model_params[0],
        args.z_band_centers, args.z_band_hw,
        args.z_band_aniso, args.z_band_opa,
    )
    if not args.z_band_prune:
        print('  (diagnostic only — pass --z-band-prune to add these to the prune mask)')
        zband_mask = None

    print()
    print(f"{'asset':>6} {'N_before':>11} {'bad_opa':>10} {'bad_smin':>10} "
          f"{'bad_aniso':>10} {'bad_zband':>10} {'union':>10} {'after':>11} {'%pruned':>8}")
    print("-" * 100)

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
        # z-band mask only applies to bg (asset 0); for objects, _xyz is in
        # bbox-local frame so a world-z filter is meaningless.
        extra_bad = zband_mask if (i == 0 and zband_mask is not None) else None
        pruned_p, n_b, n_a, bd = _prune_asset(
            p, args.opacity, args.sigma_min, args.max_aniso,
            drop_optimizer=not args.keep_optimizer,
            extra_bad=extra_bad,
        )
        out_params.append(p if args.dry_run else pruned_p)
        total_after += n_a
        pct = 100.0 * (n_b - n_a) / max(n_b, 1)
        print(f"  [{i:>2}]  {n_b:>9,}  {bd['opa']:>10,} {bd['smin']:>10,} "
              f"{bd['aniso']:>10,} {bd['zband']:>10,} {bd['union']:>10,}  "
              f"{n_a:>9,}  {pct:>7.1f}%")

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
