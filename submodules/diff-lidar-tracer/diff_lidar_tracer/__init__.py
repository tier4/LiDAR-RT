import sysconfig
import torch
import torch.nn as nn
from typing import NamedTuple

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class _Tracer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                optix_context,
                training,
                ray_o,
                ray_d,
                vertices,
                means3D,
                grads3D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                tracer_settings,
                pixel_weight,
                target_depth,
                contributor_alpha_threshold,
                contributor_alpha_sharpness,
                enable_n_contributors,
                ):

        # Restructure arguments the way that the C++ lib expects them
        args = (optix_context,
                training,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug,
                pixel_weight,
                target_depth,
                float(contributor_alpha_threshold),
                float(contributor_alpha_sharpness),
                bool(enable_n_contributors))

        # Invoke C++/CUDA/OptiX tracer
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                out_attr_float32, out_attr_uint32, accum_gaussian_weights, accum_gaussian_sky_weights, accum_at_target, accum_gaussian_front_weights, n_contributors_soft = _C.trace_surfels(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            out_attr_float32, out_attr_uint32, accum_gaussian_weights, accum_gaussian_sky_weights, accum_at_target, accum_gaussian_front_weights, n_contributors_soft = _C.trace_surfels(*args)

        # Keep relevant tensors for backward (incl. target_depth + accum_at_target
        # so the backward kernel can re-trace and reproduce the W_target running
        # sum and gradient term).
        ctx.tracer_settings = tracer_settings
        ctx.optix_context = optix_context
        ctx.contributor_alpha_threshold = float(contributor_alpha_threshold)
        ctx.contributor_alpha_sharpness = float(contributor_alpha_sharpness)
        ctx.save_for_backward(ray_o, ray_d, vertices, means3D, shs, colors_precomp, opacities, scales, rotations, cov3Ds_precomp,
                              out_attr_float32, out_attr_uint32, target_depth, accum_at_target, n_contributors_soft)

        # Return the per-Gaussian hit counter for training gradient filtering.
        # accum_gaussian_sky_weights is zero unless pixel_weight was a non-empty
        # tensor — used by the sky hard-prune path. accum_at_target is zero
        # unless target_depth was a non-empty tensor — used by the front-side
        # accumulation loss to penalise alpha contributions in front of the
        # real LiDAR hit. accum_gaussian_front_weights is zero unless
        # target_depth was supplied — per-Gaussian summed alpha*T limited to
        # hits before target_depth, drives the multi-view front-side hard
        # prune (mirror of accum_gaussian_sky_weights for sky_prune).
        return out_attr_float32, accum_gaussian_weights, accum_gaussian_sky_weights, accum_at_target, accum_gaussian_front_weights, n_contributors_soft

    @staticmethod
    def backward(ctx, grad_out_attr_float32, _, _sky, grad_accum_at_target, _front, grad_n_contributors_soft):

        # Restore necessary values from context
        tracer_settings = ctx.tracer_settings
        optix_context = ctx.optix_context
        contributor_alpha_threshold = ctx.contributor_alpha_threshold
        contributor_alpha_sharpness = ctx.contributor_alpha_sharpness
        (ray_o, ray_d, vertices, means3D, shs, colors_precomp, opacities, scales,
         rotations, cov3Ds_precomp, out_attr_float32, out_attr_uint32,
         target_depth, accum_at_target, n_contributors_soft) = ctx.saved_tensors
        # Restructure args as C++ method expects them
        args = (optix_context,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug,
                out_attr_float32,
                out_attr_uint32,
                grad_out_attr_float32,
                target_depth,
                accum_at_target,
                grad_accum_at_target,
                n_contributors_soft,
                grad_n_contributors_soft,
                contributor_alpha_threshold,
                contributor_alpha_sharpness)

        # Compute gradients for relevant tensors by invoking backward method
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means3D, grad_shs, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, grad_grads3D = _C.trace_surfels_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means3D, grad_shs, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, grad_grads3D = _C.trace_surfels_backward(*args)
        grads = (
            None,
            None,
            None,
            None,
            None,
            grad_means3D,
            grad_grads3D,
            grad_shs,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,  # pixel_weight (not differentiated)
            None,  # target_depth (not differentiated)
            None,  # contributor_alpha_threshold (scalar, not differentiated)
            None,  # contributor_alpha_sharpness (scalar, not differentiated)
            None,  # enable_n_contributors (bool, not differentiated)
        )

        return grads


class TracingSettings(NamedTuple):
    image_height: int  # no use, only for compatibility
    image_width: int  # no use, only for compatibility
    tanfovx: float  # no use, only for compatibility
    tanfovy: float  # no use, only for compatibility
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class Tracer(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

        # Find the OptiX shared library
        self.pkg_dir = sysconfig.get_path('purelib') + '/diff_lidar_tracer'

        # Create the OptiX context
        self.optix_context = _C.OptiXStateWrapper(self.pkg_dir)

    def build_acceleration_structure(self,
                                     vertices: torch.Tensor,
                                     triangles: torch.Tensor,
                                     rebuild: bool = 1,
                                     ):
        # Invoke C++/CUDA/OptiX acceleration structure building routine
        self.vertices = vertices
        return _C.build_acceleration_structure(self.optix_context, vertices, triangles, rebuild)

    def forward(self,
                ray_o: torch.Tensor,
                ray_d: torch.Tensor,
                mesh_normals: torch.Tensor,
                means3D: torch.Tensor,
                grads3D: torch.Tensor,
                shs: torch.Tensor = None,
                colors_precomp: torch.Tensor = None,
                opacities: torch.Tensor = None,
                scales: torch.Tensor = None,
                rotations: torch.Tensor = None,
                cov3Ds_precomp: torch.Tensor = None,
                tracer_settings: TracingSettings = None,
                pixel_weight: torch.Tensor = None,
                target_depth: torch.Tensor = None,
                contributor_alpha_threshold: float = 0.1,
                contributor_alpha_sharpness: float = 50.0,
                enable_n_contributors: bool = False,
                ):

        # Check if colors or SHs are provided
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        # Check if scales/rotations or cov3Ds_precomp is provided
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Create dummy color tracing input
        if shs is None: shs = torch.Tensor([]).cuda()
        if colors_precomp is None: colors_precomp = torch.Tensor([]).cuda()
        # Create dummy tracing transformation matrix input
        if scales is None: scales = torch.Tensor([]).cuda()
        if rotations is None: rotations = torch.Tensor([]).cuda()
        if cov3Ds_precomp is None: cov3Ds_precomp = torch.Tensor([]).cuda()
        # Empty pixel_weight => kernel falls back to no-op for sky weights.
        if pixel_weight is None: pixel_weight = torch.Tensor([]).cuda()
        # Empty target_depth => kernel falls back to no-op for the front-side
        # accumulation snapshot.
        if target_depth is None: target_depth = torch.Tensor([]).cuda()

        # Invoke the autograd function
        return _Tracer.apply(
            self.optix_context,
            self.training,
            ray_o,
            ray_d,
            self.vertices,
            means3D,
            grads3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            tracer_settings,
            pixel_weight,
            target_depth,
            contributor_alpha_threshold,
            contributor_alpha_sharpness,
            enable_n_contributors,
        )
