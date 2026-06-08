#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
import os
from simple_knn._C import distCUDA2
from lib.scene.bounding_box import BoundingBox
from lib.utils.console_utils import *
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from lib.utils.general_utils import generate_random_quaternion_with_fixed_normal
from lib.utils.sh_utils import RGB2SH

class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, dimension, sh_degree : int, extent : int = 200, bounding_box : BoundingBox= None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # Sky-mask hard-prune stats. sky_view_count counts views where this
        # Gaussian's contribution was concentrated in sky pixels; view_count
        # counts views where it was observed contributing meaningfully. Both
        # accumulate between densify_and_prune calls and are reset after a
        # prune cycle. See add_sky_stats / prune_sky_mask in densify_and_prune.
        self.sky_view_count = torch.empty(0)
        self.view_count = torch.empty(0)
        # Front-side hard-prune stats. front_view_count counts views where
        # this Gaussian's contribution was concentrated in front of the GT
        # target_depth (= phantom in front of the real surface). Shares
        # view_count above as the denominator. See add_front_stats /
        # the front_prune block in densify_and_prune.
        self.front_view_count = torch.empty(0)
        self.optimizer = None
        self.densify_scale_threshold = 0
        self.spatial_lr_scale = 0

        self.bounding_box = bounding_box
        self.extent = extent

        self.dimension = dimension # 3dgs/2dgs

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        try:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        except:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        print("Number of points of restored gaussian: ", self.get_local_xyz.shape[0])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)

    #@property
    def get_rotation(self, timestamp = 0.0):
        # if self.bounding_box is not None and self.bounding_box.frame:
        #     rot_in_local = torch.nn.functional.normalize(self._rotation, dim=1)
        #     quaternion = quaternion_raw_multiply(timer, rot_in_local, self.bounding_box.frame[timestamp][1])
        #     return quaternion
        # else:
        #     return self.rotation_activation(self._rotation)
        if self.bounding_box is not None and timestamp in self.bounding_box.frame:
            obj_rot = self.bounding_box.frame[timestamp][1]
        else:
            obj_rot = torch.zeros((1, 4), device="cuda")
        return obj_rot, self.rotation_activation(self._rotation)

    def get_world_xyz(self, timestamp = 0.0):
        if self.bounding_box is not None and timestamp in self.bounding_box.frame:
            R = build_rotation(self.bounding_box.frame[timestamp][1]).squeeze(0)
            return self._xyz @ R.T + self.bounding_box.frame[timestamp][0]
        else:
            return self._xyz

    @property
    def get_local_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, use_normals=False):
        self.spatial_lr_scale = self.extent
        fused_point_cloud = pcd.points.float().cuda()

        color_intensity = pcd.color_intensity.float().cuda()
        fused_color = RGB2SH(color_intensity)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialization: ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, self.dimension)
        
        if use_normals:
            normals = pcd.normals.float().cuda()
            rots = generate_random_quaternion_with_fixed_normal(normals)
        else:
            rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_local_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.densify_scale_threshold = training_args.densify_scale_threshold
        self.densify_weight_threshold = training_args.densify_weight_threshold
        self.xyz_gradient_accum = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        n = self.get_local_xyz.shape[0]
        self.sky_view_count = torch.zeros((n, 1), device="cuda")
        self.view_count = torch.zeros((n, 1), device="cuda")
        self.front_view_count = torch.zeros((n, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.sky_view_count.numel() > 0:
            self.sky_view_count = self.sky_view_count[valid_points_mask]
            self.view_count = self.view_count[valid_points_mask]
        if self.front_view_count.numel() > 0:
            self.front_view_count = self.front_view_count[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_local_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_local_xyz.shape[0]), device="cuda")
        # Append zeros for the new Gaussians so the existing entries' sky stats
        # survive the densify step — densify_and_prune snapshots these before
        # clone/split and uses the snapshot to compute the sky-prune mask.
        num_new = new_xyz.shape[0]
        if self.sky_view_count.numel() > 0 and num_new > 0:
            self.sky_view_count = torch.cat(
                [self.sky_view_count, torch.zeros((num_new, 1), device="cuda")], dim=0
            )
            self.view_count = torch.cat(
                [self.view_count, torch.zeros((num_new, 1), device="cuda")], dim=0
            )
        if self.front_view_count.numel() > 0 and num_new > 0:
            self.front_view_count = torch.cat(
                [self.front_view_count, torch.zeros((num_new, 1), device="cuda")], dim=0
            )

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self.get_local_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded = torch.zeros((n_init_points), device="cuda")
        padded[:grads.shape[0]] = grads
        grad_mask = torch.where(padded >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grad_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.densify_scale_threshold*self.extent)
        num = selected_pts_mask.sum().item()

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        if self.dimension == 2:
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_local_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return num

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        grad_mask = torch.where(grads >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grad_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.densify_scale_threshold*self.extent)
        num = selected_pts_mask.sum().item()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return num

    def densify_and_prune(self, opt, min_opacity, max_screen_size,
                          sensor_centers=None, min_range_prune=0.0,
                          skip_densify=False,
                          sky_prune_enabled=False,
                          sky_view_consistency_threshold=0.8,
                          sky_prune_min_views=3,
                          aniso_prune_enabled=False,
                          max_aniso_prune=10.0,
                          aniso_prune_max_opacity=0.5,
                          front_prune_enabled=False,
                          front_view_consistency_threshold=0.8,
                          front_prune_min_views=3,
                          occupancy_grid=None,
                          occupancy_prune_enabled=False,
                          occupancy_prune_opacity_threshold=0.5,
                          dead_prune_enabled=False,
                          dead_prune_min_views=1,
                          ego_prune_enabled=False,
                          ego_w2e=None,
                          ego_bboxes=None):
        # When skip_densify is True, we run only the pruning steps below
        # (low-opacity, bbox-escape, min_range_prune, big-points). This lets
        # us keep cleaning up after an asset has hit its point cap, instead
        # of freezing both densification AND pruning the moment we cross it.

        # --- Dead-Gaussian hard prune (runs FIRST, before clone/split). ---
        # A Gaussian whose view_count stayed below the threshold over an
        # entire densification cycle (~100 iter) was never crossed by any
        # training ray with meaningful contribution — invisible from every
        # viewpoint. Analysis of trained ckpts showed this class accounts
        # for 60% (Waymo) to 78% (T4) of bg, with α drifted to 1.0 and no
        # other prune mechanism able to catch them. Running this BEFORE
        # clone/split avoids confusing fresh spawns (view_count=0 by
        # construction) with truly-dead Gaussians. After the prune, the
        # rest of the densify/prune flow proceeds on the survivors.
        prune_dead_num = 0
        if (dead_prune_enabled and self.view_count.numel() > 0
                and self.get_local_xyz.shape[0] > 0):
            vc = self.view_count.squeeze(-1)
            dead_mask = vc < dead_prune_min_views
            n_dead = int(dead_mask.sum().item())
            n_pre = self.get_local_xyz.shape[0]
            if 0 < n_dead < n_pre:
                self.prune_points(dead_mask)
                prune_dead_num = n_dead
                print(f'Hard prune dead Gaussians (view_count<{dead_prune_min_views}): '
                      f'{n_dead}/{n_pre}')

        # --- Ego-swept-volume hard prune (bg only, OBB). ---
        # The ego vehicle physically occupied each pose's volume — no static
        # scene geometry can exist there. For each cached ego pose (base +
        # interpolated), transform bg Gaussian world positions into the ego
        # frame and check against the configured ego bboxes (main body +
        # mirrors). Pruned if inside ANY pose × ANY bbox.
        #
        # ego_w2e:    (M, 3, 4) — world→ego rigid transform per pose
        #                          (R^T | -R^T t) of ego2world. M ≈ 250 with
        #                          5-interp/segment over 50 frames.
        # ego_bboxes: (K, 2, 3) — K bboxes, each [min_xyz, max_xyz] in ego frame.
        #                          K=2 in T4 default (vehicle body + side mirrors).
        prune_ego_num = 0
        if (ego_prune_enabled and ego_w2e is not None and ego_bboxes is not None
                and ego_w2e.numel() > 0 and ego_bboxes.numel() > 0
                and self.get_local_xyz.shape[0] > 0):
            bg_xyz = self.get_world_xyz()
            n_pre = bg_xyz.shape[0]
            w2e = ego_w2e.to(bg_xyz.device, dtype=bg_xyz.dtype)   # (M, 3, 4)
            bb = ego_bboxes.to(bg_xyz.device, dtype=bg_xyz.dtype)  # (K, 2, 3)
            R = w2e[:, :, :3]       # (M, 3, 3)
            t = w2e[:, :, 3]        # (M, 3)
            bb_min = bb[:, 0, :]    # (K, 3)
            bb_max = bb[:, 1, :]    # (K, 3)
            # Chunk over bg points so the (M, n, 3) intermediate stays bounded.
            # 50K × 250 × 3 × 4 B = 150 MB. K=2 keeps the (K, M, n) bool small.
            chunk = 50_000
            inside_any = torch.zeros(n_pre, dtype=torch.bool, device=bg_xyz.device)
            for s in range(0, n_pre, chunk):
                e = min(s + chunk, n_pre)
                p_world = bg_xyz[s:e]                       # (n, 3)
                # p_ego[m, n] = R[m] @ p_world[n] + t[m]
                p_ego = torch.einsum('mij,nj->mni', R, p_world) + t[:, None, :]
                # (K, M, n) inside-flag, then OR over (K, M) → (n,)
                in_kmn = (
                    (p_ego.unsqueeze(0) >= bb_min[:, None, None, :])
                    & (p_ego.unsqueeze(0) <= bb_max[:, None, None, :])
                ).all(dim=-1)
                inside_any[s:e] = in_kmn.any(dim=0).any(dim=0)
            n_ego = int(inside_any.sum().item())
            if 0 < n_ego < n_pre:
                self.prune_points(inside_any)
                prune_ego_num = n_ego
                print(f'Hard prune ego-swept (OBB, {bb.shape[0]} bbox × '
                      f'{w2e.shape[0]} poses): {n_ego}/{n_pre}')

        mean_grads = (self.xyz_gradient_accum / self.denom).nan_to_num(0.0).squeeze(-1)

        if skip_densify:
            clone_num = 0
            split_num = 0
        else:
            clone_num = self.densify_and_clone(mean_grads, opt.densify_grad_threshold)
            split_num = self.densify_and_split(mean_grads, opt.densify_grad_threshold)
        print(f"clone_num: {clone_num}, split_num: {split_num}")

        low_opacity = (self.get_opacity < opt.thresh_opa_prune).squeeze()
        prune_mask = low_opacity
        prune_opacity_num = low_opacity.sum().item()
        prune_scale_num = 0
        prune_aniso_num = 0
        prune_front_num = 0
        prune_occ_num = 0

        # Anisotropy hard prune (bg only). T4's Hesai OT128 packs beams near
        # the horizon, and Tokyo urban scenes have building edges crossing
        # the horizon line in every direction. The combination produces a
        # ring of edge-tracking phantoms at sensor height (z≈2.18 m) that
        # share a signature: extremely elongated 2D surfels (σ_max/σ_min ≫ 1)
        # with moderate opacity. Catch them at densify time so the BVH and
        # the depth/intensity losses don't keep dragging more in around the
        # same edge each cycle. Object Gaussians stay exempt — vehicles
        # legitimately are elongated and the bbox-escape check guards them.
        if (aniso_prune_enabled and self.bounding_box is None
                and self.get_local_xyz.shape[0] > 0):
            sigma = self.get_scaling
            sig_min = sigma.min(dim=-1).values
            sig_max = sigma.max(dim=-1).values
            aniso = sig_max / sig_min.clamp_min(1e-12)
            opa = self.get_opacity.squeeze(-1)
            aniso_phantom = (aniso > max_aniso_prune) & (opa < aniso_prune_max_opacity)
            prune_aniso_num = int(aniso_phantom.sum().item())
            if prune_aniso_num > 0:
                prune_mask = torch.logical_or(prune_mask, aniso_phantom)
                print(f'Hard prune aniso>{max_aniso_prune} & opa<{aniso_prune_max_opacity}: '
                      f'{prune_aniso_num}')

        # Hard prune: object Gaussians whose center has escaped the bbox.
        # Runs every densification step (not gated by max_screen_size) so
        # bbox-escaping Gaussians cannot accumulate during early training.
        if self.bounding_box is not None:
            center_outside = torch.logical_or(
                (self.get_local_xyz < self.bounding_box.min_xyz).any(dim=-1),
                (self.get_local_xyz > self.bounding_box.max_xyz).any(dim=-1),
            )
            prune_mask = torch.logical_or(prune_mask, center_outside)
            print(f'Hard prune centers outside bbox: {center_outside.sum().item()}')

        # Hard prune background Gaussians within min_range_prune of any sensor.
        # Below the LiDAR's minimum measurable range no real returns exist, so
        # these are necessarily phantom Gaussians (object Gaussians are handled
        # via the bbox check above and skipped here).
        #
        # torch.cdist uses the expansion |a|^2 + |b|^2 - 2*a*b, which suffers
        # catastrophic cancellation when both vectors live near a far-from-origin
        # frame (UTM coords are ~1e5, |a|^2 ~ 1e10, so float32's ~7-digit
        # precision leaves ~hundred-meter noise on the squared distance). The
        # noise made every close-range Gaussian look like it lived inside the
        # min_range_prune sphere of some sensor — exactly the "donut hole"
        # around the trajectory we were seeing. Shift the origin to the sensor
        # centroid before cdist; differences stay in the < ~10^3 m range and
        # the cancellation goes away.
        if (self.bounding_box is None and sensor_centers is not None
                and min_range_prune > 0 and self.get_local_xyz.shape[0] > 0):
            sc = sensor_centers.to(self.get_local_xyz.device)
            origin = sc.mean(dim=0, keepdim=True)
            dists = torch.cdist(
                (self.get_local_xyz - origin).float(),
                (sc - origin).float(),
            )
            min_dist = dists.min(dim=1).values
            too_close = min_dist < min_range_prune
            prune_mask = torch.logical_or(prune_mask, too_close)
            print(f'Hard prune too-close-to-sensor (<{min_range_prune}m): {too_close.sum().item()}')

        if max_screen_size:

            #big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * self.extent * opt.prune_size_threshold
            prune_scale_num = big_points_ws.sum().item()
            #prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            print(f'Prune big points in world: {prune_scale_num} lower than min opacity: {prune_opacity_num}')

            if self.bounding_box is not None:
                # Refer to Street Gaussian by Yan et al. in 2024.
                # Prune points outside the tracking box
                repeat_num = 2
                stds = self.get_scaling
                if self.dimension == 2:
                    stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
                stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1]
                means = torch.zeros_like(self.get_local_xyz)
                means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
                samples = torch.normal(mean=means, std=stds) # [N, M, 3]
                rots = build_rotation(self._rotation) # [N, 3, 3]
                rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
                origins = self.get_local_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]

                samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]
                num_gaussians = self.get_local_xyz.shape[0]
                if num_gaussians > 0:
                    points_inside_box = torch.logical_and(
                        torch.all((samples_xyz >= self.bounding_box.min_xyz).view(num_gaussians, -1), dim=-1),
                        torch.all((samples_xyz <= self.bounding_box.max_xyz).view(num_gaussians, -1), dim=-1),
                    )
                    points_outside_box = torch.logical_not(points_inside_box)
                    prune_mask = torch.logical_or(prune_mask, points_outside_box)

                    print(f'Prune points outside bbox: {points_outside_box.sum()}')
    
        # Sky-mask hard prune.
        # Gaussians whose ratio of (#views judged sky)/(#views observed) exceeds
        # sky_view_consistency_threshold, after being observed in at least
        # sky_prune_min_views views, are removed outright. This is the safety
        # net for sky phantoms that the depth/free-space losses fail to thin
        # out — those losses only push opacity down, and a Gaussian with
        # opacity still above thresh_opa_prune survives indefinitely. The
        # multi-view consistency requirement guards against sky-mask false
        # positives (boundary pixels, thin structures misclassified as sky).
        prune_sky_num = 0
        if (sky_prune_enabled and self.view_count.numel() > 0
                and self.get_local_xyz.shape[0] > 0):
            vc = self.view_count.squeeze(-1)
            svc = self.sky_view_count.squeeze(-1)
            enough_views = vc >= sky_prune_min_views
            sky_ratio = svc / vc.clamp_min(1.0)
            mostly_sky = sky_ratio > sky_view_consistency_threshold
            sky_prune_mask = enough_views & mostly_sky
            prune_sky_num = sky_prune_mask.sum().item()
            if prune_sky_num > 0:
                prune_mask = torch.logical_or(prune_mask, sky_prune_mask)
                print(f'Hard prune sky-mask phantoms: {prune_sky_num}')

        # Front-side hard prune (mirror of sky_prune for hit rays). A bg
        # Gaussian whose contribution sits *in front of* the LiDAR's real GT
        # return across many views is a horizon-ring / near-range phantom
        # that the front_acc loss can't eliminate fast enough (loss only
        # pushes opacity down; opacity_reset resets it every 3K iter). The
        # multi-view consistency requirement (>=min_views observations,
        # >threshold fraction front-judged) guards against false positives
        # at viewpoints where occlusion legitimately puts surfels in front.
        if (front_prune_enabled and self.view_count.numel() > 0
                and self.front_view_count.numel() > 0
                and self.get_local_xyz.shape[0] > 0):
            vc = self.view_count.squeeze(-1)
            fvc = self.front_view_count.squeeze(-1)
            enough_views = vc >= front_prune_min_views
            front_ratio = fvc / vc.clamp_min(1.0)
            mostly_front = front_ratio > front_view_consistency_threshold
            front_prune_mask = enough_views & mostly_front
            prune_front_num = front_prune_mask.sum().item()
            if prune_front_num > 0:
                prune_mask = torch.logical_or(prune_mask, front_prune_mask)
                print(f'Hard prune front-of-GT phantoms: {prune_front_num}')

        # Occupancy hard prune (bg only). The voxel-based loss_occupancy is a
        # soft per-Gaussian-mean push that hovers at equilibrium even when
        # actual phantoms are removed (mean stays high because pruning catches
        # only specific subsets — sky_prune / front_prune / opacity_prune all
        # miss out-of-LiDAR-FOV Gaussians that drift to high opacity unchecked).
        # This block closes the gap: any bg Gaussian whose world-frame centre
        # lands in a free voxel AND whose opacity exceeds threshold is
        # definitionally a phantom (LiDAR never returned there yet it's
        # rendering visibly), hard-prune outright. Low-opacity free-voxel
        # Gaussians are left for opacity_prune to handle — they may still
        # represent borderline learning candidates.
        if (occupancy_prune_enabled and self.bounding_box is None
                and occupancy_grid is not None
                and self.get_local_xyz.shape[0] > 0):
            world_xyz = self.get_world_xyz()  # bg → just _xyz
            is_free = occupancy_grid.free_mask(world_xyz)
            opa = self.get_opacity.squeeze(-1)
            high_opa_in_free = is_free & (opa > occupancy_prune_opacity_threshold)
            prune_occ_num = int(high_opa_in_free.sum().item())
            if prune_occ_num > 0:
                prune_mask = torch.logical_or(prune_mask, high_opa_in_free)
                print(f'Hard prune occupancy phantoms '
                      f'(free voxel & opa>{occupancy_prune_opacity_threshold}): '
                      f'{prune_occ_num}')

        if prune_mask.sum() < self.get_local_xyz.shape[0]:
            self.prune_points(prune_mask)

        # Reset sky/front stats so the next densify cycle starts from a clean
        # slate. We deliberately reset AFTER prune_points so the filter doesn't
        # bypass the multi-view judgement above. view_count is shared as the
        # denominator for both sky and front stats.
        if self.view_count.numel() > 0:
            self.sky_view_count.zero_()
            self.view_count.zero_()
        if self.front_view_count.numel() > 0:
            self.front_view_count.zero_()

        torch.cuda.empty_cache()
        return (clone_num, split_num, prune_scale_num, prune_opacity_num,
                prune_sky_num, prune_aniso_num, prune_front_num, prune_occ_num,
                prune_dead_num, prune_ego_num)

    def add_densification_stats(self, mean_grads, update_filter):
        self.xyz_gradient_accum += torch.norm(mean_grads, dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_view_count(self, total_weight, min_total_contrib=1e-3):
        # Bump view_count for Gaussians observed (= contributed above
        # min_total_contrib) in this view. view_count is the shared
        # denominator for both sky_prune and front_prune ratios, so it must
        # be bumped exactly once per view regardless of which sub-stats are
        # being collected.
        if self.view_count.numel() == 0:
            return
        tw = total_weight.view(-1)
        observed = tw > min_total_contrib
        if observed.any():
            self.view_count[observed, 0] += 1

    def add_sky_stats(self, total_weight, sky_weight,
                      min_total_contrib=1e-3, sky_ratio_threshold=0.8):
        # Per-view sky judgement. Bumps sky_view_count only. Caller must
        # bump view_count via add_view_count separately (so sky + front can
        # share the denominator without double-counting).
        if self.view_count.numel() == 0:
            return
        tw = total_weight.view(-1)
        sw = sky_weight.view(-1)
        observed = tw > min_total_contrib
        if not observed.any():
            return
        ratio = sw / tw.clamp_min(1e-12)
        sky_judged = observed & (ratio > sky_ratio_threshold)
        self.sky_view_count[sky_judged, 0] += 1

    def add_front_stats(self, total_weight, front_weight,
                        min_total_contrib=1e-3, front_ratio_threshold=0.8):
        # Per-view front-side judgement (mirror of add_sky_stats). Bumps
        # front_view_count only; caller bumps view_count via add_view_count.
        if self.front_view_count.numel() == 0:
            return
        tw = total_weight.view(-1)
        fw = front_weight.view(-1)
        observed = tw > min_total_contrib
        if not observed.any():
            return
        ratio = fw / tw.clamp_min(1e-12)
        front_judged = observed & (ratio > front_ratio_threshold)
        self.front_view_count[front_judged, 0] += 1

    # Refer to Street Gaussian by Yan et al. in 2024.
    def box_reg_loss(self):
        reg_loss = 0
        if self.bounding_box is not None:
            box_loss_1 = torch.clamp_min(self.get_local_xyz - self.bounding_box.max_xyz, min=0.).mean()
            box_loss_2 = torch.clamp_min(self.bounding_box.min_xyz - self.get_local_xyz, min=0.).mean()
            box_loss = (box_loss_1 + box_loss_2) / self.extent
            scale_loss = (self.get_scaling.max(dim=1).values / self.extent).mean()
            reg_loss = box_loss * 100 + scale_loss

        return reg_loss