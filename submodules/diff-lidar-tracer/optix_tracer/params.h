// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include "config.h"
#include "glm/glm.hpp"


// Define the data types

#ifndef UINT32_MAX
typedef unsigned int uint32_t;
#endif

#ifndef UINT64_MAX
typedef unsigned long int uint64_t;
#endif


// Define the data structure used in the OptiX kernel
struct RayGenData {};
struct HitGroupData {};
struct MissData {};


// Define the global data structure used in the OptiX kernel
struct Params
{
    // OptiX handler
    OptixTraversableHandle handle;

    // Global parameters
    bool training;  // training or testing

    // Input parameters
    int P, H, W, D, M;  // Gaussian number, height, width, SHs number
    float3* ray_o;  // (H, W, 3), ray origin
    float3* ray_d;  // (H, W, 3), ray direction
    float3* vertices;  // (P * 2, 3), primitive vertices
    float* background;  // (3), background color
    glm::vec3* means3D;  // (P, 3), center coordinates
    float* shs;  // (P, M), SHs
    float* colors_precomp;  // (P, C), precomputed parameters
    float* opacities;  // (P, 1), opacities
    glm::vec2* scales;  // (P, 2), scales
    float scale_modifier;
    glm::vec4* rotations;  // (P, 4), rotations
    float* transMat_precomp;
    float* viewmatrix;
    float* projmatrix;
    glm::vec3* campos;

    // Output forward results
    float* out_attr_float32;  // (H, W, C), RGB color or other features
    int* out_attr_uint32;
    float* accum_gaussian_weights;

    // Optional per-pixel weight (H, W) and per-Gaussian sky-weighted accumulator (P).
    // When pixel_weight is non-null, the kernel adds alpha*T*pixel_weight[tidx] to
    // accum_gaussian_sky_weights[gidx] for every contributing Gaussian — used by the
    // sky-mask hard-prune path to detect Gaussians whose rendering contribution
    // concentrates in sky pixels across many views.
    float* pixel_weight;
    float* accum_gaussian_sky_weights;

    // Optional per-pixel target_depth (H, W) and per-pixel "accumulated alpha
    // up to target_depth" output (H, W). When target_depth is non-null, the
    // kernel records W_target = sum over hits where dpt < target_depth[tidx]
    // of alpha*T. Used by the front-side accumulation loss: rays where GT
    // returned a hit at gt_depth should have W_target ≈ 0 because no surface
    // should exist in front of the real hit. Sky rays pass target_depth =
    // large sentinel so W_target == total W (subsumes the older freespace
    // loss). Pass nullptr for both to skip the snapshot path entirely.
    float* target_depth;          // (H, W)
    float* accum_at_target;       // (H, W)
    // Backward-side gradient (filled by autograd, read by backward.cu).
    float* dL_daccum_at_target;   // (H, W)

    // Per-Gaussian front-side accumulator (P,). Atomic-added in forward when
    // a hit's dpt is strictly less than the pixel's target_depth. Combined
    // with accum_gaussian_weights (total contribution regardless of depth),
    // the ratio accum_gaussian_front_weights / accum_gaussian_weights gives
    // "fraction of this Gaussian's contribution that landed in front of GT"
    // — the per-view signal for the multi-view front-side hard prune
    // (mirror of accum_gaussian_sky_weights for sky_prune).
    float* accum_gaussian_front_weights;

    // Soft per-pixel contributor count (H, W). At each hit, the kernel adds
    // sigmoid((alpha - contributor_alpha_threshold) * contributor_alpha_sharpness)
    // — so high-α hits contribute ≈1, low-α hits ≈0, smoothly differentiable.
    // Lets a Python loss penalise "edge pixels with many stacked thin
    // Gaussians" by L1 on (n_contributors - 1). Backward picks up
    // dL_dn_contributors and routes the sigmoid derivative back into α.
    float* n_contributors_soft;          // (H, W)
    float* dL_dn_contributors_soft;      // (H, W) backward upstream grad
    float  contributor_alpha_threshold;  // sigmoid midpoint (alpha value)
    float  contributor_alpha_sharpness;  // larger → step-like

    // Input upstream gradients
    float* dL_dout_attr_float32;  // (H, W, C), gradient of RGB color or other features

    // Output gradients
    glm::vec3* dL_dmeans3D;  // (P, 3), gradient of center coordinates
    glm::vec3* dL_dgrads3D_abs;
    glm::vec3* dL_dshs;  // (P, M, 3), gradient of SHs
    float* dL_dcolors;  // (P, C), gradient of middle colors
    float* dL_dopacities;  // (P, 1), gradient of opacities
    glm::vec2* dL_dscales;  // (P, 2), gradient of scales
    glm::vec4* dL_drotations;  // (P, 4), gradient of rotations
    float* dL_dtransMat_precomp;  // (P, 9), gradient of trans matrix
};


// Define the primitive info
struct IntersectionInfo {
    float tmx;  // t range along the ray
    uint32_t idx;  // intersection primitive ID
};
// Typedef
typedef struct IntersectionInfo IntersectionInfo;

// Define the ray payload.
// Ray pyaload is used to pass data between optixTrace
// and the programs invoked during ray traversal.
struct RayPayload {
    float dpt;  // trace depth during the whole chunkify tracing
    uint32_t cnt;  // record number of intersections for one chunk
    IntersectionInfo* buffer;  // hit buffer for one chunk
};
// Typedef
typedef struct RayPayload RayPayload;
