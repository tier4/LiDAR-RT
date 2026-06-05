/**
 * @file trace_surfels.cpp
 * @author xbillowy
 * @brief 
 * @version 0.1
 * @date 2024-08-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <math.h>
#include <memory>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <optix.h>
#include <optix_stubs.h>

#include "trace_surfels.h"
#include "optix_tracer/common.h"
#include "optix_tracer/config.h"
#include "optix_tracer/params.h"

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}


void BuildAccelerationStructure(
    OptiXStateWrapper& stateWrapper,
    torch::Tensor& vertices,
    torch::Tensor& triangles,
    unsigned int rebuild)
{
    // Check the dimensions of the input vertices and triangles
    if (vertices.ndimension() != 2 || vertices.size(1) != 3) {
        AT_ERROR("vertices must have dimensions (num_vertices, 3)");
    }
    if (triangles.ndimension() != 2 || triangles.size(1) != 3) {
        AT_ERROR("triangles must have dimensions (num_triangles, 3)");
    }

    // Use default options for simplicity
    // In a real use case we would want to enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    // Determine if we need to rebuild the acceleration structure
    if (rebuild > 0)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(stateWrapper.optixState->d_gas_output_buffer)));
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    }
    else
    {
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    }

    // Prepare triangle inputs
    CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(vertices.contiguous().data_ptr<float>());
    CUdeviceptr d_triangles = reinterpret_cast<CUdeviceptr>(triangles.contiguous().data_ptr<int>());
    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = (uint32_t)(vertices.size(0));
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.numIndexTriplets = (uint32_t)(triangles.size(0));
    triangle_input.triangleArray.indexBuffer = d_triangles;

    // Buffer sizes
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(stateWrapper.optixState->context, &accel_options,
                                             &triangle_input,
                                             1,  // Number of build inputs
                                             &gas_buffer_sizes));
    // Allocate memory for the temporary buffer used during acceleration structure building
    void *d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes));
    // Allocate memory for the non-compacted output acceleration structure and computed compact size
    void *d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_gas_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(&d_buffer_temp_output_gas_and_compacted_size, compacted_gas_size_offset + 8));
    // Emit Property
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result =
        reinterpret_cast<CUdeviceptr>(reinterpret_cast<char *>(d_buffer_temp_output_gas_and_compacted_size) +
            compacted_gas_size_offset);

    // Build the acceleration structure
    OPTIX_CHECK(optixAccelBuild(stateWrapper.optixState->context,
                                0,  // CUDA stream
                                &accel_options,
                                &triangle_input,
                                1,  // Number of build inputs
                                reinterpret_cast<CUdeviceptr>(d_temp_buffer_gas),
                                gas_buffer_sizes.tempSizeInBytes,
                                reinterpret_cast<CUdeviceptr>(
                                    d_buffer_temp_output_gas_and_compacted_size),
                                gas_buffer_sizes.outputSizeInBytes,
                                &stateWrapper.optixState->gas_handle,
                                &emitProperty,  // emitted property list
                                1));  // num emitted properties

    // Free the temporary buffer
    CUDA_CHECK(cudaFree(d_temp_buffer_gas));

    // Get the compacted size
    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void *>(emitProperty.result),
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    // Build the compacted acceleration structure if necessary
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(&stateWrapper.optixState->d_gas_output_buffer, compacted_gas_size));
        // Use handle as input and output
        OPTIX_CHECK(optixAccelCompact(stateWrapper.optixState->context, 0, stateWrapper.optixState->gas_handle,
                                      reinterpret_cast<CUdeviceptr>(stateWrapper.optixState->d_gas_output_buffer),
                                      compacted_gas_size, &stateWrapper.optixState->gas_handle));
        // Free the temporary buffer
        CUDA_CHECK(cudaFree(d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        stateWrapper.optixState->d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsCUDA(
    const OptiXStateWrapper& stateWrapper,
    const bool training,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug,
    const torch::Tensor& pixel_weight,
    const torch::Tensor& target_depth)
{
    // Create CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Check the dimensions of the input xyz
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    // Get dimensions of the input
    const int P = means3D.size(0);
    const int H = ray_o.size(0);
    const int W = ray_o.size(1);

    // Check input
    CHECK_INPUT(ray_o);
    CHECK_INPUT(ray_d);
    CHECK_INPUT(background);
    CHECK_INPUT(means3D);
    CHECK_INPUT(shs);
    CHECK_INPUT(colors_precomp);
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(transMat_precomp);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);

    // Define different output tensor options
    torch::TensorOptions int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::TensorOptions bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    torch::TensorOptions float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // Create normal, distortion, RGB color, depth, acc output tensors output
    torch::Tensor out_attr_float32 = torch::zeros({H, W, NUM_CHANNELS_F}, float_opts);  // normal
    torch::Tensor out_attr_uint32 = -1 * torch::ones({H, W, NUM_CHANNELS_I}, int_opts);  // number of contributions
    torch::Tensor accum_gaussian_weights = torch::zeros({P}, float_opts);
    // Sky-weighted per-Gaussian accumulator. Only meaningful when pixel_weight is
    // provided (non-empty); otherwise stays at zero and the caller can ignore it.
    torch::Tensor accum_gaussian_sky_weights = torch::zeros({P}, float_opts);
    // Front-side per-pixel accumulator. Only meaningful when target_depth is
    // provided (non-empty); zero otherwise.
    torch::Tensor accum_at_target = torch::zeros({H, W}, float_opts);
    // Per-Gaussian front-side accumulator. Zero unless target_depth was
    // provided (non-empty); used by the multi-view front-side hard prune.
    torch::Tensor accum_gaussian_front_weights = torch::zeros({P}, float_opts);


    // Create global parameters for the OptiX program
    Params params;
    params.handle = stateWrapper.optixState->gas_handle;
    // Store input parameters
    params.P = P;
    params.H = H;
    params.W = W;
    params.D = degree;
    int M = 0;
    if (shs.size(0) != 0)
    {
        M = shs.size(1);
    }
    params.M = M;
    params.training = training;
    params.ray_o = reinterpret_cast<float3*>(ray_o.contiguous().data_ptr<float>());
    params.ray_d = reinterpret_cast<float3*>(ray_d.contiguous().data_ptr<float>());
    params.vertices = reinterpret_cast<float3*>(vertices.contiguous().data_ptr<float>());
    params.background = background.contiguous().data_ptr<float>();
    params.means3D = reinterpret_cast<glm::vec3*>(means3D.contiguous().data_ptr<float>());
    params.shs = shs.contiguous().data_ptr<float>();
    params.colors_precomp = colors_precomp.contiguous().data_ptr<float>();
    params.opacities = opacities.contiguous().data_ptr<float>();
    params.scales = reinterpret_cast<glm::vec2*>(scales.contiguous().data_ptr<float>());
    params.scale_modifier = scale_modifier;
    params.rotations = reinterpret_cast<glm::vec4*>(rotations.contiguous().data_ptr<float>());
    params.transMat_precomp = transMat_precomp.contiguous().data_ptr<float>();
    params.viewmatrix = viewmatrix.contiguous().data_ptr<float>();
    params.projmatrix = projmatrix.contiguous().data_ptr<float>();
    params.campos = reinterpret_cast<glm::vec3*>(campos.contiguous().data_ptr<float>());
    // Store output tensors;
    params.out_attr_float32 = out_attr_float32.contiguous().data_ptr<float>();
    params.out_attr_uint32 = out_attr_uint32.contiguous().data_ptr<int>();
    params.accum_gaussian_weights = accum_gaussian_weights.contiguous().data_ptr<float>();
    // pixel_weight is optional: caller may pass an empty tensor to skip the
    // sky-weighted accumulation path entirely.
    params.pixel_weight = pixel_weight.numel() > 0 ? pixel_weight.contiguous().data_ptr<float>() : nullptr;
    params.accum_gaussian_sky_weights = accum_gaussian_sky_weights.contiguous().data_ptr<float>();
    // target_depth is optional: caller may pass an empty tensor to skip the
    // front-side accumulation snapshot path entirely. accum_at_target is
    // always allocated but stays zero unless target_depth was supplied.
    params.target_depth = target_depth.numel() > 0 ? target_depth.contiguous().data_ptr<float>() : nullptr;
    params.accum_at_target = accum_at_target.contiguous().data_ptr<float>();
    params.dL_daccum_at_target = nullptr;  // unused in forward
    // Per-Gaussian front-side accumulator (only active when target_depth was
    // provided; otherwise the atomicAdd path is gated off by target_depth==nullptr).
    params.accum_gaussian_front_weights = target_depth.numel() > 0
        ? accum_gaussian_front_weights.contiguous().data_ptr<float>()
        : nullptr;


    // Allocate memory for the parameters
    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params),
                          &params, sizeof(params),
                          cudaMemcpyHostToDevice));

    // Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(stateWrapper.optixState->pipelie_surfel_tracing_forward, stream, d_params, sizeof(Params),
                           &stateWrapper.optixState->sbt_surfel_tracing_forward, H, W, 1));

    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Return
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        out_attr_float32, out_attr_uint32, accum_gaussian_weights,
        accum_gaussian_sky_weights, accum_at_target,
        accum_gaussian_front_weights);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsBackwardCUDA(
    const OptiXStateWrapper& stateWrapper,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug,
    const torch::Tensor& out_attr_float32,
    const torch::Tensor& out_attr_uint32,
    const torch::Tensor& dL_dout_attr_float32,
    const torch::Tensor& target_depth,
    const torch::Tensor& accum_at_target,
    const torch::Tensor& dL_daccum_at_target
){
    // Create CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Check input
    CHECK_INPUT(ray_o);
    CHECK_INPUT(ray_d);
    CHECK_INPUT(background);
    CHECK_INPUT(means3D);
    CHECK_INPUT(shs);
    CHECK_INPUT(colors_precomp);
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(transMat_precomp);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);

    // Get dimensions of the input
    const int P = means3D.size(0);
    const int H = ray_o.size(0);
    const int W = ray_o.size(1);
    int M = 0;
    if (shs.size(0) != 0)
    {
        M = shs.size(1);
    }

    // Create output gradient Tensors
    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dshs = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dopacities = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    torch::Tensor dL_dtransMat_precomp = torch::zeros({P, 9}, means3D.options());
    torch::Tensor dL_dgrads3D_abs = torch::zeros({P, 3}, means3D.options());
    // Create global parameters for the OptiX program
    Params params;
    params.handle = stateWrapper.optixState->gas_handle;
    // Store input parameters
    params.P = P;
    params.H = H;
    params.W = W;
    params.D = degree;
    params.M = M;
    // Store input parameters
    params.ray_o = reinterpret_cast<float3*>(ray_o.contiguous().data_ptr<float>());
    params.ray_d = reinterpret_cast<float3*>(ray_d.contiguous().data_ptr<float>());
    params.vertices = reinterpret_cast<float3*>(vertices.contiguous().data_ptr<float>());
    params.background = background.contiguous().data_ptr<float>();
    params.means3D = reinterpret_cast<glm::vec3*>(means3D.contiguous().data_ptr<float>());
    params.shs = shs.contiguous().data_ptr<float>();
    params.colors_precomp = colors_precomp.contiguous().data_ptr<float>();
    params.opacities = opacities.contiguous().data_ptr<float>();
    params.scales = reinterpret_cast<glm::vec2*>(scales.contiguous().data_ptr<float>());
    params.scale_modifier = scale_modifier;
    params.rotations = reinterpret_cast<glm::vec4*>(rotations.contiguous().data_ptr<float>());
    params.transMat_precomp = transMat_precomp.contiguous().data_ptr<float>();
    params.viewmatrix = viewmatrix.contiguous().data_ptr<float>();
    params.projmatrix = projmatrix.contiguous().data_ptr<float>();
    params.campos = reinterpret_cast<glm::vec3*>(campos.contiguous().data_ptr<float>());
    // Store output tensors
    params.out_attr_float32 = out_attr_float32.contiguous().data_ptr<float>();
    params.out_attr_uint32 = out_attr_uint32.contiguous().data_ptr<int>();
    // Store input upstream gradients
    params.dL_dout_attr_float32 = dL_dout_attr_float32.contiguous().data_ptr<float>();
    // Front-side accumulation pointers. All three must be non-null together
    // for the new gradient path to fire; otherwise it's a no-op.
    params.target_depth = target_depth.numel() > 0 ? target_depth.contiguous().data_ptr<float>() : nullptr;
    params.accum_at_target = accum_at_target.numel() > 0 ? accum_at_target.contiguous().data_ptr<float>() : nullptr;
    params.dL_daccum_at_target = dL_daccum_at_target.numel() > 0 ? dL_daccum_at_target.contiguous().data_ptr<float>() : nullptr;
    params.accum_gaussian_front_weights = nullptr;  // forward-only path
    // Store output gradients
    params.dL_dmeans3D = reinterpret_cast<glm::vec3*>(dL_dmeans3D.contiguous().data_ptr<float>());
    params.dL_dgrads3D_abs = reinterpret_cast<glm::vec3*>(dL_dgrads3D_abs.contiguous().data_ptr<float>());
    params.dL_dshs = reinterpret_cast<glm::vec3*>(dL_dshs.contiguous().data_ptr<float>());
    params.dL_dcolors = dL_dcolors.contiguous().data_ptr<float>();
    params.dL_dopacities = dL_dopacities.contiguous().data_ptr<float>();
    params.dL_dscales = reinterpret_cast<glm::vec2*>(dL_dscales.contiguous().data_ptr<float>());
    params.dL_drotations = reinterpret_cast<glm::vec4*>(dL_drotations.contiguous().data_ptr<float>());
    params.dL_dtransMat_precomp = dL_dtransMat_precomp.contiguous().data_ptr<float>();

    // Allocate memory for the parameters
    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params),
                          &params, sizeof(params),
                          cudaMemcpyHostToDevice));

    // Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(stateWrapper.optixState->pipelie_surfel_tracing_backward, stream, d_params, sizeof(Params),
                            &stateWrapper.optixState->sbt_surfel_tracing_backward, H, W, 1));

    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        dL_dmeans3D, dL_dshs, dL_dcolors, dL_dopacities, dL_dscales, dL_drotations, dL_dtransMat_precomp, dL_dgrads3D_abs);
}
