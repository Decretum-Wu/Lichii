// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <tuple>
#include <set>
#include <span>
#include <fstream>
#include <mutex>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "common/cityhash.h"
#include "common/dkb_frame_trace.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/native_query_trace.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/render_trace.h"
#include "video_core/indirect_reset.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/framebuffer_clear_layers.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_fsr3_upscaler.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#endif
#include "video_core/renderer_vulkan/submission_policy.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/textures/texture.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_draw_stages.h"
#include "video_core/nx2_profile.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {

/// read_gpu_memory: the first address space (the bound channel's first) that maps the whole range,
/// read as the GPU reads it, without flushing GPU caches.
VideoCore::Nx2Debug::GpuReadResult ReadGpuMemory(std::span<Tegra::MemoryManager* const> spaces,
                                                 GPUVAddr address, u64 size) {
    VideoCore::Nx2Debug::GpuReadResult result{};
    if (spaces.empty()) {
        result.error = "no GPU address space is bound yet";
        return result;
    }
    for (Tegra::MemoryManager* const space : spaces) {
        if (space->IsFullyMappedRange(address, size)) {
            result.bytes.resize(size);
            space->ReadBlockUnsafe(address, result.bytes.data(), size);
            result.address_space = space->GetID();
            return result;
        }
    }
    constexpr u64 page_mask = 0xfff;
    const GPUVAddr end = address + size;
    GPUVAddr first = address;
    while (first < end) {
        const GPUVAddr next = (std::min)((first | page_mask) + 1, end);
        if (!spaces.front()->IsFullyMappedRange(first, next - first)) {
            break;
        }
        first = next;
    }
    result.error = fmt::format("GPU address {:#x} is not mapped (the first unmapped byte in the "
                               "bound channel's address space; {} address spaces checked)",
                               first, spaces.size());
    return result;
}

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

// The part of a volume slice a clear writes: the scissor (when the clear uses it) limited to the
// render area and the slice.
VkRect2D VolumeClearRect(const Maxwell& regs, VkExtent2D render_area, VideoCommon::Extent3D slice) {
    VkRect2D rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0)
                                                   : VkRect2D{{0, 0}, {slice.width, slice.height}};
    const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
        const s64 begin = std::clamp<s64>(offset, 0, limit);
        const s64 end = std::clamp<s64>(s64{offset} + extent, begin, limit);
        offset = static_cast<s32>(begin);
        extent = static_cast<u32>(end - begin);
    };
    clamp_axis(rect.offset.x, rect.extent.width, (std::min)(render_area.width, slice.width));
    clamp_axis(rect.offset.y, rect.extent.height, (std::min)(render_area.height, slice.height));
    return rect;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}

/// Earlier writes must be visible to a dispatch, including its indirect arguments.
void RecordDispatchBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, READ_BARRIER);
}

/// EDEN_NX2_DRAW_PACKET_VERIFY=1, read once.
bool DrawPacketVerify() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_DRAW_PACKET_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// What the dynamic state values read besides the registers. The pipeline lookup and the
/// repacked vertex buffers are filled only for the groups that read them.
struct DynamicStateInputs {
    const Device* device{};
    bool is_rescaling{};
    Maxwell::PrimitiveTopology topology{};
    u32 color_output_mask{~0U};
    bool d24_depth_bias_rescale{};
    const GraphicsPipeline* lookup_pipeline{};
    u32 repacked_vertex_buffers{};
    std::array<f32, 2> viewport_offset{}; ///< Ocarina FSR camera jitter (render pixels)
};

/// The values of dynamic state group `group`, read from the registers. Stage A and the draw packet
/// verify both read them here.
void ReadDynamicGroup(DynamicStatePacket::Group group, const Maxwell& regs,
                      const DynamicStateInputs& inputs, DynamicStatePacket& out) {
    using G = DynamicStatePacket;
    const Device& device = *inputs.device;
    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool two_sided = regs.stencil_two_side_enable != 0;
    switch (group) {
    case G::Viewports:
        if (!regs.viewport_scale_offset_enabled) {
            const float x = static_cast<float>(regs.surface_clip.x);
            float y = static_cast<float>(regs.surface_clip.y);
            const float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
            float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
            if (lower_left) {
                y += height;
                height = -height;
            }
            out.viewports.fill(VkViewport{
                .x = x + inputs.viewport_offset[0],
                .y = y + inputs.viewport_offset[1],
                .width = width,
                .height = height,
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            });
            return;
        }
        for (size_t index = 0; index < Maxwell::NumViewports; ++index) {
            out.viewports[index] = GetViewportState(
                device, regs, index,
                inputs.is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f);
            out.viewports[index].x += inputs.viewport_offset[0];
            out.viewports[index].y += inputs.viewport_offset[1];
        }
        return;
    case G::Scissors: {
        if (!regs.viewport_scale_offset_enabled) {
            const u32 x = regs.surface_clip.x;
            u32 y = regs.surface_clip.y;
            const u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
            const u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
            if (lower_left) {
                y = regs.surface_clip.height - (y + height);
            }
            VkRect2D scissor{};
            scissor.offset.x = static_cast<int32_t>(x);
            scissor.offset.y = static_cast<int32_t>(y);
            scissor.extent.width = width;
            scissor.extent.height = height;
            out.scissors.fill(scissor);
            return;
        }
        u32 up_scale = 1;
        u32 down_shift = 0;
        if (inputs.is_rescaling) {
            up_scale = Settings::values.resolution_info.up_scale;
            down_shift = Settings::values.resolution_info.down_shift;
        }
        for (size_t index = 0; index < Maxwell::NumViewports; ++index) {
            out.scissors[index] = GetScissorState(regs, index, up_scale, down_shift);
        }
        return;
    }
    case G::DepthBias: {
        float units = regs.depth_bias / 2.0f;
        const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                            regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                            regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                            regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;
        if (is_d24 && inputs.d24_depth_bias_rescale) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
        out.depth_bias = {units, regs.depth_bias_clamp, regs.slope_scale_depth_bias};
        return;
    }
    case G::BlendConstants:
        out.blend_constants = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                               regs.blend_color.a};
        return;
    case G::DepthBounds:
        out.depth_bounds = {regs.depth_bounds[0], regs.depth_bounds[1]};
        return;
    case G::StencilReference:
        out.stencil_reference = {regs.stencil_front_ref, regs.stencil_back_ref, two_sided};
        return;
    case G::StencilWriteMask:
        out.stencil_write_mask = {regs.stencil_front_mask, regs.stencil_back_mask, two_sided};
        return;
    case G::StencilCompareMask:
        out.stencil_compare_mask = {regs.stencil_front_func_mask, regs.stencil_back_func_mask,
                                    two_sided};
        return;
    case G::LineWidth:
        out.line_width =
            regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
        return;
    case G::CullMode:
        out.cull_mode = regs.gl_cull_test_enabled ? MaxwellToVK::CullFace(regs.gl_cull_face)
                                                  : VK_CULL_MODE_NONE;
        return;
    case G::DepthCompareOp:
        out.depth_compare_op = MaxwellToVK::ComparisonOp(regs.depth_test_func);
        return;
    case G::FrontFace: {
        VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
        if (regs.window_origin.flip_y != 0) {
            front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                               : VK_FRONT_FACE_CLOCKWISE;
        }
        out.front_face = front_face;
        return;
    }
    case G::StencilOp: {
        const auto face_ops = [](const auto& op) {
            return G::StencilFaceOps{
                .fail = MaxwellToVK::StencilOp(op.fail),
                .pass = MaxwellToVK::StencilOp(op.zpass),
                .depth_fail = MaxwellToVK::StencilOp(op.zfail),
                .compare = MaxwellToVK::ComparisonOp(op.func),
            };
        };
        out.stencil_op = {
            .front = face_ops(regs.stencil_front_op),
            .back = two_sided ? face_ops(regs.stencil_back_op) : G::StencilFaceOps{},
            .two_sided = two_sided,
        };
        return;
    }
    case G::DepthBoundsTestEnable:
        out.depth_bounds_test_enable = regs.depth_bounds_enable && device.IsDepthBoundsSupported();
        return;
    case G::DepthTestEnable:
        out.depth_test_enable = regs.depth_test_enable != 0;
        return;
    case G::DepthWriteEnable:
        out.depth_write_enable = regs.depth_write_enabled != 0;
        return;
    case G::StencilTestEnable:
        out.stencil_test_enable = regs.stencil_enable != 0;
        return;
    case G::PrimitiveTopology:
        out.primitive_topology = MaxwellToVK::PrimitiveTopology(device, inputs.topology);
        return;
    case G::PrimitiveRestartEnable: {
        bool enable = regs.primitive_restart.enabled != 0;
        if (device.IsMoltenVK()) {
            enable = true;
        } else if (enable) {
            enable = IsPrimitiveRestartSupported(
                device, MaxwellToVK::PrimitiveTopology(device, inputs.topology));
        }
        out.primitive_restart_enable = enable;
        return;
    }
    case G::RasterizerDiscardEnable:
        out.rasterizer_discard_enable = regs.rasterize_enable == 0;
        return;
    case G::DepthBiasEnable: {
        constexpr size_t POINT = 0;
        constexpr size_t LINE = 1;
        constexpr size_t POLYGON = 2;
        static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
            POINT,   // Points
            LINE,    // Lines
            LINE,    // LineLoop
            LINE,    // LineStrip
            POLYGON, // Triangles
            POLYGON, // TriangleStrip
            POLYGON, // TriangleFan
            POLYGON, // Quads
            POLYGON, // QuadStrip
            POLYGON, // Polygon
            LINE,    // LinesAdjacency
            LINE,    // LineStripAdjacency
            POLYGON, // TrianglesAdjacency
            POLYGON, // TriangleStripAdjacency
            POLYGON, // Patches
        };
        const std::array enabled_lut{
            regs.polygon_offset_point_enable,
            regs.polygon_offset_line_enable,
            regs.polygon_offset_fill_enable,
        };
        out.depth_bias_enable =
            enabled_lut[POLYGON_OFFSET_ENABLE_LUT[static_cast<u32>(inputs.topology)]] != 0;
        return;
    }
    case G::LogicOp: {
        const auto op_value = static_cast<u32>(regs.logic_op.op);
        out.logic_op = op_value >= 0x1500 && op_value < 0x1510
                           ? static_cast<VkLogicOp>(op_value - 0x1500)
                           : VK_LOGIC_OP_NO_OP;
        return;
    }
    case G::LogicOpEnable:
        out.logic_op_enable = regs.logic_op.enable != 0;
        return;
    case G::DepthClampEnable: {
        const auto clip = regs.viewport_clip_control.geometry_clip;
        out.depth_clamp_enable =
            !(clip == Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
              clip == Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
              clip == Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
        return;
    }
    case G::LineRasterizationMode:
        out.line_rasterization_mode = regs.line_anti_alias_enable != 0 && device.SupportsSmoothLines()
                                          ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                          : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
        return;
    case G::LineStippleEnable:
        out.line_stipple_enable = regs.line_stipple_enable != 0;
        return;
    case G::ConservativeRasterization:
        out.conservative_rasterization = regs.conservative_raster_enable != 0;
        return;
    case G::AlphaToCoverageEnable:
        out.alpha_to_coverage_enable = inputs.lookup_pipeline != nullptr &&
                                       inputs.lookup_pipeline->SupportsAlphaToCoverage() &&
                                       regs.anti_alias_alpha_control.alpha_to_coverage != 0;
        return;
    case G::AlphaToOneEnable:
        out.alpha_to_one_enable = inputs.lookup_pipeline != nullptr &&
                                  inputs.lookup_pipeline->SupportsAlphaToOne() &&
                                  regs.anti_alias_alpha_control.alpha_to_one != 0;
        return;
    case G::ColorWriteMask:
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            VkColorComponentFlags& current = out.color_write_mask[index];
            current = 0;
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
            current &= (inputs.color_output_mask >> (index * 4)) & 0xF;
        }
        return;
    case G::BlendEnable:
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            out.blend_enable[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        return;
    case G::BlendEquations: {
        auto& blends = out.blend_equations;
        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };
        if (regs.blend_per_target_enabled) {
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(blends[index], regs.blend_per_target[index]);
            }
            return;
        }
        // Temporary workaround for games that use iterated blending
        if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
            blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blends[0].colorBlendOp = VK_BLEND_OP_ADD;
            blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
        } else {
            blend_setup(blends[0], regs.blend);
        }
        // One blend equation for all targets
        for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
            blends[index] = blends[0];
        }
        return;
    }
    case G::ColorWriteEnable:
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            out.color_write_enable[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE
                                                                                   : VK_FALSE;
        }
        return;
    case G::VertexInput: {
        auto& state = out.vertex_input;
        state.num_bindings = 0;
        state.num_attributes = 0;
        const u32 max_attributes = static_cast<u32>(std::min<size_t>(
            Maxwell::NumVertexAttributes, device.GetMaxVertexInputAttributes()));
        const u32 max_bindings = static_cast<u32>(
            std::min<size_t>(Maxwell::NumVertexArrays, device.GetMaxVertexInputBindings()));
        const auto repacked = [&](u32 binding) {
            return ((inputs.repacked_vertex_buffers >> binding) & 1) != 0;
        };
        for (u32 index = 0; index < max_attributes; ++index) {
            const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
            const u32 binding{attribute.buffer};
            if (attribute.constant || binding >= max_bindings) {
                continue;
            }
            state.attributes[state.num_attributes++] = {
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = index,
                .binding = binding,
                .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
                .offset = attribute.offset + (repacked(binding) && attribute.offset >= 65 ? 3 : 0),
            };
        }
        for (u32 binding = 0; binding < max_bindings; ++binding) {
            const auto& input_binding{regs.vertex_streams[binding]};
            const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
            state.bindings[state.num_bindings++] = {
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .binding = binding,
                .stride = repacked(binding) ? 72U : input_binding.stride.Value(),
                .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE
                                          : VK_VERTEX_INPUT_RATE_VERTEX,
                .divisor = is_instanced ? input_binding.frequency : 1,
            };
        }
        return;
    }
    case G::Count:
        return;
    }
}

} // Anonymous namespace

void RasterizerVulkan::ReadMeshEmulationProbe() {
    const MeshEmulationProbe probe{mesh_probe};
    mesh_probe.pending = false;
    const size_t num_words{probe.data.size() / sizeof(u32)};
    const auto word = [&](u64 index) -> u32 {
        if (index >= num_words) {
            return 0;
        }
        u32 value;
        std::memcpy(&value, probe.data.data() + index * sizeof(u32), sizeof(value));
        return value;
    };
    const auto position = [&](u64 base, u32 vertex) {
        std::array<f32, 4> result{};
        for (u32 element = 0; element < 4; ++element) {
            const u32 line{probe.position_lines[element]};
            if (line != ~0U) {
                result[element] = std::bit_cast<f32>(word(base + u64{vertex} * probe.vertex_lines +
                                                          line));
            }
        }
        return result;
    };
    u64 triangles{}, degenerate{}, outside{}, bad_index{};
    std::string sample;
    for (u32 group = 0; group < probe.groups; ++group) {
        const u64 base{u64{group} * probe.block_words};
        const u64 index_base{base + u64{probe.max_vertices} * probe.vertex_lines};
        for (u32 primitive = 0; primitive < probe.max_primitives; ++primitive) {
            const std::array<u32, 3> indices{word(index_base + primitive * 3),
                                             word(index_base + primitive * 3 + 1),
                                             word(index_base + primitive * 3 + 2)};
            if (indices[0] == indices[1] && indices[1] == indices[2]) {
                ++degenerate;
                continue;
            }
            ++triangles;
            bool any_bad{};
            bool any_inside{};
            for (const u32 vertex : indices) {
                if (vertex >= probe.max_vertices) {
                    any_bad = true;
                    continue;
                }
                const auto p{position(base, vertex)};
                const f32 w{std::abs(p[3])};
                any_inside |= w > 0.0f && std::abs(p[0]) <= w && std::abs(p[1]) <= w;
            }
            bad_index += any_bad;
            outside += !any_inside;
            if (sample.size() < 400 && triangles <= 2) {
                for (const u32 vertex : indices) {
                    const auto p{position(base, std::min(vertex, probe.max_vertices - 1))};
                    sample += fmt::format(" g{} v{} ({:.3g} {:.3g} {:.3g} {:.3g})", group, vertex,
                                          p[0], p[1], p[2], p[3]);
                }
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Mesh emulation probe frame {}: {} groups, {} triangles ({} degenerate slots), {} "
             "with every vertex off screen, {} with a bad index; first triangles:{}",
             probe.frame, probe.groups, triangles, degenerate, outside, bad_index, sample);
}

const RasterizerVulkan::MeshArena& RasterizerVulkan::AcquireMeshArena() {
    const u64 tick{scheduler.CurrentTick()};
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // A finished submission's counter holds the words its draws asked for, including those
    // that did not fit: grow later arenas to hold them.
    const u64 max_size{(std::min)(u64{1} << 30, u64{device.GetMaxStorageBufferRange()})};
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        arena.counter.Invalidate();
        u32 words;
        std::memcpy(&words, arena.counter.Mapped().data(), sizeof(words));
        const u64 needed{u64{words} * 4};
        if (needed > arena.size && mesh_arena_size < max_size) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), max_size);
            LOG_INFO(Render_Vulkan, "Mesh emulation: a submission needed {} MiB of output, arenas "
                                    "grow to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    std::erase_if(mesh_arenas, [&](const MeshArena& arena) {
        return arena.counted && arena.size < mesh_arena_size;
    });
    auto it{std::ranges::find_if(mesh_arenas, [](const MeshArena& arena) { return arena.counted; })};
    if (it == mesh_arenas.end()) {
        const auto make_buffer = [&](u64 size, VkBufferUsageFlags usage, MemoryUsage memory) {
            return memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                memory);
        };
        it = mesh_arenas.insert(
            mesh_arenas.end(),
            MeshArena{
                .buffer = make_buffer(mesh_arena_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      MemoryUsage::DeviceLocal),
                .counter = make_buffer(sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryUsage::Download),
                .size = mesh_arena_size,
            });
    }
    // The GPU is done with this arena, and this submission's commands have not been sent yet.
    const u32 zero{};
    std::memcpy(it->counter.Mapped().data(), &zero, sizeof(zero));
    it->counter.Flush();
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      back_staging_pool(VideoCommon::DrawStages::BackStagesRequested()
                            ? std::make_unique<StagingBufferPool>(device, memory_allocator,
                                                                  scheduler)
                            : nullptr),
      back_compute_pass_descriptor_queue(
          back_staging_pool ? std::make_unique<ComputePassDescriptorQueue>(
                                  device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE)
                            : nullptr),
      back_descriptor_queue(back_staging_pool
                                ? std::make_unique<GuestDescriptorQueue>(
                                      device, 1, device.IsExtDescriptorBufferSupported())
                                : nullptr),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler,
                           back_staging_pool ? *back_staging_pool : staging_pool, staging_pool,
                           guest_descriptor_queue,
                           back_compute_pass_descriptor_queue ? *back_compute_pass_descriptor_queue
                                                              : compute_pass_descriptor_queue,
                           descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      frame_dump(device, memory_allocator, scheduler) {
    VideoCommon::DrawStages::Configure();
    if (back_descriptor_queue) {
        buffer_cache_runtime.SetStageDescriptorQueue(*back_descriptor_queue);
    }
    scheduler.SetQueryCache(query_cache);
    buffer_cache.SetPendingCpuWrites(
        [this](const std::function<void(DAddr, size_t)>& callback) {
            gpu.ForEachPendingCpuWrite(callback);
        });
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument) {
            VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::IdleHook);
            static_cast<Scheduler*>(argument)->FlushDeferred();
        },
        &scheduler);
}

RasterizerVulkan::~RasterizerVulkan() {
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr);
    if (back_ring) {
        StopBackStages();
    }
    scheduler.WaitWorker();
    scheduler.Finish();
}

GraphicsPipeline* RasterizerVulkan::TakeReusableDraw(bool is_indexed, bool is_indirect) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    const ReusableDraw draw = std::exchange(reusable_draw, ReusableDraw{});
    const FastDraw::Delta delta = std::exchange(FastDraw::g_delta, FastDraw::Delta{});
    taken_delta = delta;
    const auto full = [&](FullDraw reason) -> GraphicsPipeline* {
        ++fast_draw_stats.full[static_cast<size_t>(reason)];
        return nullptr;
    };
    if (draw.pipeline == nullptr) {
        return full(FullDraw::NoCandidate);
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (is_indirect || draw.is_indexed != is_indexed || draw.maxwell3d != maxwell3d ||
        draw.topology != draw_state.topology || !draw_state.inline_index_draw_indexes.empty() ||
        frame_dump.Capturing()) {
        return full(FullDraw::Kind);
    }
    if (draw.serial != FastDraw::Serial()) {
        const u32 last_break = FastDraw::g_last_break.load(std::memory_order_relaxed);
        ++fast_draw_stats.breaks[last_break & 0xff];
        if (const u32 method = last_break >> 8; method < Maxwell::NUM_REGS) {
            ++fast_draw_stats.break_methods[method];
        }
        return full(FullDraw::StateChanged);
    }
    if (draw.map_serial != gpu_memory->MapSerial()) {
        ++fast_draw_stats.breaks[static_cast<size_t>(FastDraw::Reason::Map)];
        return full(FullDraw::StateChanged);
    }
    if (draw.tick != scheduler.CurrentTick()) {
        return full(FullDraw::Submitted);
    }
    if (draw.render_pass != RenderPassSerialOfA() || !RenderPassActiveOfA() ||
        BoundPipelineOfA() != draw.pipeline) {
        return full(FullDraw::RenderPass);
    }
    if (!draw.pipeline->TakesDeltaDraws()) {
        ++fast_draw_stats.delta_blockers[static_cast<size_t>(draw.pipeline->DeltaDrawBlocker())];
        return full(FullDraw::DeltaPipeline);
    }
    const auto& handles = draw.pipeline->HandleConstantBuffers();
    for (size_t stage = 0; stage < handles.size() && !delta.Empty(); ++stage) {
        if ((delta.binds[stage] & handles[stage]) != 0) {
            return full(FullDraw::HandleBind);
        }
        if (delta.num_writes == 0) {
            continue;
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        for (u32 mask = handles[stage]; mask != 0; mask &= mask - 1) {
            const u32 slot = std::countr_zero(mask);
            const auto& cbuf = cbufs[slot];
            FastDraw::DeviceSpans& spans = handle_spans[stage][slot];
            spans.Update(*gpu_memory, cbuf.address, cbuf.size);
            if (delta.Reaches(cbuf.address, cbuf.size, spans)) {
                return full(FullDraw::HandleData);
            }
        }
    }
    taken_draw = draw;
    return draw.pipeline;
}

bool RasterizerVulkan::DeltaStillHolds(u64 serial, u64 map_serial) {
    const auto full = [&](FullDraw reason) {
        ++fast_draw_stats.full[static_cast<size_t>(reason)];
        return false;
    };
    // TakeReusableDraw ran before the caches were locked, while other threads could still write,
    // unmap or delete what the delta draw keeps.
    if (taken_draw.serial != serial || taken_draw.map_serial != map_serial) {
        return full(FullDraw::StateChanged);
    }
    if (taken_draw.tick != scheduler.CurrentTick()) {
        return full(FullDraw::Submitted);
    }
    if (taken_draw.render_pass != RenderPassSerialOfA() || !RenderPassActiveOfA()) {
        return full(FullDraw::RenderPass);
    }
    // A delta draw prepares none of the images it samples: one with an upload pending or a newer
    // alias to copy in (a render target aliasing it, written by the draw before) needs Configure.
    if (taken_draw.pipeline->DeltaImagesChange()) {
        return full(FullDraw::SampledImages);
    }
    return true;
}

void RasterizerVulkan::NoteResolvedDraw(GraphicsPipeline* pipeline, bool is_indexed,
                                        bool is_indirect, u64 serial, u64 map_serial) {
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto topology = draw_state.topology;
    const auto ineligible = [&]() -> std::optional<Ineligible> {
        if (is_indirect) return Ineligible::Indirect;
        if (mesh_draw) return Ineligible::Mesh;
        if (!pipeline->IsBuilt()) return Ineligible::NotBuilt;
        if (pipeline->WritesImages()) return Ineligible::WritesImages;
        if (maxwell3d->regs.transform_feedback_enabled != 0) return Ineligible::TransformFeedback;
        if (!draw_state.inline_index_draw_indexes.empty()) return Ineligible::InlineIndex;
        if (topology == Maxwell::PrimitiveTopology::Quads ||
            topology == Maxwell::PrimitiveTopology::QuadStrip) {
            return Ineligible::Quads;
        }
        if (RepackedVertexBuffers() != 0) return Ineligible::Repacked;
        if (!RenderPassActiveOfA()) return Ineligible::NoRenderPass;
        if (BoundPipelineOfA() != pipeline) return Ineligible::PipelineNotBound;
        if (!VideoCommon::GpuSplit::OnBackThread()) return Ineligible::OffBackThread;
        return std::nullopt;
    }();
    if (ineligible) {
        ++fast_draw_stats.ineligible[static_cast<size_t>(*ineligible)];
        return;
    }
    reusable_draw = ReusableDraw{
        .pipeline = pipeline,
        .maxwell3d = maxwell3d,
        .serial = serial,
        .map_serial = map_serial,
        .tick = scheduler.CurrentTick(),
        .render_pass = RenderPassSerialOfA(),
        .topology = topology,
        .is_indexed = is_indexed,
    };
}

void RasterizerVulkan::ReportFastDraws() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    FastDrawStats& stats = fast_draw_stats;
    if (stats.window_start == 0) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < 10'000'000'000ULL) {
        return;
    }
    // Logged only with EDEN_NX2_GPU_PROFILE or EDEN_NX2_FAST_DRAW_VERIFY.
    if (!VideoCommon::Nx2Profile::Enabled() && !fast_draw_verify) {
        stats = FastDrawStats{.window_start = now};
        return;
    }
    const double seconds = static_cast<double>(now - stats.window_start) / 1e9;
    const auto per_second = [&](FullDraw reason) {
        return static_cast<double>(stats.full[static_cast<size_t>(reason)]) / seconds;
    };
    LOG_INFO(Render_Vulkan,
             "Fast draw per second: {:.0f} delta | full: {:.0f} after an ineligible draw, {:.0f} "
             "different kind, {:.0f} state changed, {:.0f} after a submit, {:.0f} render pass or "
             "pipeline changed, {:.0f} pipeline takes no delta draws, {:.0f} handle buffer bound, "
             "{:.0f} handle data written, {:.0f} sampled image to prepare",
             static_cast<double>(stats.delta) / seconds, per_second(FullDraw::NoCandidate),
             per_second(FullDraw::Kind), per_second(FullDraw::StateChanged),
             per_second(FullDraw::Submitted), per_second(FullDraw::RenderPass),
             per_second(FullDraw::DeltaPipeline), per_second(FullDraw::HandleBind),
             per_second(FullDraw::HandleData), per_second(FullDraw::SampledImages));
    static constexpr std::array<const char*, static_cast<size_t>(Ineligible::Count)>
        ineligible_names{"indirect",       "mesh",          "not built",     "writes images",
                         "transform feedback", "inline index", "quads",       "repacked vertices",
                         "no render pass", "pipeline not bound", "off the back thread"};
    std::string ineligible;
    for (size_t reason = 0; reason < stats.ineligible.size(); ++reason) {
        if (stats.ineligible[reason] != 0) {
            ineligible += fmt::format(" {} {:.0f}", ineligible_names[reason],
                                      static_cast<double>(stats.ineligible[reason]) / seconds);
        }
    }
    const auto micros = [&](DrawPath path) {
        const size_t index = static_cast<size_t>(path);
        return stats.prepare_draws[index] == 0
                   ? 0.0
                   : static_cast<double>(stats.prepare_ns[index]) / 1e3 /
                         static_cast<double>(stats.prepare_draws[index]);
    };
    const u64 all_draws = stats.prepare_draws[0] + stats.prepare_draws[1];
    const u64 all_ns = stats.prepare_ns[0] + stats.prepare_ns[1];
    static constexpr std::array<const char*,
                                static_cast<size_t>(GraphicsPipeline::DeltaBlocker::Count)>
        blocker_names{"none", "writes images", "mesh", "computed handles", "not configured"};
    std::string blockers;
    for (size_t blocker = 0; blocker < stats.delta_blockers.size(); ++blocker) {
        if (stats.delta_blockers[blocker] != 0) {
            blockers += fmt::format(" {} {:.0f}", blocker_names[blocker],
                                    static_cast<double>(stats.delta_blockers[blocker]) / seconds);
        }
    }
    ineligible += " | pipelines without delta draws:" + (blockers.empty() ? std::string{" none"} : blockers);
    LOG_INFO(Render_Vulkan,
             "Fast draw ineligible per second:{} | prepare us per draw: full {:.2f}, delta {:.2f}, "
             "all {:.2f} ({:.0f} draws/s)",
             ineligible, micros(DrawPath::Full), micros(DrawPath::Delta),
             all_draws == 0 ? 0.0 : static_cast<double>(all_ns) / 1e3 / static_cast<double>(all_draws),
             static_cast<double>(all_draws) / seconds);
    if (fast_draw_verify) {
        LOG_INFO(Render_Vulkan,
                 "Fast draw verify per second: {:.0f} checked, {:.0f} pipeline mismatches, {:.0f} "
                 "view or sampler mismatches, {:.0f} kept buffer binding mismatches (totals {} {} "
                 "{} {})",
                 static_cast<double>(stats.verified) / seconds,
                 static_cast<double>(stats.pipeline_mismatches) / seconds,
                 static_cast<double>(stats.view_mismatches) / seconds,
                 static_cast<double>(stats.binding_mismatches) / seconds, stats.verified,
                 stats.pipeline_mismatches, stats.view_mismatches, stats.binding_mismatches);
    }
    std::string breaks;
    for (size_t reason = 0; reason < stats.breaks.size(); ++reason) {
        if (stats.breaks[reason] != 0) {
            breaks += fmt::format(" {} {:.0f}", VideoCommon::Nx2FastDraw::reason_names[reason],
                                  static_cast<double>(stats.breaks[reason]) / seconds);
        }
    }
    std::vector<std::pair<u32, u32>> methods;
    for (u32 method = 1; method < stats.break_methods.size(); ++method) {
        if (stats.break_methods[method] != 0) {
            methods.emplace_back(stats.break_methods[method], method);
        }
    }
    const size_t shown = std::min<size_t>(methods.size(), 12);
    std::partial_sort(methods.begin(), methods.begin() + shown, methods.end(), std::greater{});
    std::string top;
    for (size_t i = 0; i < shown; ++i) {
        top += fmt::format(" {:#x}:{:.0f}", methods[i].second,
                           static_cast<double>(methods[i].first) / seconds);
    }
    LOG_INFO(Render_Vulkan, "Fast draw state changes per second by last break:{} | by method:{}",
             breaks, top);
    stats = FastDrawStats{.window_start = now};
}

// Experimental adaptation of PokeDinner commit 76df4591a747: execute the reset shader's
// indirect record writes on the host until general GPU-address translation is available.
bool RasterizerVulkan::TryNx2IndirectReset(bool is_indexed, u32 instance_count) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return !value || std::string_view{value} != "0";
    }();
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    if (!enabled || is_indexed || instance_count != 1 || regs.IsMeshEnabled() ||
        draw.topology != Maxwell::PrimitiveTopology::Points ||
        !maxwell3d->GetNativeMacroEngine().native_turing) {
        return false;
    }
    const GPUVAddr program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    std::array<u32, VideoCommon::NativeIndirectReset::Program.size()> code{};
    if (!gpu_memory->IsFullyMappedRange(program + 128, sizeof(code))) {
        return false;
    }
    gpu_memory->ReadBlock(program + 128, code.data(), sizeof(code));
    if (!VideoCommon::NativeIndirectReset::Matches(code)) {
        return false;
    }
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::IndirectReset);
    FlushWork();
    gpu_memory->FlushCaching();
    const auto reject = [](std::string_view reason) {
        static u32 reported_rejections{};
        if (reported_rejections++ < 8) {
            LOG_WARNING(HW_GPU, "Experimental indirect reset skipped: {}", reason);
        }
        return false;
    };
    const auto in_gpu_range = [this](u64 address, size_t size) {
        return size != 0 && address <= (std::numeric_limits<u64>::max)() - size &&
               gpu_memory->IsWithinGPUAddressRange(address) &&
               gpu_memory->IsWithinGPUAddressRange(address + size - 1);
    };
    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[0];
    if (!cbuf.enabled || cbuf.size < 0x120 || !in_gpu_range(cbuf.address, 0x120) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address + 0x110, 16)) {
        return reject("descriptor unavailable");
    }
    std::array<u64, 2> descriptor{};
    gpu_memory->ReadBlock(cbuf.address + 0x110, descriptor.data(), sizeof(descriptor));
    const u64 list = descriptor[0];
    const u64 first = draw.vertex_buffer.first;
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(list, static_cast<u32>(descriptor[1]),
                                                        first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlock(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    for (const u64 pointer : pointers) {
        if (!pointer) {
            return reject("record is null");
        }
        if (!in_gpu_range(pointer, sizeof(record)) ||
            !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
            return reject("record not mapped");
        }
        if (pointer < range->begin + range->size && range->begin < pointer + sizeof(record)) {
            return reject("record overlaps pointer list");
        }
        const auto segments = gpu_memory->GetSubmappedRange(pointer, sizeof(record));
        if (segments.size() != 1 || segments[0].first != pointer ||
            segments[0].second != sizeof(record)) {
            return reject("record crosses a mapping boundary");
        }
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        AccelerateInlineToMemory(pointer, bytes.size(), bytes);
    }
    static u32 reported{};
    if (reported++ < 5) {
        LOG_INFO(HW_GPU, "Experimental indirect reset: {} records", count);
    }
    gpu.TickWork();
    return true;
}

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, u32 instance_count, Func&& draw_func,
                                   bool is_indirect) {

    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_draws;
    }

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            const auto& regs = maxwell3d->regs;
            const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                           regs.pipelines[1].reserved[1];
            const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                           regs.pipelines[5].reserved[1];
            LOG_DEBUG(HW_GPU, "Native graphics draw not recorded: frame={} vs={:#x} ps={:#x}",
                      render_trace_frame, vs, ps);
        }
    };
    namespace Profile = VideoCommon::Nx2Profile;
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    // Latched here, on the thread that draws: turning fast draws on starts with a full resolve,
    // and with them off nothing below runs.
    if (const bool on = FastDraw::Enabled(); on != fast_draw_on) {
        fast_draw_on = on;
        reusable_draw = {};
        FastDraw::g_delta = {};
        fast_draw_stats = {};
    }
    fast_draw_verify = fast_draw_on && FastDraw::VerifyEnabled();
    const u64 fast_draw_start = fast_draw_on ? Profile::Now() : 0;
    // EDEN_NX2_BACK_STAGES=1: stage B runs on its own thread for the draws of the split's back.
    if (back_staging_pool && VideoCommon::GpuSplit::OnBackThread()) {
        bool requires_inline = Settings::IsGPUFenceBehaviorAccurate();
#ifdef YUZU_LEGACY
        requires_inline |= Settings::values.vram_usage_mode.GetValue() ==
                           Settings::VramUsageMode::Aggressive;
#endif
        if (requires_inline) {
            if (back_ring) {
                StopBackStages();
            }
            static bool warned{};
            if (!std::exchange(warned, true)) {
                LOG_WARNING(Render_Vulkan,
                            "Back stages: selected fence/VRAM behavior requires inline draws");
            }
        } else if (!back_ring) {
            StartBackStages();
        }
    }
    if (back_ring) {
        // The ring stops on its own thread as the split's back exits (g_back_exit).
        ASSERT(std::this_thread::get_id() == back_owner);
        if (back_native_failed.exchange(false, std::memory_order_relaxed)) {
            maxwell3d->GetNativeMacroEngine().native_execution_failed = true;
        }
        // Stage B submitted on its own: A takes the state tracker's part of it.
        if (const u64 resets = scheduler.BackResets(); resets != back_resets_seen) {
            back_resets_seen = resets;
            state_tracker.InvalidateCommandBufferState();
        }
    }
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }
    if (frame_dump.Capturing()) {
        frame_dump.DrainIfFull();
    }
    // Stage A's recordings (texture uploads, copies, clears, render target changes) go to the
    // draw's command list; B splices it in before its own. A draw that stops early splices here.
    // With B on its own thread A always records into the packet slot it holds.
    if (!back_ring) {
        scheduler.BeginCommandList();
    }
    VideoCommon::DrawStages::Enter(VideoCommon::DrawStages::Stage::A);
    SCOPE_EXIT {
        if (!back_ring) {
            scheduler.SpliceCommandList();
        }
        VideoCommon::DrawStages::EndPacket();
    };

    const auto report_failure = [&](std::string_view reason) {
        TraceDrawFailure(reason);
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
    };
    GraphicsPipeline* const delta_pipeline =
        fast_draw_on ? TakeReusableDraw(is_indexed, is_indirect) : nullptr;
    // EDEN_NX2_FAST_DRAW_VERIFY: resolve in full and compare with what the delta path keeps.
    GraphicsPipeline* const verify_pipeline = fast_draw_verify ? delta_pipeline : nullptr;
    GraphicsPipeline* const pipeline{delta_pipeline && !verify_pipeline ? delta_pipeline : [&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    const bool verify_views = verify_pipeline != nullptr && verify_pipeline == pipeline;
    if (verify_pipeline != nullptr) {
        ++fast_draw_stats.verified;
        fast_draw_stats.pipeline_mismatches += verify_pipeline != pipeline ? 1 : 0;
    }
    if (!pipeline) {
        report_failure("no pipeline");
        TraceRenderState(is_indexed, "no-pipeline", nullptr);
        // Mesh emulation survey: how many task/mesh draws a device without
        // VK_EXT_mesh_shader drops, how many are indirect and how many groups they launch.
        if (maxwell3d->regs.IsMeshEnabled() && !device.IsExtMeshShaderSupported()) {
            static u64 draws{}, indirect{}, groups{}, max_groups{};
            ++draws;
            if (is_indirect) {
                ++indirect;
            } else {
                const u64 count{maxwell3d->draw_manager.draw_state.vertex_buffer.count};
                groups += count;
                max_groups = std::max(max_groups, count);
            }
            if (draws == 1 || draws % 5000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh emulation survey: {} mesh draws dropped, {} indirect, {} groups "
                            "in direct ones (max {} per draw), frame {}",
                            draws, indirect, groups, max_groups, render_trace_frame);
            }
        }
        return;
    }
    if (back_ring) {
        const VkDeviceSize bytes = pipeline->DescriptorBufferBytes();
        if (bytes != 0 && !descriptor_buffer_ring.Plan(bytes)) {
            const Scheduler::DirectScope stall{
                VideoCommon::DrawStages::Join::ADescriptorWrap};
            scheduler.Finish();
            descriptor_buffer_ring.Rewind();
            (void)descriptor_buffer_ring.Plan(bytes);
        }
    }
    if (VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        TrackOotTonemappedImage(*pipeline);
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    // EDEN_NX2_NO_MESH_FILE=<path>: skip them while that file exists (checked every 1024 draws),
    // to compare the same scene with and without them in one run.
    static const bool no_mesh_always{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    static const char* const no_mesh_file{std::getenv("EDEN_NX2_NO_MESH_FILE")};
    static bool no_mesh{no_mesh_always};
    static u32 no_mesh_check{};
    if (no_mesh_file && (no_mesh_check++ % 1024) == 0) {
        no_mesh = std::filesystem::exists(no_mesh_file);
    }
    const bool emulated_mesh{mesh_draw && pipeline->IsEmulatedMesh()};
    if (mesh_draw &&
        (no_mesh || (!device.IsExtMeshShaderSupported() && !emulated_mesh) ||
         (emulated_mesh && is_indirect))) {
        report_failure(no_mesh         ? "EDEN_NX2_NO_MESH"
                       : emulated_mesh ? "indirect emulated mesh draw"
                                       : "no VK_EXT_mesh_shader");
        TraceRenderState(is_indexed, "mesh-skipped", nullptr);
        return;
    }
    // Mesh emulation: one output block per mesh workgroup, written by the pipeline's compute
    // dispatch and drawn as groups * vertices-per-group triangle list vertices.
    mesh_emulation_groups = 0;
    mesh_emulation_vertices_per_group = 0;
    if (emulated_mesh) {
        // The same group count the native path hands vkCmdDrawMeshTasksEXT (MakeDrawParams:
        // the index count for indexed draws), within the device's dispatch limit.
        const auto& draw_state{maxwell3d->draw_manager.draw_state};
        const u32 groups{(std::min)(MakeDrawParams(draw_state, 1, is_indexed).num_vertices,
                                    device.GetMaxComputeWorkGroupCount()[0])};
        // Mesh emulation statistics: draws with and without groups, and the frames with any
        // groups (the game culls on the GPU and sets the counts itself).
        {
            static u64 draws{}, empty{}, total_groups{}, frames{}, frames_drawn{};
            static u64 last_frame{~0ULL};
            static bool frame_drawn{};
            if (render_trace_frame != last_frame) {
                if (last_frame != ~0ULL) {
                    ++frames;
                    frames_drawn += frame_drawn;
                }
                last_frame = render_trace_frame;
                frame_drawn = false;
            }
            ++draws;
            empty += groups == 0;
            total_groups += groups;
            frame_drawn |= groups != 0;
            if (draws % 2000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "Mesh emulation draws: {} ({} with no groups), {} groups; frames with "
                         "emulated draws: {} of {}",
                         draws, empty, total_groups, frames_drawn, frames);
            }
        }
        // Read the pending probe before Request can hand its buffer out again.
        if (mesh_probe.pending && scheduler.IsFree(mesh_probe.tick)) {
            ReadMeshEmulationProbe();
        }
        // EDEN_NX2_MESH_EMULATION_PROBE=1: every 60 frames, read one draw's output back (from a
        // host-visible download buffer, as device-local ones are not mapped on every GPU).
        static const bool probe{std::getenv("EDEN_NX2_MESH_EMULATION_PROBE") != nullptr};
        const bool has_task{pipeline->MeshEmulationHasTask()};
        const bool probe_draw{probe && !has_task && !mesh_probe.pending && groups != 0 &&
                              render_trace_frame % 60 == 0};
        // With a task program the draw's groups are task workgroups. The mesh workgroups they ask
        // for are only known on the GPU, so their output goes to the submission's arena.
        const u64 block_bytes{pipeline->MeshEmulationBlockBytes()};
        StagingBufferRef output{};
        u64 bytes{};
        if (has_task) {
            const MeshArena& arena{AcquireMeshArena()};
            bytes = arena.size;
            pipeline->SetMeshEmulationOutput(*arena.buffer, 0, bytes, groups);
        } else {
            bytes = (std::max)(u64{groups} * block_bytes, u64{4});
            output = staging_pool.Request(bytes, probe_draw ? MemoryUsage::Download
                                                            : MemoryUsage::DeviceLocal);
            pipeline->SetMeshEmulationOutput(output.buffer, output.offset, bytes, groups);
        }
        mesh_emulation_indirect = {};
        if (mesh_task_probe.pending && scheduler.IsFree(mesh_task_probe.tick)) {
            mesh_task_probe.pending = false;
            std::array<u32, Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS> words{};
            std::memcpy(words.data(), mesh_task_probe.args.data(), sizeof(words));
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation task probe: {} tasks asked for {} mesh workgroups, {} "
                     "dispatched at arena word {} (arena {} words)",
                     mesh_task_probe.tasks,
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_REQUESTED], words[0],
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_BASE],
                     mesh_task_probe.capacity);
        }
        if (has_task && groups != 0) {
            const bool task_probe{probe && !mesh_task_probe.pending &&
                                  render_trace_frame % 60 == 0};
            const auto request = [&](u64 size, MemoryUsage usage = MemoryUsage::DeviceLocal) {
                size = (std::max)(size, u64{16});
                const StagingBufferRef ref{staging_pool.Request(size, usage)};
                return std::make_pair(GraphicsPipeline::MeshEmulationBuffer{ref.buffer, ref.offset,
                                                                            size},
                                      ref.mapped_span.subspan(
                                          (std::min)(ref.offset, u64{ref.mapped_span.size()})));
            };
            const auto payload{
                request(u64{groups} * pipeline->MeshEmulationPayloadStride() * 4).first};
            const auto offsets{request(u64{groups} * 4).first};
            constexpr u64 args_bytes{Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS * 4};
            const auto [args, args_span] =
                request(args_bytes, task_probe ? MemoryUsage::Download : MemoryUsage::DeviceLocal);
            const u32 capacity_words{static_cast<u32>(
                (std::min)(bytes / 4, u64{(std::numeric_limits<u32>::max)()}))};
            if (task_probe && args_span.size() >= args_bytes) {
                mesh_task_probe = {
                    .args = args_span,
                    .tick = scheduler.CurrentTick(),
                    .tasks = groups,
                    .capacity = capacity_words,
                    .pending = true,
                };
            }
            const MeshArena& arena{mesh_arenas[mesh_arena_current]};
            pipeline->SetMeshEmulationTaskBuffers(
                payload, offsets, args, {*arena.counter, 0, sizeof(u32)}, capacity_words);
            mesh_emulation_indirect = {args.buffer, args.offset};
        }
        if (probe_draw && output.mapped_span.size() >= output.offset + bytes) {
            const MeshEmulationSetup& setup{*pipeline->MeshEmulation()};
            mesh_probe = {
                .data = output.mapped_span.subspan(output.offset, bytes),
                .tick = scheduler.CurrentTick(),
                .frame = render_trace_frame,
                .groups = groups,
                .block_words = setup.block_words,
                .vertex_lines = setup.vertex_lines,
                .max_vertices = setup.max_vertices,
                .max_primitives = setup.max_primitives,
                .position_lines = setup.position_lines,
                .pending = true,
            };
        } else if (probe_draw && output.mapped_span.empty()) {
            static bool warned{};
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan, "Mesh emulation probe: output buffer is not mapped");
            }
        }
        mesh_emulation_groups = groups;
        mesh_emulation_vertices_per_group = pipeline->MeshEmulationVerticesPerGroup();
    }
    // With stage B on its own thread A uses no buffer cache state: B holds the cache's lock
    // while it records a packet.
    std::unique_lock buffer_lock{buffer_cache.mutex, std::defer_lock};
    std::unique_lock texture_lock{texture_cache.mutex, std::defer_lock};
    if (back_ring) {
        texture_lock.lock();
    } else {
        std::lock(buffer_lock, texture_lock);
    }
    // What the next draw's TakeReusableDraw compares with, read once the caches are locked: a
    // break from here on, this draw's own resolve included, ends the delta chain.
    const u64 draw_serial = FastDraw::Serial();
    const u64 draw_map_serial = gpu_memory->MapSerial();
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    u32 repack_dirty;
    if (back_ring) {
        // B sets the runtime's layout from the packet.
        const u32 mask = buffer_cache_runtime.VertexRepackMaskFor(maxwell3d->regs,
                                                                  pipeline->HasDynamicVertexInput());
        repack_dirty = back_repack_mask ^ mask;
        back_repack_mask = mask;
    } else {
        repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
            maxwell3d->regs, pipeline->HasDynamicVertexInput());
    }
    if (repack_dirty) {
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        flags[Dirty::VertexInput] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if (repack_dirty & (1U << binding)) {
                flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    const bool delta_holds =
        delta_pipeline != nullptr && DeltaStillHolds(draw_serial, draw_map_serial);
    const DrawPath path = delta_holds && !verify_pipeline ? DrawPath::Delta : DrawPath::Full;
    // Stage A: everything that reads the registers, dirty flags and memory as they are at the
    // draw goes into the packet.
    DrawPacket& packet = back_ring ? back_slot->packet : draw_packet;
    packet.sequence = ++draw_sequence;
    packet.vertex_repack_mask = back_repack_mask;
    packet.pipeline = pipeline;
    packet.is_indexed = is_indexed;
    packet.is_indirect = is_indirect;
    packet.buffers.Reset(is_indexed);
    packet.dynamic.mask = 0;
    // What the delta draw would have kept: the texture buffer bindings.
    const u64 kept_bindings =
        verify_views ? buffer_cache.KeptGraphicsBindingsHash(false, false, true) : 0;
    buffer_cache.BeginGraphicsUniforms(path == DrawPath::Delta ? &taken_delta : nullptr);
    if (path == DrawPath::Delta) {
        if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->PrepareDelta(packet)) {
            report_failure("configure failed");
            TraceRenderState(is_indexed, "configure-failed", nullptr);
            return;
        }
        ++fast_draw_stats.delta;
    } else {
        if (fast_draw_on) {
            pipeline->CaptureNextDelta(verify_views);
        }
        if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Prepare(packet)) {
            report_failure("configure failed");
            TraceRenderState(is_indexed, "configure-failed", nullptr);
            return;
        }
        if (verify_views && pipeline->DeltaMismatch()) {
            ++fast_draw_stats.view_mismatches;
        }
    }
    packet.mesh_draw = mesh_draw;
    if (mesh_draw) {
        packet.mesh_emulation_groups = mesh_emulation_groups;
        packet.mesh_emulation_vertices_per_group = mesh_emulation_vertices_per_group;
        packet.mesh_emulation_indirect_buffer = mesh_emulation_indirect.buffer;
        packet.mesh_emulation_indirect_offset = mesh_emulation_indirect.offset;
        packet.mesh_dispatch = pipeline->MeshDispatch();
    }
    if (is_indirect) {
        packet.indirect = maxwell3d->draw_manager.indirect_state;
    } else {
        packet.params =
            MakeDrawParams(maxwell3d->draw_manager.draw_state, instance_count, is_indexed);
    }
    packet.zpass_pixel_count = maxwell3d->regs.zpass_pixel_count_enable != 0;
    packet.query_tfb = TransformFeedbackQueryState::FromRegs(
        maxwell3d->regs, maxwell3d->draw_manager.draw_state.topology);
    if (packet.buffers.transform_feedback && device.IsExtTransformFeedbackSupported()) {
        UNIMPLEMENTED_IF(maxwell3d->regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         maxwell3d->regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
    // B binds the pipeline after A took the dynamic state: when that bind makes the state enables
    // dirty again, A takes them now so B records them with the draw, as before the split.
    if (back_ring ? back_view.RefreshesStateEnable(pipeline)
                  : scheduler.RefreshesStateEnable(pipeline)) {
        state_tracker.InvalidateStateEnableFlag();
    }
    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        UpdateOcarinaJitter();
        UpdateDynamicStates(pipeline, packet.dynamic);
    }
    const u64 prepared_tick = scheduler.CurrentTick();
    if (DrawPacketVerify()) {
        VerifyDrawPacket(packet);
    }
    packet.descriptor_data = guest_descriptor_queue.DrawEntries();
    packet.descriptor_count = static_cast<u32>(guest_descriptor_queue.UpdateCount());

    // What stage A does once the draw is recorded or handed over.
    const auto finish_a = [&] {
        constexpr u64 final_vertex = 0x16aff33d999f993fULL;
        constexpr u64 final_fragment = 0x52402d8e8ab55263ULL;
        const auto hashes = pipeline->ShaderHashes();
        if (oot_4k_scene_frame == render_trace_frame && hashes[5] == final_fragment &&
            std::ranges::find(hashes, final_vertex) != hashes.end()) {
            const auto target_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
            if (target_id) {
                const auto& target = texture_cache.GetImageView(target_id);
                if (target.size.width == 2560 && target.size.height == 1440 &&
                    target.format == VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB) {
                    oot_4k_guest_present = target.ImageHandle();
                }
            }
        }
        // The game draws Navi's glow, fireflies and dust right after the fog. They write no depth
        // and move on their own, so FSR must not take them from its history. DLSS doesn't use the
        // mask.
        constexpr u64 fog_fragment = 0x3ddfb1d7a854c849ULL;
        if (hashes[5] == fog_fragment && oot_live_dlss && OcarinaTemporalEnabled() &&
            oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr &&
            oot_camera_sample_frame == render_trace_frame && !frame_dump.Capturing()) {
            const auto scene_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
            if (scene_id) {
                const auto& scene = texture_cache.GetImageView(scene_id);
                if (scene.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                    scene.size.width == oot_motion.extent.width &&
                    scene.size.height == oot_motion.extent.height && !scene.IsRescaled() &&
                    scene.range == VideoCommon::SubresourceRange{}) {
                    oot_effects_scene = scene.Handle(Shader::TextureType::Color2D);
                    oot_effects_address = scene.gpu_addr;
                    oot_effects_frame = render_trace_frame;
                    oot_live_dlss->SaveEffectsBase(oot_effects_scene, oot_motion.extent,
                                                   render_trace_frame);
                }
            }
        }
        if (fast_draw_on) {
            NoteResolvedDraw(pipeline, is_indexed, is_indirect, draw_serial, draw_map_serial);
            const size_t index = static_cast<size_t>(path);
            fast_draw_stats.prepare_ns[index] += Profile::Now() - fast_draw_start;
            ++fast_draw_stats.prepare_draws[index];
        }
    };
    // With stage B on its own thread A hands the packet over, unless a diagnostic reads what B
    // records or binds: that draw runs both stages here once B is idle (cmdlist-plan ruling 7).
    std::optional<Scheduler::DirectScope> direct;
    if (back_ring) {
        const bool diagnostics = [&] {
            static const bool pending_write_trace =
                std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
            static const bool indirect_readback = [] {
                const char* value = std::getenv("EDEN_DKB_INDIRECT_READBACK");
                return value && std::string_view{value} == "1";
            }();
            static const bool native_capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return OcarinaTemporalEnabled() || verify_views || pending_write_trace || frame_dump.Capturing() ||
                   VideoCommon::RenderTrace::SampleFrame(render_trace_frame) ||
                   (is_indirect && indirect_readback && Common::DkbFrameTrace::Enabled()) ||
                   (native_capture && maxwell3d->GetNativeMacroEngine().native_turing);
        }();
        if (!diagnostics) {
            if (Common::DkbFrameTrace::Enabled() && !is_indirect &&
                (packet.params.num_vertices == 0 ||
                 (!packet.mesh_draw && packet.params.num_instances == 0))) {
                ++trace_zero_direct;
            }
            CommitBackSlot(is_indirect);
            submitted = true;
            finish_a();
            return;
        }
        direct.emplace(VideoCommon::DrawStages::Join::Diagnostics);
        buffer_lock.lock();
        buffer_cache_runtime.SetVertexRepackMask(packet.vertex_repack_mask);
    }

    // Stage B: records the draw from the packet alone.
    scheduler.SpliceCommandList();
    VideoCommon::DrawStages::Enter(VideoCommon::DrawStages::Stage::B);
    if (Profile::Scope scope{Profile::Zone::Record}; !pipeline->Record(packet)) {
        // The dirty flags A took stay recorded for the next draw.
        if (back_ring) {
            MergeDynamicState(back_dynamic, packet.dynamic);
        }
        RecordDynamicStates(packet.dynamic);
        report_failure("configure failed");
        TraceRenderState(is_indexed, "configure-failed", nullptr);
        return;
    }
    if (verify_views &&
        kept_bindings != buffer_cache.KeptGraphicsBindingsHash(false, false, true)) {
        ++fast_draw_stats.binding_mismatches;
    }

    TrackOcarinaCamera();
    TrackOcarinaActor(pipeline->ShaderHashes()[1], is_indexed, is_indirect);
    TracePendingCpuWrites(false, pipeline->ShaderHashes());
    TraceRenderState(is_indexed, "configured", packet.framebuffer);
    if (frame_dump.Capturing()) {
        FrameDumpPass(packet.framebuffer);
        frame_dump.NoteDraw(pipeline->ShaderHashes());
        bool recorded = false;
        if (frame_dump.CapturingSampled()) {
            boost::container::small_vector<FrameDump::Sampled, 16> sampled;
            for (const auto& image : pipeline->SampledImages()) {
                sampled.push_back({image.stage, image.tic, &texture_cache.GetImageView(image.id),
                                   &texture_cache.GetSampler(image.sampler)});
            }
            recorded = frame_dump.NoteSampled(sampled);
        }
        if (frame_dump.CapturingBuffers()) {
            recorded |= FrameDumpBuffers(false, pipeline->ShaderHashes());
        }
        if (recorded) {
            scheduler.RequestRenderpass(packet.framebuffer);
        }
    }
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace pipeline: frame={} draw={} hashes={:016x} dynamic_vertex_input={}",
                  render_trace_frame, render_trace_draws, fmt::join(pipeline->ShaderHashes(), ","),
                  pipeline->HasDynamicVertexInput());
    }

    if (is_indirect && !packet.mesh_draw && Common::DkbFrameTrace::Enabled()) {
        static const bool readback = [] {
            const char* value = std::getenv("EDEN_DKB_INDIRECT_READBACK");
            return value && std::string_view{value} == "1";
        }();
        if (readback) {
            CaptureIndirectTrace();
        }
    }

    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        // A submit since A reset the command buffer's dynamic state: take all of it again.
        if (scheduler.CurrentTick() != prepared_tick) {
            const VideoCommon::DrawStages::JoinScope join{
                VideoCommon::DrawStages::Join::BDynamicState};
            UpdateDynamicStates(pipeline, packet.dynamic);
        }
        if (back_ring) {
            MergeDynamicState(back_dynamic, packet.dynamic);
        }
        RecordDynamicStates(packet.dynamic);
    }

    query_cache.NotifySegment(true);
    HandleTransformFeedback(packet.buffers.transform_feedback);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, packet.zpass_pixel_count);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        scheduler.TraceGraphicsPipeline(pipeline->ShaderHashes());
        trace_draw_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        draw_func(packet);
    }
    submitted = true;
    buffer_cache.NoteWorkRecorded(false);
    finish_a();
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}


namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (!trace || regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    {
        // A draw's first and count can come from memory a skipped (rejected) compute program
        // should have written. Past 16M vertices they are garbage: building the quad index
        // buffer for one threw out of the GPU thread and ended the process (Hyrule Warriors:
        // Age of Imprisonment, entering a new area), and indexed ones read far out of bounds.
        const auto& state = maxwell3d->draw_manager.draw_state;
        const u64 first = is_indexed ? state.index_buffer.first : state.vertex_buffer.first;
        const u64 count = is_indexed ? state.index_buffer.count : state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (first + count > MaxVertices) {
            static std::atomic<u32> skipped{};
            if (skipped.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU, "Draw skipped: {} {} from {} (vertex/index count past {})",
                            count, is_indexed ? "indices" : "vertices", first, MaxVertices);
            }
            return;
        }
    }
    if (TryNx2IndirectReset(is_indexed, instance_count)) {
        return;
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if (vs == trace_vs && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    // A direct draw with no vertices, or no instances outside task/mesh pipelines, records
    // nothing. Skip it before the pipeline, buffer and descriptor work: dirty state, deferred
    // clears and query counter toggles stay pending for the next recorded draw, clear or render
    // pass end, which apply them before anything can observe them. With transform feedback on
    // the draw goes through, since its counters are begun and closed around draws.
    if (maxwell3d->regs.transform_feedback_enabled == 0 &&
        (MakeDrawParams(maxwell3d->draw_manager.draw_state, instance_count, is_indexed)
                 .num_vertices == 0 ||
         (instance_count == 0 && [this] {
             const GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
             return pipeline != nullptr && !pipeline->IsMeshPipeline();
         }()))) {
        if (Common::DkbFrameTrace::Enabled()) {
            ++trace_draws;
            ++trace_zero_direct;
        }
        gpu.TickWork();
        return;
    }
    PrepareDraw(is_indexed, instance_count,
                [this](const DrawPacket& packet) { RecordDirectDraw(packet); });
}

void RasterizerVulkan::DrawIndirect() {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_indirect;
    }
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(
        params.is_indexed, 0, [this](const DrawPacket& packet) { RecordIndirectDraw(packet); },
        true);
    buffer_cache.SetDrawIndirect(nullptr);
}

void RasterizerVulkan::RecordDirectDraw(const DrawPacket& packet) {
    const DrawParams& draw_params = packet.params;
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace direct: frame={} draw={} count={} instances={} first_index={} "
                  "base_vertex={} base_instance={} mesh={}",
                  render_trace_frame, render_trace_draws, draw_params.num_vertices,
                  draw_params.num_instances, draw_params.first_index, draw_params.base_vertex,
                  draw_params.base_instance, packet.mesh_draw);
    }

    if (Common::DkbFrameTrace::Enabled() &&
        !VideoCommon::DrawStages::t_stage_b_thread &&
        (draw_params.num_vertices == 0 || (!packet.mesh_draw && draw_params.num_instances == 0))) {
        ++trace_zero_direct;
    }

    if (packet.mesh_draw && packet.mesh_emulation_vertices_per_group != 0) {
        const u32 vertices{packet.mesh_emulation_groups * packet.mesh_emulation_vertices_per_group};
        if (vertices == 0) {
            return;
        }
        if (packet.mesh_emulation_indirect_buffer) {
            // With a task program the distribution pass wrote the vertex count.
            scheduler.Record([this, buffer = packet.mesh_emulation_indirect_buffer,
                              offset = packet.mesh_emulation_indirect_offset](
                                 vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawIndirect(buffer,
                                    offset +
                                        Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET,
                                    1, 0);
            });
            return;
        }
        scheduler.Record([this, vertices](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            cmdbuf.Draw(vertices, 1, 0, 0);
        });
        return;
    }
    if (packet.mesh_draw) {
        scheduler.Record([this, groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
        });
        return;
    }
    scheduler.Record([this, draw_params, hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
        if (scheduler.GraphicsPipelineMissing()) {
            return;
        }
        scheduler.TraceGpuWork(
            cmdbuf, hashes.first, hashes.second,
            {draw_params.num_vertices, draw_params.num_instances, 0}, [&] {
                if (draw_params.is_indexed) {
                    cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                       draw_params.first_index, draw_params.base_vertex,
                                       draw_params.base_instance);
                } else {
                    cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                                draw_params.base_vertex, draw_params.base_instance);
                }
            });
    });

    // Log draw call
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = packet.is_indexed ?
            fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                draw_params.num_vertices, draw_params.num_instances,
                draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
            fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                draw_params.num_vertices, draw_params.num_instances,
                draw_params.base_vertex, draw_params.base_instance);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            packet.is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::RecordIndirectDraw(const DrawPacket& packet) {
    const auto& params = packet.indirect;
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace indirect: frame={} draw={} address={:#x} count_address={:#x} "
                  "bytes={} max_draws={} stride={} include_count={} byte_count={} mesh={}",
                  render_trace_frame, render_trace_draws, params.indirect_start_address,
                  params.count_start_address, params.buffer_size, params.max_draw_counts,
                  params.stride, params.include_count, params.is_byte_count, packet.mesh_draw);
    }
    if (packet.mesh_draw) {
        TraceDrawFailure("indirect task/mesh unsupported");
        // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
        static bool logged{};
        if (!std::exchange(logged, true)) {
            LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
        }
        return;
    }
    const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
    const auto& buffer = indirect_buffer.first;
    const auto& offset = indirect_buffer.second;
    if (params.is_byte_count) {
        scheduler.Record([this, buffer_obj = buffer->Handle(), offset,
                          stride = params.stride](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                            static_cast<u32>(stride));
        });
        return;
    }
    if (params.include_count) {
        const auto count = buffer_cache.GetDrawIndirectCount();
        const auto& draw_buffer = count.first;
        const auto& offset_base = count.second;
        scheduler.Record([this, draw_buffer_obj = draw_buffer->Handle(),
                          buffer_obj = buffer->Handle(), offset_base, offset,
                          params](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirectCount(
                    buffer_obj, offset, draw_buffer_obj, offset_base,
                    static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                         static_cast<u32>(params.max_draw_counts),
                                         static_cast<u32>(params.stride));
            }
        });
        return;
    }
    scheduler.Record([this, buffer_obj = buffer->Handle(), offset, params,
                      hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
        if (scheduler.GraphicsPipelineMissing()) {
            return;
        }
        scheduler.TraceGpuWork(cmdbuf, hashes.first, hashes.second, {}, [&] {
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                           static_cast<u32>(params.max_draw_counts),
                                           static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirect(buffer_obj, offset,
                                    static_cast<u32>(params.max_draw_counts),
                                    static_cast<u32>(params.stride));
            }
        });
    });

    // Log indirect draw call
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string log_params = fmt::format("drawCount={}, stride={}",
            params.max_draw_counts, params.stride);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
            log_params, VK_SUCCESS);
    }
}

namespace {
/// EDEN_NX2_BACK_STAGES_VERIFY: the bytes of a range B reads, its first 256 KiB.
template <typename Memory>
u64 HashDeviceRange(Memory& memory, u64 begin, u64 size) {
    static constexpr u64 MaxBytes = 256 * 1024;
    thread_local std::vector<u8> scratch;
    scratch.resize(static_cast<size_t>((std::min)(size, MaxBytes)));
    memory.ReadBlockUnsafe(begin, scratch.data(), scratch.size());
    return Common::CityHash64(reinterpret_cast<const char*>(scratch.data()), scratch.size());
}
} // Anonymous namespace

bool RasterizerVulkan::BackView::RefreshesStateEnable(const GraphicsPipeline* next) const {
    return next != nullptr && needs_state_enable_refresh && next->UsesExtendedDynamicState();
}

void RasterizerVulkan::BackView::Bind(const GraphicsPipeline* next, const Framebuffer* target) {
    if (target != framebuffer) {
        framebuffer = target;
        ++render_pass;
    }
    changes = Scheduler::AViewChanges();
    // As Scheduler::UpdateGraphicsPipeline leaves it.
    if (next == pipeline) {
        if (next->UsesExtendedDynamicState()) {
            needs_state_enable_refresh = false;
        }
        return;
    }
    pipeline = next;
    needs_state_enable_refresh = !next->UsesExtendedDynamicState();
}

void RasterizerVulkan::StartBackStages() {
    namespace DrawStages = VideoCommon::DrawStages;
    back_owner = std::this_thread::get_id();
    back_slots = std::make_unique<BackSlot[]>(DrawStages::BackRing::Capacity);
    back_ring = std::make_unique<DrawStages::BackRing>();
    back_view = {};
    back_held.fill({});
    back_active.clear();
    back_read_active.fill(false);
    back_repack_mask = buffer_cache_runtime.RepackedVertexBuffers();
    back_resets_seen = scheduler.BackResets();
    back_b_resets = back_resets_seen;
    back_dynamic = {};
    back_verify = {};
    descriptor_buffer_ring.PrepareFrame(scheduler);
    // The first packets carry every dynamic state group, B's running copy starts from them.
    state_tracker.InvalidateCommandBufferState();
    DrawStages::g_threaded_verify = DrawStages::BackStagesVerifyRequested();
    DrawStages::g_after_join_argument = this;
    DrawStages::g_after_join = [](void* rasterizer) {
        static_cast<RasterizerVulkan*>(rasterizer)->SpliceCapturedCommands();
    };
    DrawStages::g_back_exit_argument = this;
    DrawStages::g_back_exit = [](void* rasterizer) {
        static_cast<RasterizerVulkan*>(rasterizer)->StopBackStages();
    };
    back_ring->Start([this](u32 index) { ConsumeBackSlot(index); },
                     [](u64 position) { VideoCommon::GpuSplit::Get().NoteStageBConsumed(position); },
                     {});
    DrawStages::g_ring = back_ring.get();
    DrawStages::g_threaded = true;
    DrawStages::t_draw_thread = true;
    DrawStages::ClaimRing();
    back_slot = &back_slots[back_ring->Acquire()];
    Scheduler::CaptureInto(&back_slot->list);
    LOG_INFO(Render_Vulkan, "Draw packets: stage B runs on its own thread{}",
             DrawStages::g_threaded_verify ? " (EDEN_NX2_BACK_STAGES_VERIFY)" : "");
}

void RasterizerVulkan::StopBackStages() {
    namespace DrawStages = VideoCommon::DrawStages;
    // On the thread that drew (the split's back as it exits): B records every packet, then that
    // thread's capture and the ring's hooks are cleared before anything else records.
    back_ring->Join();
    if (back_native_failed.exchange(false, std::memory_order_relaxed)) {
        maxwell3d->GetNativeMacroEngine().native_execution_failed = true;
    }
    DrawStages::ReleaseRing();
    DrawStages::g_threaded = false;
    DrawStages::g_threaded_verify = false;
    DrawStages::g_ring = nullptr;
    DrawStages::g_after_join = nullptr;
    DrawStages::g_back_exit = nullptr;
    Scheduler::CaptureInto(nullptr);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        scheduler.SpliceList(back_slot->list);
        // B bound the graphics uniform buffers from the packets: the bindings' buffers were
        // not kept up while it ran.
        buffer_cache.ForgetGraphicsUniformBuffers();
    }
    back_ring->Stop();
    back_ring.reset();
    back_slots.reset();
    back_slot = nullptr;
    state_tracker.InvalidateCommandBufferState();
    LOG_INFO(Render_Vulkan, "Draw packets: stage B stopped with the split's back");
}

void RasterizerVulkan::CommitBackSlot(bool indirect) {
    BackSlot& slot = *back_slot;
    slot.indirect = indirect;
    slot.native_turing = maxwell3d->GetNativeMacroEngine().native_turing;
    slot.committed = back_ring->Committed();
    slot.map_serial = gpu_memory->MapSerial();
    NoteBackReads(slot);
    back_view.Bind(slot.packet.pipeline, slot.packet.framebuffer);
    back_ring->Commit();
    back_slot = &back_slots[back_ring->Acquire()];
    Scheduler::CaptureInto(&back_slot->list);
}

void RasterizerVulkan::SpliceCapturedCommands() {
    if (back_native_failed.exchange(false, std::memory_order_relaxed)) {
        maxwell3d->GetNativeMacroEngine().native_execution_failed = true;
    }
    if (back_splicing || back_slot == nullptr || back_slot->list.Empty()) {
        return;
    }
    back_splicing = true;
    scheduler.SpliceList(back_slot->list);
    back_splicing = false;
}

void RasterizerVulkan::NoteBackReads(BackSlot& slot) {
    namespace DrawStages = VideoCommon::DrawStages;
    const auto& buffers = slot.packet.buffers;
    const auto hold = [this](u32 read_slot, u64 begin, u64 size) {
        auto& held = back_held[read_slot];
        if (!back_read_active[read_slot] && begin != 0 && size != 0) {
            back_active.push_back(read_slot);
            back_read_active[read_slot] = true;
        }
        held = begin != 0 ? std::pair{begin, size} : std::pair<u64, u64>{};
    };
    if (buffers.index_dirty) {
        hold(DrawStages::IndexSlot, buffers.index.device_addr, buffers.index.size);
    }
    for (const auto& stream : buffers.vertex_streams) {
        if ((buffers.vertex_resolve >> stream.index) & 1) {
            hold(DrawStages::VertexSlot(stream.index), stream.binding.device_addr,
                 stream.binding.size);
        }
    }
    for (u32 stage = 0; stage < DrawPacket::NUM_STAGES; ++stage) {
        if ((buffers.storage_stages >> stage) & 1) {
            for (u32 index = 0; index < 64; ++index) {
                back_held[DrawStages::StorageSlot(stage, index)] = {};
            }
        }
        if ((buffers.texture_stages >> stage) & 1) {
            for (u32 index = 0; index < 32; ++index) {
                back_held[DrawStages::TexelSlot(stage, index)] = {};
            }
        }
    }
    for (const auto& buffer : buffers.storage_buffers) {
        hold(DrawStages::StorageSlot(buffer.stage, buffer.index), buffer.binding.device_addr,
             buffer.binding.size);
    }
    for (const auto& buffer : buffers.texture_buffers) {
        hold(DrawStages::TexelSlot(buffer.stage, buffer.index), buffer.binding.device_addr,
             buffer.binding.size);
    }
    for (u32 order = 0; order < DrawStages::UniformSlots; ++order) {
        back_held[DrawStages::UniformSlot(order)] = {};
    }
    u32 order = 0;
    for (const auto& bind : buffers.uniform_binds) {
        hold(DrawStages::UniformSlot(order++), bind.device_addr, bind.size);
    }
    hold(DrawStages::IndirectSlot, buffers.indirect ? buffers.indirect_buffer.device_addr : 0,
         buffers.indirect ? buffers.indirect_buffer.size : 0);
    hold(DrawStages::IndirectCountSlot,
         buffers.indirect_count ? buffers.count_buffer.device_addr : 0,
         buffers.indirect_count ? buffers.count_buffer.size : 0);
    std::erase_if(back_active, [this](u32 read_slot) {
        if (back_held[read_slot].second != 0) {
            return false;
        }
        back_read_active[read_slot] = false;
        return true;
    });
    const bool verify = DrawStages::g_threaded_verify;
    slot.reads.clear();
    if (verify) {
        slot.committed_at = std::chrono::steady_clock::now();
    }
    const auto fresh = [&buffers](u32 read_slot) {
        if (read_slot == DrawStages::IndexSlot) {
            return buffers.index_dirty;
        }
        if (read_slot < DrawStages::StorageSlot(0, 0)) {
            return ((buffers.vertex_resolve >> (read_slot - DrawStages::VertexSlot(0))) & 1) != 0;
        }
        if (read_slot < DrawStages::TexelSlot(0, 0)) {
            return std::ranges::any_of(buffers.storage_buffers, [read_slot](const auto& buffer) {
                return DrawStages::StorageSlot(buffer.stage, buffer.index) == read_slot;
            });
        }
        if (read_slot < DrawStages::UniformSlot(0)) {
            return std::ranges::any_of(buffers.texture_buffers, [read_slot](const auto& buffer) {
                return DrawStages::TexelSlot(buffer.stage, buffer.index) == read_slot;
            });
        }
        return true;
    };
    for (const u32 read_slot : back_active) {
        const auto [begin, size] = back_held[read_slot];
        DrawStages::NoteReadAt(read_slot, begin, size, slot.committed);
        if (verify) {
            const u32 order = read_slot - DrawStages::UniformSlot(0);
            const bool streamed = read_slot >= DrawStages::UniformSlot(0) &&
                                  read_slot < DrawStages::IndirectSlot &&
                                  order < buffers.uniform_binds.size() &&
                                  buffers.uniform_binds[order].streamed;
            slot.reads.push_back({begin, size, HashDeviceRange(device_memory, begin, size),
                                  read_slot, fresh(read_slot), streamed});
        }
    }
}

void RasterizerVulkan::VerifyBackReads(const BackSlot& slot) {
    auto& stats = back_verify;
    ++stats.packets;
    if (slot.committed != stats.next) {
        ++stats.out_of_order;
        if (stats.logged++ < 16) {
            LOG_ERROR(Render_Vulkan, "Back stages verify: packet {} recorded after packet {}",
                      slot.committed, stats.next - 1);
        }
    }
    stats.next = slot.committed + 1;
    // B reads by device address, as A resolved it at the draw. A change of the address space
    // since is counted: only a change of the bytes below is a difference.
    stats.remapped += slot.map_serial != gpu_memory->MapSerial() ? 1 : 0;
    for (const auto& read : slot.reads) {
        const u64 begin = read.begin;
        const u64 size = read.size;
        ++stats.ranges;
        if (HashDeviceRange(device_memory, begin, size) == read.hash) {
            continue;
        }
        if (VideoCommon::DrawStages::HostCachedVertexRead(
                read.slot, buffer_cache.IsRegionCpuModified(begin, size))) {
            if (stats.vertex_cached++ < 4) {
                LOG_INFO(Render_Vulkan,
                         "Back stages cached vertex: packet {} (draw {}) slot {} (vertex {}) "
                         "{:#x}+{:#x} changed after stage A with no CPU-modified page at B",
                         slot.committed, slot.packet.sequence, read.slot,
                         read.slot - VideoCommon::DrawStages::VertexSlot(0), begin, size);
            }
            continue;
        }
        ++stats.differing;
        if (stats.logged++ < 16) {
            LOG_ERROR(Render_Vulkan,
                      "Back stages verify: packet {} (draw {}) reads {:#x}+{:#x}, which changed "
                      "after stage A built it",
                      slot.committed, slot.packet.sequence, begin, size);
            namespace DrawStages = VideoCommon::DrawStages;
            const u32 s = read.slot;
            const std::string kind =
                s == DrawStages::IndexSlot ? std::string{"index"}
                : s < DrawStages::StorageSlot(0, 0)
                    ? fmt::format("vertex {}", s - DrawStages::VertexSlot(0))
                : s < DrawStages::TexelSlot(0, 0)
                    ? fmt::format("storage stage {} index {}", (s - DrawStages::StorageSlot(0, 0)) / 64,
                                  (s - DrawStages::StorageSlot(0, 0)) % 64)
                : s < DrawStages::UniformSlot(0)
                    ? fmt::format("texel stage {} index {}", (s - DrawStages::TexelSlot(0, 0)) / 32,
                                  (s - DrawStages::TexelSlot(0, 0)) % 32)
                : s < DrawStages::IndirectSlot
                    ? fmt::format("uniform bind {}", s - DrawStages::UniformSlot(0))
                : s == DrawStages::IndirectSlot ? std::string{"indirect"}
                                                : std::string{"indirect count"};
            const auto& buffers = slot.packet.buffers;
            LOG_ERROR(Render_Vulkan,
                      "Back stages verify: that range is slot {} ({}), set by {}, streamed {}; "
                      "draw indexed {} (index format {}, {} bytes, index set by this packet {}), "
                      "indirect {}; at B the range is CPU-modified {}, GPU-modified {} (anywhere "
                      "in it); packet age {} us",
                      s, kind, read.fresh ? "this packet" : "an earlier packet", read.streamed,
                      buffers.is_indexed, static_cast<u32>(buffers.index_format),
                      buffers.index_format_size, buffers.index_dirty, slot.indirect,
                      buffer_cache.IsRegionCpuModified(begin, size),
                      buffer_cache.IsRegionGpuModified(begin, size),
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - slot.committed_at)
                          .count());
        }
    }
}

void RasterizerVulkan::ConsumeBackSlot(u32 index) {
    std::scoped_lock lock{buffer_cache.mutex};
    RecordBackDraw(back_slots[index]);
}

void RasterizerVulkan::RecordBackDraw(BackSlot& slot) {
    namespace DrawStages = VideoCommon::DrawStages;
    namespace Profile = VideoCommon::Nx2Profile;
    const DrawPacket& packet = slot.packet;
    query_cache_runtime.SetStageTransformFeedbackState(&packet.query_tfb);
    SCOPE_EXIT {
        query_cache_runtime.SetStageTransformFeedbackState(nullptr);
    };
    if (DrawStages::g_threaded_verify) {
        VerifyBackReads(slot);
    }
    scheduler.SpliceList(slot.list);
    GraphicsPipeline* const pipeline = packet.pipeline;
    buffer_cache_runtime.SetVertexRepackMask(packet.vertex_repack_mask);
    const bool recorded = [&] {
        Profile::Scope scope{Profile::Zone::Record};
        return pipeline->Record(packet);
    }();
    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        // A submit of B's own lost the command buffer's dynamic state, A's packets carry only
        // what changed: record B's running copy of all of it.
        MergeDynamicState(back_dynamic, packet.dynamic);
        const u64 resets = scheduler.BackResets();
        if (resets != back_b_resets) {
            back_b_resets = resets;
            ++back_verify.b_submits;
            RecordDynamicStates(back_dynamic);
        } else {
            RecordDynamicStates(packet.dynamic);
        }
    }
    if (recorded) {
        query_cache.NotifySegment(true);
        HandleTransformFeedback(packet.buffers.transform_feedback);
        query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                  packet.zpass_pixel_count);
        Profile::Scope scope{Profile::Zone::DrawRecord};
        scheduler.TraceGraphicsPipeline(pipeline->ShaderHashes());
        trace_draw_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        if (slot.indirect) {
            RecordIndirectDraw(packet);
        } else {
            RecordDirectDraw(packet);
        }
        buffer_cache.NoteWorkRecorded(false);
    } else {
        TraceDrawFailure("configure failed");
        if (slot.native_turing) {
            back_native_failed.store(true, std::memory_order_relaxed);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - back_verify.last_report < std::chrono::seconds{10}) {
        return;
    }
    back_verify.last_report = now;
    const auto& stats = back_ring->stats;
    LOG_INFO(Render_Vulkan,
             "Back stages: {} packets in {} hand-overs, {} joins and {} full-ring waits ({} ms "
             "waited by A), B slept {} times and was busy {} ms; {} submits of B's own, {} draws "
             "skipped for a full descriptor buffer frame",
             stats.packets.load(), stats.batches.load(), stats.joins.load(),
             stats.full_waits.load(), stats.a_wait_ns.load() / 1'000'000, stats.b_sleeps.load(),
             stats.b_busy_ns.load() / 1'000'000, back_verify.b_submits,
             descriptor_buffer_ring.FullOnStageB());
    if (DrawStages::g_threaded_verify) {
        LOG_INFO(Render_Vulkan,
                 "Back stages verify: {} packets, {} ranges checked, {} changed after A built "
                 "the packet, {} out of order; {} packets recorded after the address space "
                 "changed; {} vertex ranges changed with no CPU-modified page at B (host-cached)",
                 back_verify.packets, back_verify.ranges, back_verify.differing,
                 back_verify.out_of_order, back_verify.remapped, back_verify.vertex_cached);
    }
}

VideoCommon::OotNetwork::BypassState* RasterizerVulkan::OotNetworkBypass() {
    // Temporal upscaling replaces the same output pass and takes precedence until it fails.
    if (OcarinaTemporalEnabled() && !(oot_live_dlss && oot_live_dlss->EvaluationFailed())) {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_INFO(Render_Vulkan, "Ocarina of Time: No sharpening is off while temporal upscaling is on");
        }
        return nullptr;
    }
    return &oot_network_bypass;
}

void RasterizerVulkan::TrackOotTonemappedImage(const GraphicsPipeline& pipeline) {
    const auto& hashes = pipeline.ShaderHashes();
    if (std::none_of(hashes.begin(), hashes.end(), VideoCommon::OotNetwork::IsTonemapShader)) {
        return;
    }
    const auto& target = maxwell3d->regs.rt[0];
    if (target.Address() == 0) {
        return;
    }
    std::scoped_lock lock{oot_network_bypass.mutex};
    if (oot_network_bypass.tonemapped.Address() != target.Address()) {
        LOG_INFO(Render_Vulkan, "Ocarina of Time: tonemapped image at {:#x} {}x{}", target.Address(),
                 target.width, target.height);
    }
    oot_network_bypass.tonemapped = target;
}

void RasterizerVulkan::TraceDrawFailure(std::string_view reason) {
    if (!Common::DkbFrameTrace::Enabled()) {
        return;
    }
    ++trace_skipped;
    if (trace_failure_details++ >= 4) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    std::array<GPUVAddr, 6> stages{};
    for (size_t i = 0; i < stages.size(); ++i) {
        if (regs.IsShaderConfigEnabled(i)) {
            stages[i] = maxwell3d->GetNativeMacroEngine().native_turing
                            ? (u64{regs.pipelines[i].reserved[0] & 0xff} << 32) |
                                  regs.pipelines[i].reserved[1]
                            : regs.program_region.Address() + regs.pipelines[i].offset;
        }
    }
    LOG_INFO(Render_Vulkan,
             "DKB draw skipped frame={} reason={} stages[0..5]={:#x} rt0={:#x} {}x{}",
             trace_frame, reason, fmt::join(stages, ","), regs.rt[0].Address(),
             regs.rt[0].width, regs.rt[0].height);
}

void RasterizerVulkan::CaptureIndirectTrace() {
    if (!trace_sample_frame || trace_captures >= 4) {
        return;
    }
    ++trace_captures;
    const auto slot = std::find_if(indirect_traces.begin(), indirect_traces.end(),
                                   [](const auto& trace) { return !trace.pending; });
    if (slot == indirect_traces.end()) {
        ++trace_capture_dropped;
        return;
    }
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto [commands, command_offset] = buffer_cache.GetDrawIndirectBuffer();
    const u32 max_draws = static_cast<u32>(params.max_draw_counts);
    std::array<VkBufferCopy, 8> copies{};
    u32 sampled{};
    const u64 command_bytes = params.is_byte_count ? 4 : 8;
    const u64 available = command_offset <= commands->SizeBytes()
                              ? std::min<u64>(params.buffer_size,
                                              commands->SizeBytes() - command_offset)
                              : 0;
    const u32 limit = params.is_byte_count ? 1 : std::min(max_draws, 8U);
    for (u32 i = 0; i < limit; ++i) {
        if (available < command_bytes || (i && params.stride > (available - command_bytes) / i)) {
            break;
        }
        const u64 offset = command_offset + i * params.stride;
        copies[sampled++] = VkBufferCopy{offset, i * 8ULL, command_bytes};
    }
    VkBuffer count_buffer{};
    VkBufferCopy count_copy{};
    if (params.include_count && !params.is_byte_count) {
        const auto [count, offset] = buffer_cache.GetDrawIndirectCount();
        if (offset > count->SizeBytes() || sizeof(u32) > count->SizeBytes() - offset) {
            ++trace_capture_dropped;
            return;
        }
        count_buffer = count->Handle();
        count_copy = VkBufferCopy{offset, 64, sizeof(u32)};
    }
    if (!sampled && !count_buffer) {
        LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} max_draws={} sampled=0",
                 trace_frame, params.indirect_start_address, max_draws);
        return;
    }
    if (!slot->readback) {
        slot->readback = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = 68,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }
    slot->frame = trace_frame;
    slot->address = params.indirect_start_address;
    slot->count_address = params.count_start_address;
    slot->max_draws = max_draws;
    slot->sampled_commands = sampled;
    slot->include_count = count_buffer != VK_NULL_HANDLE;
    slot->byte_count = params.is_byte_count;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = commands->Handle(), destination = *slot->readback, copies, sampled,
                      count_buffer, count_copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        if (sampled) {
            cmdbuf.CopyBuffer(source, destination, vk::Span(copies.data(), sampled));
        }
        if (count_buffer) {
            cmdbuf.CopyBuffer(count_buffer, destination, count_copy);
        }
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    slot->tick = scheduler.CurrentTick();
    slot->pending = true;
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    scheduler.NoteRenderPassStores();
}

void RasterizerVulkan::CollectIndirectTrace() {
    for (auto& trace : indirect_traces) {
        if (!trace.pending || !scheduler.IsFree(trace.tick)) {
            continue;
        }
        trace.readback.Invalidate();
        std::array<u32, 17> words{};
        std::memcpy(words.data(), trace.readback.Mapped().data(), sizeof(words));
        if (trace.byte_count) {
            LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} gpu_byte_count={}",
                     trace.frame, trace.address, words[0]);
        } else {
            const u32 count = trace.include_count ? words[16] : trace.max_draws;
            const u32 requested_draws = std::min(count, trace.max_draws);
            const u32 sampled = std::min(requested_draws, trace.sampled_commands);
            u32 zero_commands{};
            std::string commands;
            for (u32 i = 0; i < sampled; ++i) {
                zero_commands += words[2 * i] == 0 || words[2 * i + 1] == 0;
                commands += fmt::format("{}{}x{}", i ? "," : "", words[2 * i], words[2 * i + 1]);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB indirect frame={} address={:#x} count_address={:#x} draw_count={} "
                     "count_source={} max_draws={} requested_draws={} sampled={} zero_commands={} "
                     "vertices_or_indices_x_instances=[{}] conditional_execution=unobserved",
                     trace.frame, trace.address, trace.count_address, count,
                     trace.include_count ? "gpu" : "command-parameter",
                     trace.max_draws, requested_draws, sampled, zero_commands, commands);
        }
        trace.pending = false;
    }
}

void RasterizerVulkan::DrawTexture() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Copy);

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    draw_packet.dynamic.mask = 0;
    UpdateDynamicStates(nullptr, draw_packet.dynamic);
    RecordDynamicStates(draw_packet.dynamic);

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer);
    }

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

// A 1-layer depth target bound with a 3D color target limits the framebuffer to one layer, so
// vkCmdClearAttachments can't reach the target's other slices. Those slices are zeroed by copying
// zeros into the clear's rectangle of exactly the slices the clear names; the rest of the volume
// keeps its contents. The rectangle can be smaller than the target: in handheld mode the game
// clears a 48-wide fog volume through a 64-wide render target.
bool RasterizerVulkan::ClearVolumeSlices(const Framebuffer* framebuffer, u32 first_slice,
                                         u32 slice_count) {
    const auto& regs = maxwell3d->regs;
    const u32 target_index = regs.clear_surface.RT;
    const auto& color_buffer_ids = texture_cache.GetRenderTargets().color_buffer_ids;
    // Depth and stencil clears take the normal path, so only a colour-only clear qualifies.
    if (target_index >= color_buffer_ids.size() || regs.clear_surface.Z || regs.clear_surface.S) {
        return false;
    }
    const auto view_id = color_buffer_ids[target_index];
    const u32 depth = regs.rt[target_index].depth;
    if (!view_id || depth <= framebuffer->Layers() || first_slice >= depth) {
        return false;
    }
    const auto& view = texture_cache.GetImageView(view_id);
    const bool is_volume = True(view.flags & VideoCommon::ImageViewFlagBits::Slice);
    const bool clears_all_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool clears_to_zero =
        std::ranges::all_of(regs.clear_color, [](f32 value) { return value == 0.0f; });
    if (!is_volume || !clears_all_channels || !clears_to_zero) {
        return false;
    }
    const VkRect2D rect = VolumeClearRect(regs, framebuffer->RenderArea(), view.size);
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return true;
    }
    const u32 width = rect.extent.width;
    const u32 height = rect.extent.height;
    // The host image can use a wider format than the guest's (B10G11R11 falls back to
    // R16G16B16A16): size the zeros for the widest color texel, 16 bytes.
    constexpr size_t WidestTexelBytes = 16;
    const size_t slice_bytes = size_t{width} * height * WidestTexelBytes;
    const auto zeros = staging_pool.Request(slice_bytes, MemoryUsage::Upload);
    std::memset(zeros.mapped_span.data(), 0, slice_bytes);
    const u32 count = std::min(slice_count, depth - first_slice);
    const VkImage image = view.ImageHandle();
    const u32 level = view.range.base.level;
    // The clear's slices count from the view's first slice.
    const u32 base_slice = static_cast<u32>(view.range.base.layer);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image, level, base_slice, rect, width, height, first_slice, count,
                      buffer = zeros.buffer, offset = zeros.offset](vk::CommandBuffer cmdbuf) {
        // Earlier passes may still be reading or writing the volume.
        constexpr VkMemoryBarrier clear_waits{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, clear_waits);
        for (u32 slice = first_slice; slice < first_slice + count; ++slice) {
            const VkBufferImageCopy zero_slice{
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                .imageOffset = {rect.offset.x, rect.offset.y, static_cast<s32>(base_slice + slice)},
                .imageExtent = {width, height, 1},
            };
            cmdbuf.CopyBufferToImage(buffer, image, VK_IMAGE_LAYOUT_GENERAL, zero_slice);
        }
        constexpr VkMemoryBarrier clear_finished{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, clear_finished);
    });
    return true;
}

void RasterizerVulkan::Clear(u32 layer_count) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Clear);
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer);
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    const u32 base_array_layer = regs.clear_surface.layer;
    const auto clear_layers =
        GetFramebufferClearLayers(base_array_layer, layer_count, framebuffer->Layers());
    if (clear_layers.count != layer_count) {
        LOG_DEBUG(Render_Vulkan,
                  "Framebuffer clear layers: base={}, requested={}, clipped={}, framebuffer={}",
                  base_array_layer, layer_count, clear_layers.count, framebuffer->Layers());
    }
    if (ClearVolumeSlices(framebuffer, base_array_layer, layer_count)) {
        return;
    }
    if (clear_layers.count == 0) {
        return;
    }

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = base_array_layer,
        .layerCount = clear_layers.count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_clears++ < 2048) {
        const u32 target = regs.clear_surface.RT;
        LOG_DEBUG(Render_Vulkan,
                  "Render trace clear: frame={} docked={} color={} depth={} stencil={} "
                  "color_target={} color_address={:#x} depth_address={:#x} depth_value={} base_layer={} layers={} "
                  "framebuffer_layers={} rect=({},{} {}x{})",
                  render_trace_frame, Settings::IsDockedMode(), use_color, use_depth, use_stencil,
                  target, target < regs.rt.size() ? regs.rt[target].Address() : 0,
                  regs.zeta.Address(), regs.clear_depth,
                  clear_rect.baseArrayLayer, clear_rect.layerCount, framebuffer->Layers(),
                  clear_rect.rect.offset.x, clear_rect.rect.offset.y,
                  clear_rect.rect.extent.width, clear_rect.rect.extent.height);
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 clear_layers.covers_all_layers &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    // Deferred load-op clears have no render pass yet. Starting a query here
    // would begin it outside the pass and end it inside the next draw's pass.
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                              !can_defer_clear && regs.zpass_pixel_count_enable);
    if (state_tracker.TouchViewports()) {
        maxwell3d->dirty.flags[Dirty::Scissors] = true;
        DynamicStatePacket& dynamic = draw_packet.dynamic;
        dynamic.mask = 0;
        ReadDynamicGroup(DynamicStatePacket::Viewports, regs,
                         {.device = &device, .is_rescaling = texture_cache.IsRescaling()},
                         dynamic);
        dynamic.Set(DynamicStatePacket::Viewports);
        RecordDynamicStates(dynamic);
    }

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

void RasterizerVulkan::DispatchCompute() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Compute);
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    FlushWork();
    gpu_memory->FlushCaching();
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.DrainIfFull();
    }

    u64 native_key{};
    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline(native_key)};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");
        return;
    }
    std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
    std::optional<ByteAddressedShadow> byte_addressed_shadow;
    buffer_cache.Nx2CollectListCopies(true);
    SCOPE_EXIT {
        buffer_cache.Nx2CollectListCopies(false);
        if (byte_addressed_shadow) {
            texture_cache_runtime.RetireByteAddressedShadow(*byte_addressed_shadow);
        }
    };
    const auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    // The full-size 1080p pass is replaced only by FSR (native AA); DLAA through it is untested.
    const bool can_replace = OcarinaTemporalEnabled() && pipeline->CanReplaceOcarinaUpscale() &&
                             (pipeline->ShaderHash() != ComputePipeline::OcarinaFullSizeShader ||
                              oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr) &&
                             !indirect_address && !frame_dump.Capturing();
    // Until a dispatch overwrites the scene, each direct one binds only after the effects are
    // marked. The indirect path records its bind at once.
    const bool marking_effects = oot_effects_frame == render_trace_frame && OcarinaTemporalEnabled() &&
                                 !indirect_address && !frame_dump.Capturing();
    ComputePipeline::BindCommand deferred_bind;
    if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                             texture_cache, texture_cache_runtime, byte_addressed_shadow,
                             can_replace || marking_effects ? &deferred_bind : nullptr)) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }
    SCOPE_EXIT {
        if (!frame_dump.Capturing()) {
            return;
        }
        boost::container::small_vector<ImageView*, 8> written;
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            written.push_back(&texture_cache.GetImageView(id));
        }
        frame_dump.CaptureDispatch(pipeline->ShaderHash(), written);
    };
    TracePendingCpuWrites(true, std::array{pipeline->ShaderHash()});
    if (frame_dump.CapturingSampled()) {
        boost::container::small_vector<FrameDump::Sampled, 16> sampled;
        for (const auto& image : pipeline->SampledImages()) {
            sampled.push_back({5, image.tic, &texture_cache.GetImageView(image.id),
                               &texture_cache.GetSampler(image.sampler)});
        }
        frame_dump.NoteDispatchSampled(pipeline->ShaderHash(), sampled);
    }
    if (frame_dump.CapturingBuffers()) {
        FrameDumpBuffers(true, std::array{pipeline->ShaderHash()});
    }

    const auto& qmd{kepler_compute->launch_description};
    const auto publish_collision_map = [&] {
        if (native_key != 0x6e0ae200ee9aa220ULL ||
            (qmd.const_buffer_enable_mask & 1) == 0) {
            return;
        }
        const auto& constants = qmd.const_buffer_config[0];
        if (constants.size < 0x3c) {
            return;
        }
        // WaterSurfaceCollisionHeightMap exposes its output handle in CB0.
        const u32 handle = gpu_memory->Read<u32>(constants.Address() + 0x38);
        const u32 index = Tegra::Texture::TexturePair(handle, qmd.linked_tsc != 0).first;
        texture_cache.PublishComputeCollisionImage(index);
    };
    if (frame_dump.CapturingBuffers()) {
        std::optional<std::array<u32, 3>> indirect_grid;
        if (indirect_address) {
            indirect_grid = std::array{gpu_memory->Read<u32>(*indirect_address),
                                       gpu_memory->Read<u32>(*indirect_address + 4),
                                       gpu_memory->Read<u32>(*indirect_address + 8)};
        }
        frame_dump.NoteLaunch(pipeline->ShaderHash(),
                              {qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z},
                              {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
                              indirect_address, indirect_grid);
    }
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.NoteComputeWrites();
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        scheduler.Record([this, pipeline, indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            RecordDispatchBarrier(cmdbuf);
            if (!pipeline->IsBound()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {}, [&] {
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
        });
        buffer_cache.NoteWorkRecorded(true);
        SubmitNx2ListCopies();
        publish_collision_map();
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.NoteComputeWrites();
    const bool barrier_recorded = can_replace || marking_effects;
    if (barrier_recorded) {
        // Effects snapshots and temporal upscaling can record work before the native dispatch.
        scheduler.Record([](vk::CommandBuffer cmdbuf) { RecordDispatchBarrier(cmdbuf); });
    }
    if (marking_effects) {
        const auto written = pipeline->WrittenImages();
        if (std::ranges::any_of(written, [this](VideoCommon::ImageViewId id) {
                return id && texture_cache.GetImageView(id).gpu_addr == oot_effects_address;
            })) {
            oot_live_dlss->MarkEffects(oot_effects_scene, render_trace_frame);
            oot_effects_frame.reset();
        }
    }
    const auto* const replacement =
        can_replace && !byte_addressed_shadow && TryOcarinaUpscale(*pipeline, true)
            ? oot_live_dlss.get()
            : nullptr;
    if (!replacement) {
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
    }
    scheduler.Record([this, pipeline, dim, byte_addressed_shadow, bind = std::move(deferred_bind),
                      replacement, barrier_recorded](vk::CommandBuffer cmdbuf) {
        if (!barrier_recorded) {
            RecordDispatchBarrier(cmdbuf);
        }
        if (replacement && !replacement->EvaluationFailed()) {
            return;
        }
        if (replacement) {
            LOG_WARNING(Render_Vulkan, "Ocarina temporal evaluation failed; dispatching native upscaler");
        }
        if (bind) {
            bind(cmdbuf);
        }
        if (!pipeline->IsBound()) {
            return;
        }
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
        }
        scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, dim,
                               [&] { cmdbuf.Dispatch(dim[0], dim[1], dim[2]); });
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
        }
    });
    if (can_replace || marking_effects) {
        // The deferred native binding may follow DLSS's command buffer bindings.
        scheduler.InvalidateState();
    }
    buffer_cache.NoteWorkRecorded(true);
    SubmitNx2ListCopies();
    publish_collision_map();

    if (!can_replace) {
        TryOcarinaUpscale(*pipeline, false);
    }

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::QueryCache);
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::QueryCache);
    VideoCommon::DrawStages::NoteGpuWrite(
        VideoCommon::DrawStages::Join::Query, *gpu_memory, gpu_addr,
        True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout) ? 16 : 4);
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::SubmitNx2ListCopies() {
    // Submitted at once: ReadGpuWrittenCommands waits on these copies, so they can't sit out
    // the longer DeferFlush interval (#161).
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush();
    }
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerVulkan::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync, u64* pending_tick) {
    // No cache lock: the front only reads the copies, under their own lock.
    switch (buffer_cache.Nx2ReadListCopy(addr, out, past_sync, pending_tick)) {
    case BufferCache::Nx2ListRead::Copied:
        return GpuWrittenList::Copied;
    case BufferCache::Nx2ListRead::Synced:
        return GpuWrittenList::Synced;
    case BufferCache::Nx2ListRead::NeedsSync:
        return GpuWrittenList::NeedsSync;
    case BufferCache::Nx2ListRead::Pending:
        return GpuWrittenList::Pending;
    default:
        return GpuWrittenList::Unknown;
    }
}

bool RasterizerVulkan::IsGpuTickDone(u64 tick) {
    if (scheduler.IsFree(tick)) {
        return true;
    }
    // A deferred submission (Scheduler::DeferFlush) is only made by the split's back: ask it to
    // submit, or the tick never completes while the front polls.
    if (!scheduler.IsSubmitted(tick) && VideoCommon::GpuSplit::Get().Active()) {
        VideoCommon::GpuSplit::Get().RequestIdleHook();
    }
    scheduler.GetMasterSemaphore().Refresh();
    return scheduler.IsFree(tick);
}

void RasterizerVulkan::NoteGpuWrittenCommandListSynced(DAddr page) {
    buffer_cache.Nx2NoteListSynced(page);
}

void RasterizerVulkan::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    gpu.InvalidateGPUCache();
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2FlushListPages(pages);
}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & (VideoCommon::CacheType::TextureCache | VideoCommon::CacheType::BufferCache))) {
        // CPU writes into GPU-written memory are only collected when they happen. Apply them
        // first, or the download writes the stale GPU copy over them: the Yoshi demo writes a
        // native program over memory a compute pass had bound as a written texel buffer, and its
        // title menu or background read as zeros once the shader cache's read flushed it.
        gpu.InvalidateGPUCache();
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        VideoCommon::DrawStages::Count(VideoCommon::DrawStages::Join::BufferFlush);
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

void RasterizerVulkan::FlushRegionForRead(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    // As FlushRegion, but the buffer cache doesn't wait for the host GPU here (Scheduler::Finish,
    // with its lock held): the reading thread does, in FinishRead.
    gpu.InvalidateGPUCache();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemoryForRead(addr, size);
    }
    query_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::FinishRead(DAddr addr, u64 size) {
    buffer_cache.FinishReadDownloads(addr, size);
}

bool RasterizerVulkan::NoteReportPending(DAddr addr, u64 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.NoteReportPending(addr, size);
}

void RasterizerVulkan::FinishReport(DAddr addr, u64 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.FinishReport(addr, size);
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    // The split's front asks this before each macro DMA read, and the back holds the buffer
    // cache's lock through each draw: pages the GPU never wrote need no look under it.
    if (True(which & VideoCommon::CacheType::BufferCache) &&
        (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
         Tegra::Nx2MayBeGpuWritten(addr, size))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

bool RasterizerVulkan::HasPendingQueries(DAddr addr, u64 size) {
    return query_cache.IsRegionGpuModified(addr, size);
}

bool RasterizerVulkan::HoldsQueries(DAddr addr, u64 size) {
    return query_cache.HoldsQueries(addr, size);
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> texture_area;
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_area = texture_cache.GetFlushArea(addr, size);
    }
    if (texture_area) {
        // The core disarmed the read's pages before asking, and skips later reads in this area,
        // even through another device alias of those pages. The buffer cache arms only pages it
        // doesn't count as armed, so it must forget the whole area.
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ForgetReadTraps(texture_area->start_address,
                                     texture_area->end_address - texture_area->start_address);
        return *texture_area;
    }
    if (buffer_cache.FlushesOnCpuRead()) {
        std::scoped_lock lock{buffer_cache.mutex};
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0 || False(which & VideoCommon::CacheType::BufferCache)) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    // MemoryManager landed the read downloads here just before (FinishRead) and writes guest memory
    // next. Landing a newer one would wait for the host GPU with the lock held, from any thread.
    buffer_cache.ClearReplacedDownload(addr, size);
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    // The GPU's own cached writes (constant buffer data, every draw): most land where no image,
    // query or shader was ever registered, which page_filter tells without the caches' locks.
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        for (const auto& [addr, size] : sequences) {
            if (filter.Any(addr, size)) {
                return true;
            }
        }
        return false;
    };
    // Even with skip_cpu_inner_invalidation: engines writing images in software (DMA pitch to
    // block linear copies of decoded video frames) reach the texture cache only through here.
    if (any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    if (any(query_cache.page_filter)) {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    // Out of the GPU-written ranges first: applying one write downloads the GPU's bytes elsewhere
    // on its pages (WriteMemory), which would write over the next collected write.
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ClaimCpuWrites(ranges);
    }
    for (const auto& [addr, size] : ranges) {
        OnCacheInvalidation(addr, size);
    }
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only the
        // GPU thread may do: leave those pages to the fallback.
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Fence);
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Fence);
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Fence);
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Fence);
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Fence);
    fence_manager.WaitPendingFences(force);
}

void RasterizerVulkan::FlushQueryResults(DAddr addr, u64 size) {
    if (addr == 0 || size == 0 || !query_cache.SemiFlushRegion(addr, size)) {
        return;
    }
    // Below High GPU accuracy a fence runs its own callback at once, so wait for an operation
    // queued behind the pending report writes instead. On the split's back the fence's
    // submission may be deferred, and the fencing thread would wait for this thread to submit it.
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    bool written{};
    fence_manager.SyncOperation([&] {
        std::scoped_lock lock{wait_mutex};
        written = true;
        wait_cv.notify_all();
    });
    fence_manager.SignalFence([] {});
    scheduler.FlushDeferred();
    std::unique_lock lock{wait_mutex};
    wait_cv.wait(lock, [&written] { return written; });
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Flush);
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    if (device.IsMoltenVK() && scheduler.IsRenderPassActive()) {
        scheduler.DeferFragmentBarrier();
        fence_manager.SignalOrdering();
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([flags](vk::CommandBuffer cmdbuf) {
        // Later work waits for the earlier work and sees its writes.
        static constexpr VkMemoryBarrier WFI_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(flags, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WFI_BARRIER);
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Barrier);
    // The next draw breaks the render pass only if it reads what this one wrote (#161): Ocarina
    // ended ~5,900 render passes a second here.
    scheduler.DeferFragmentBarrier();
}

void RasterizerVulkan::TiledCacheBarrier() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Barrier);
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Flush);
    if (draw_counter == 0 && !pending_native_work) {
        return;
    }
    draw_counter = 0;
    pending_native_work = false;
    // A fence with a download is submitted by InnerFence::Queue, and SignalFence then comes here
    // with nothing recorded since: ~4,600 empty submits a second in Ocarina of Time's Castle
    // Town, each followed by a full rebind (from Fritchii c338269b92).
    if (!scheduler.RecordedSinceSubmit()) {
        Scheduler::CountSkippedFlush();
        return;
    }
    // Fences with downloads or queries come here right after queueing: ~9,400 a second in
    // Hyrule Warriors: Age of Imprisonment fights. Batched like the fences themselves.
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

void RasterizerVulkan::TraceRenderState(bool is_indexed, const char* outcome,
                                        const Framebuffer* framebuffer) {
    if (!VideoCommon::RenderTrace::SampleFrame(render_trace_frame)) {
        return;
    }
    if (render_trace_draws++ >= 2048) {
        if (render_trace_draws == 2049) {
            LOG_DEBUG(Render_Vulkan, "Render trace frame={} truncated after 2048 draws",
                      render_trace_frame);
        }
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    std::array<u64, 6> shaders{};
    u32 stages{};
    for (size_t stage = 0; stage < shaders.size(); ++stage) {
        shaders[stage] = (u64{regs.pipelines[stage].reserved[0] & 0xff} << 32) |
                         regs.pipelines[stage].reserved[1];
        stages |= static_cast<u32>(regs.IsShaderConfigEnabled(stage)) << stage;
    }
    const auto extent = framebuffer ? framebuffer->RenderArea() : VkExtent2D{};
    LOG_DEBUG(Render_Vulkan,
              "Render trace frame={} draw={} docked={} outcome={} indexed={} stages={:#x} "
              "shaders={:x} guest_count={} guest_instances={} framebuffer={}x{} layers={}",
              render_trace_frame, render_trace_draws, Settings::IsDockedMode(), outcome, is_indexed,
              stages, fmt::join(shaders, ","),
              is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count, draw.instance_count,
              extent.width, extent.height, framebuffer ? framebuffer->Layers() : 0);
    LOG_DEBUG(Render_Vulkan,
              "Render trace targets: color={:#x} {}x{} layers={} format={:#x} depth={:#x} "
              "{}x{} layers={} format={:#x} enabled={} clip={}x{}",
              regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height, regs.rt[0].depth.Value(),
              static_cast<u32>(regs.rt[0].format), regs.zeta.Address(), regs.zeta_size.width,
              regs.zeta_size.height, regs.zeta_size.depth.Value(), static_cast<u32>(regs.zeta.format),
              regs.zeta_enable, regs.surface_clip.width.Value(), regs.surface_clip.height.Value());
    const auto& viewport = regs.viewport_transform[0];
    const auto& scissor = regs.scissor_test[0];
    LOG_DEBUG(Render_Vulkan,
              "Render trace tests: raster={} depth_test={} depth_write={} depth_func={} "
              "cull={} face={} alpha={} func={} ref={} color_mask={:#x} "
              "viewport_scale=({},{},{}) translate=({},{},{}) scissor={} {}..{} {}..{}",
              regs.rasterize_enable, regs.depth_test_enable, regs.depth_write_enabled,
              static_cast<u32>(regs.depth_test_func), regs.gl_cull_test_enabled,
              static_cast<u32>(regs.gl_cull_face), regs.alpha_test_enabled,
              static_cast<u32>(regs.alpha_test_func), regs.alpha_test_ref, regs.color_mask[0].raw,
              viewport.scale_x, viewport.scale_y, viewport.scale_z, viewport.translate_x,
              viewport.translate_y, viewport.translate_z, scissor.enable, scissor.min_x.Value(),
              scissor.max_x.Value(), scissor.min_y.Value(), scissor.max_y.Value());
    for (size_t binding = 0; binding < regs.vertex_streams.size(); ++binding) {
        const auto& stream = regs.vertex_streams[binding];
        if (stream.IsEnabled()) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace vertex: binding={} address={:#x} limit={:#x} stride={} "
                      "frequency={} repacked={}",
                      binding, stream.Address(), regs.vertex_stream_limits[binding].Address(),
                      stream.stride.Value(), stream.frequency,
                      buffer_cache_runtime.IsVertexRepacked(static_cast<u32>(binding)));
        }
    }
    if (is_indexed) {
        LOG_DEBUG(Render_Vulkan, "Render trace index: address={:#x} limit={:#x} format={} first={}",
                  regs.index_buffer.StartAddress(), regs.index_buffer.EndAddress(),
                  static_cast<u32>(regs.index_buffer.format), draw.index_buffer.first);
    }
    LOG_DEBUG(Render_Vulkan, "Render trace condition: address={:#x} mode={} override={}",
              regs.render_enable.Address(), static_cast<u32>(regs.render_enable.mode),
              static_cast<u32>(regs.render_enable_override));
}

void RasterizerVulkan::TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes) {
    // Guest CPU writes into GPU-written buffer memory are collected, and applied to the host
    // buffers only at the next fence or cache invalidation. A hit here reads the older bytes.
    static const bool enabled = std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto*, u32) {
        if (lines > max_lines || !gpu.HasPendingCPUWrites(binding.device_addr, bound_size)) {
            return;
        }
        if (++lines > max_lines) {
            LOG_INFO(HW_GPU, "Pending write trace stopped after {} lines", max_lines);
            return;
        }
        const size_t program = compute ? 0 : stage + 1;
        LOG_INFO(HW_GPU,
                 "Pending CPU write frame={} {} shader={:016x} stage={} {}[{}]{} addr={:#x} "
                 "device={:#x} size={:#x}",
                 render_trace_frame, compute ? "dispatch" : "draw",
                 program < shader_hashes.size() ? shader_hashes[program] : 0, stage,
                 storage ? "ssbo" : "cbuf", index, written ? " written" : "", binding.gpu_addr,
                 binding.device_addr, bound_size);
    });
}

bool RasterizerVulkan::OcarinaTemporalEnabled() const {
    constexpr u64 ocarina_program = 0x0400486024DEC000ULL;
    if (program_id != ocarina_program || oot_live_failed) {
        return false;
    }
    if (oot_backend) {
        return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
    }
    oot_backend = ResolveOcarinaUpscaler(device);
    if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Native &&
        Settings::values.ocarina_fsr.GetValue() && !Fsr3Upscaler::IsSupported(device)) {
        LOG_WARNING(Render_Vulkan, "Ocarina FSR 3.1 unsupported on this device; using native upscale");
    }
    return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
}

/// Capture the two camera matrices from a draw targeting the scene color and depth. Terrain and
/// materials draw with several vertex shaders, so any draw into a scene-sized target qualifies;
/// the camera buffer's contents are checked below.
void RasterizerVulkan::TrackOcarinaCamera() {
    if (!OcarinaTemporalEnabled()) {
        return;
    }

    const auto& targets = texture_cache.GetRenderTargets();
    if (!targets.color_buffer_ids[0] || !targets.depth_buffer_id) {
        return;
    }
    const auto& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
    if (color.size.width < 640 || color.size.height < 360 ||
        color.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return;
    }
    // Reflections and half-size passes share the camera layout; the upscale reads the largest.
    if (oot_camera_sample_frame == render_trace_frame &&
        u64{color.size.width} * color.size.height <=
            u64{oot_motion.extent.width} * oot_motion.extent.height) {
        return;
    }

    buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                const auto& binding, u32 bound_size,
                                                const auto*, u32) {
        std::array<f32, 0x230 / sizeof(f32)> camera;
        if (stage != 0 || storage || index != 3 || bound_size < sizeof(camera) ||
            buffer_cache.IsRegionGpuModified(binding.device_addr, sizeof(camera))) {
            return;
        }
        device_memory.ReadBlockUnsafe(binding.device_addr, camera.data(), sizeof(camera));
        constexpr size_t extent_offset = 0x120 / sizeof(f32);
        constexpr size_t projection_wz = 0xa8 / sizeof(f32);
        if (camera[extent_offset] != static_cast<f32>(color.size.width) ||
            camera[extent_offset + 1] != static_cast<f32>(color.size.height) ||
            camera[projection_wz] != -1.0f || camera[projection_wz + 1] != 0.0f) {
            return;
        }
        const std::span values{camera};
        oot_camera_sample_frame = render_trace_frame;
        oot_motion.camera_channel = gpu_memory;
        oot_motion.extent = {color.size.width, color.size.height};
        std::copy_n(values.begin() + 0x30 / sizeof(f32), 16, oot_motion.current.begin());
        std::copy_n(values.begin() + 0x180 / sizeof(f32), 16, oot_motion.previous.begin());
    });
}

namespace {
f32 Halton(u32 index, u32 base) {
    f32 fraction = 1.0f;
    f32 result = 0.0f;
    while (index > 0) {
        fraction /= static_cast<f32>(base);
        result += fraction * static_cast<f32>(index % base);
        index /= base;
    }
    return result;
}

} // Anonymous namespace

void RasterizerVulkan::UpdateOcarinaJitter() {
    std::array<f32, 2> want{};
    // Only while FSR replaced the previous frame's upscale: a frame the game upscales itself
    // would show the offset as a shake. Only draws that depth-test against the scene depth at
    // the scene size, so a depth pre-pass and the colour passes move together; shadow maps,
    // half-size passes and full-screen passes (which keep the depth bound without using it, as
    // #678 found, some with depth writes on: with the test off nothing writes depth) stay put.
    const auto& regs = maxwell3d->regs;
    if (oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr && oot_motion.extent.width != 0 &&
        oot_dlss_frame && *oot_dlss_frame + 1 >= render_trace_frame && regs.depth_test_enable) {
        const auto& targets = texture_cache.GetRenderTargets();
        if (targets.depth_buffer_id) {
            const auto& depth = texture_cache.GetImageView(targets.depth_buffer_id);
            if (depth.size.width == oot_motion.extent.width &&
                depth.size.height == oot_motion.extent.height && !depth.IsRescaled()) {
                // Halton (2, 3) with AMD's phase count for the last FSR frame's scale.
                const u32 index = static_cast<u32>(render_trace_frame % oot_jitter_phases) + 1;
                want = {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
                oot_jitter_frame = render_trace_frame;
                oot_jitter_offset = want;
            }
        }
    }
    if (want != oot_jitter_applied) {
        oot_jitter_applied = want;
        maxwell3d->dirty.flags[Dirty::Viewports] = true;
    }
}

/// Capture the skinning pose of the recognized Link draw for frame-to-frame motion.
void RasterizerVulkan::TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect) {
    if (shader_hash != 0x28c6e527634f814aULL || !is_indexed || is_indirect ||
        !OcarinaTemporalEnabled() || oot_actor.frame == render_trace_frame) {
        return;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& regs = maxwell3d->regs;
    const auto& stream = regs.vertex_streams[0];
    if (draw.index_buffer.count != 9702 || draw.index_buffer.FormatSizeInBytes() != 2 ||
        !stream.IsEnabled() || stream.stride != 48 || regs.vertex_stream_instances.IsInstancingEnabled(0)) {
        return;
    }
    const auto attribute = [&](u32 location, VkFormat format, u32 offset) {
        const auto& attrib = regs.vertex_attrib_format[location];
        return attrib.buffer == 0 && !attrib.constant && attrib.offset == offset &&
            MaxwellToVK::VertexFormat(device, attrib.type, attrib.size) == format;
    };
    if (!attribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0) ||
        !attribute(4, VK_FORMAT_R8G8B8A8_UNORM, 40) || !attribute(5, VK_FORMAT_R8G8B8A8_UNORM, 44) ||
        !attribute(6, VK_FORMAT_R16G16B16A16_UINT, 12) || !attribute(7, VK_FORMAT_R16G16B16A16_UINT, 20)) {
        return;
    }

    OcarinaActorPose pose{
        .frame = render_trace_frame,
        .channel = gpu_memory,
        .vertex = stream.Address(),
        .indices = draw.index_buffer.IndexStart(),
        .base_vertex = static_cast<s32>(draw.base_index),
    };
    bool origin_found{}, bones_found{};
    buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                const auto& binding, u32 size, const auto*, u32) {
        if (stage != 0 || storage || (index != 4 && index != 6) ||
            buffer_cache.IsRegionGpuModified(binding.device_addr, size)) {
            return;
        }
        if (index == 4 && size >= 0x40 + 3 * sizeof(f32)) {
            device_memory.ReadBlockUnsafe(binding.device_addr + 0x40, pose.origin.data(), 3 * sizeof(f32));
            origin_found = true;
        }
        if (index == 6 && size == 13568) {
            device_memory.ReadBlockUnsafe(binding.device_addr, pose.bones.data(), sizeof(pose.bones));
            bones_found = true;
        }
    });
    if (origin_found && bones_found) {
        oot_previous_actor = oot_actor;
        oot_actor = pose;
    }
}

/// Read the shared mesh only when both poses refer to the same vertex and index buffers.
void RasterizerVulkan::ReadOcarinaActor(OcarinaActorMesh& mesh) {
    const auto& now = oot_actor;
    const auto& before = oot_previous_actor;
    if (now.frame != render_trace_frame || before.frame + 1 != now.frame ||
        now.channel != gpu_memory || before.channel != now.channel || now.vertex != before.vertex ||
        now.indices != before.indices || now.base_vertex != before.base_vertex) {
        return;
    }
    const auto read = [&](GPUVAddr address, void* dst, size_t size) {
        if (!gpu_memory->IsContinuousRange(address, size) ||
            !gpu_memory->IsFullyMappedRange(address, size)) {
            return false;
        }
        const auto device_address = gpu_memory->GpuToCpuAddress(address);
        if (!device_address || buffer_cache.IsRegionGpuModified(*device_address, size) ||
            gpu.HasPendingCPUWrites(*device_address, size)) {
            return false;
        }
        gpu_memory->ReadBlockUnsafe(address, dst, size);
        return true;
    };
    constexpr size_t index_count = 9702;
    constexpr size_t stride = 48;
    std::vector<u16> indices(index_count);
    if (!read(now.indices, indices.data(), indices.size() * sizeof(u16))) {
        return;
    }
    const auto [minimum, maximum] = std::ranges::minmax(indices);
    const s64 first = static_cast<s64>(now.base_vertex) + minimum;
    const u32 vertex_count = static_cast<u32>(maximum) - minimum + 1;
    if (first < 0 || first > (1 << 20) || vertex_count > 16384) {
        return;
    }
    std::vector<u8> vertices(vertex_count * stride);
    if (!read(now.vertex + first * stride, vertices.data(), vertices.size())) {
        return;
    }
    for (u16& index : indices) {
        index -= minimum;
    }

    mesh = {std::move(vertices), std::move(indices), now.origin, before.origin, now.bones, before.bones};
}

/// Track the scene depth on its reader dispatch, or try to replace a recognized upscale dispatch.
bool RasterizerVulkan::TryOcarinaUpscale(const ComputePipeline& pipeline, bool replace) {
    constexpr u64 depth_reader = 0x1fc7a0423116281fULL;
    const bool pass_1440p = replace && pipeline.ShaderHash() == ComputePipeline::Ocarina1440pShader;
    if ((!replace && pipeline.ShaderHash() != depth_reader) || !OcarinaTemporalEnabled()) {
        return false;
    }
    const bool native_resolution =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss &&
        Settings::GetNvidiaDlssRuntimeSettings().quality == Settings::NvidiaDlssQuality::Native;
    const bool upscale_4k = pass_1440p && !native_resolution;
    const auto read_tic = [this](u32 index) -> std::optional<Tegra::Texture::TICEntry> {
        const auto& table = kepler_compute->regs.tic;
        using Entry = Tegra::Texture::TICEntry;
        const GPUVAddr address = table.Address() + u64{index} * sizeof(Entry);
        if (index > table.limit || !gpu_memory->IsFullyMappedRange(address, sizeof(Entry))) {
            return std::nullopt;
        }
        Entry entry{};
        gpu_memory->ReadBlockUnsafe(address, &entry, sizeof(entry));
        return entry;
    };
    if (!replace) {
        constexpr u32 depth_handle_offset = 0x128;
        const auto& qmd = kepler_compute->launch_description;
        const auto& cbuf = qmd.const_buffer_config[0];
        if (!(qmd.const_buffer_enable_mask & 1) || cbuf.size < depth_handle_offset + sizeof(u32) ||
            !gpu_memory->IsFullyMappedRange(cbuf.Address(), depth_handle_offset + sizeof(u32))) {
            return false;
        }
        const u32 index = gpu_memory->Read<u32>(cbuf.Address() + depth_handle_offset) & 0xfffff;
        const auto tic = read_tic(index);
        const bool scene_size = oot_camera_sample_frame != render_trace_frame ||
                                (tic && tic->Width() == oot_motion.extent.width &&
                                 tic->Height() == oot_motion.extent.height);
        if (tic && tic->Address() && scene_size) {
            oot_motion.depth_channel = gpu_memory;
            oot_motion.depth_frame = render_trace_frame;
            oot_motion.depth_tic = index;
            oot_motion.depth_address = tic->Address();
        }
        return false;
    }
    // Logs why the game's own upscale draws, when the reason changes and every 120th repeat.
    const auto fail = [this](const char* reason) {
        if (reason != oot_fallback_reason || ++oot_fallback_repeats % 120 == 0) {
            LOG_INFO(Render_Vulkan, "Ocarina temporal: the game's upscale draws frame {} ({})",
                     render_trace_frame, reason);
        }
        if (reason != oot_fallback_reason) {
            oot_fallback_reason = reason;
            oot_fallback_repeats = 0;
        }
        return false;
    };
    if (oot_camera_sample_frame != render_trace_frame) {
        return fail("no scene camera this frame");
    }
    if (oot_motion.depth_frame != render_trace_frame) {
        return fail("no scene depth this frame");
    }
    if (oot_motion.camera_channel != gpu_memory || oot_motion.depth_channel != gpu_memory) {
        return fail("camera and depth from another channel");
    }
    const auto tic = read_tic(oot_motion.depth_tic);
    if (!tic || tic->Address() != oot_motion.depth_address ||
        tic->Width() != oot_motion.extent.width || tic->Height() != oot_motion.extent.height) {
        return fail("scene depth descriptor changed");
    }

    texture_cache.SynchronizeDescriptors(true);
    std::array images{VideoCommon::ImageViewInOut{.index = oot_motion.depth_tic}};
    texture_cache.FillImageViews(images, true);
    if (!images[0].id) {
        return fail("no scene depth image");
    }
    auto& depth = texture_cache.GetImageView(images[0].id);
    const auto* backing = depth.BackingImage();
    if (!backing || depth.IsRescaled() || depth.format != VideoCore::Surface::PixelFormat::D32_FLOAT ||
        depth.range != VideoCommon::SubresourceRange{} || backing->info.num_samples != 1) {
        return fail("unsupported scene depth");
    }

    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return fail("unexpected upscale bindings");
    }
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    auto& target = texture_cache.GetImageView(written[0]);
    using VideoCore::Surface::PixelFormat;
    const auto extent = oot_motion.extent;
    const bool native_size = pipeline.ShaderHash() == ComputePipeline::OcarinaFullSizeShader;
    const bool valid_extents =
        pass_1440p ? extent.width == 2560 && extent.height == 1440 && target.size.width == 2560 &&
                         target.size.height == 1440
        : native_size ? target.size.width == extent.width && target.size.height == extent.height
                      : target.size.width == extent.width * 4 / 3 &&
                            target.size.height == extent.height * 4 / 3;
    if (!valid_extents || !color.BackingImage() || !target.BackingImage() ||
        color.IsRescaled() || target.IsRescaled() ||
        color.Samples() != VK_SAMPLE_COUNT_1_BIT || target.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        color.range != VideoCommon::SubresourceRange{} || target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT || target.format != PixelFormat::B10G11R11_FLOAT ||
        color.size.width != extent.width || color.size.height != extent.height ||
        color.ImageHandle() == target.ImageHandle() || depth.ImageHandle() == target.ImageHandle()) {
        return fail("unexpected upscale images");
    }

    constexpr VkFormat output_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const auto color_view = color.Handle(Shader::TextureType::Color2D);
    if (!color_view || !device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatSupported(output_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, FormatType::Optimal) ||
        (target.BackingImage()->UsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, target.format).format != output_format) {
        return fail("unsupported storage format");
    }
    // Without Link's pose the mesh stays empty and the motion comes from the camera alone.
    OcarinaActorMesh mesh{};
    ReadOcarinaActor(mesh);
    try {
        const auto target_view =
            target.StorageView(Shader::TextureType::Color2D, Shader::ImageFormat::Typeless, false);
        const auto scene_reference = upscale_4k ? target.Handle(Shader::TextureType::Color2D)
                                                : VK_NULL_HANDLE;
        if (!target_view || (upscale_4k && !scene_reference)) {
            return fail("no upscale target view");
        }
        if (!oot_live_dlss) {
            oot_live_dlss = std::make_unique<OcarinaUpscaler>(
                device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator,
                staging_pool, *oot_backend);
        }
        const VkExtent2D target_extent{target.size.width, target.size.height};
        // The offset this frame's scene was drawn with (none when no scene draw was offset).
        std::array<f32, 2> jitter{};
        if (oot_jitter_frame == render_trace_frame) {
            jitter = oot_jitter_offset;
        }
        const VkExtent2D output_extent = native_resolution ? extent
                                         : upscale_4k      ? VkExtent2D{3840, 2160}
                                                           : target_extent;
        const bool recorded = oot_live_dlss->Record(color_view, depth.DepthView(), target_view,
                                                   extent, output_extent, target_extent, oot_motion.current,
                                                   oot_motion.previous, mesh, depth.ImageHandle(),
                                                   render_trace_frame, jitter);
        if (!recorded) {
            return fail("temporal upscaler declined the frame");
        }
        if (upscale_4k) {
            oot_live_dlss->SaveSceneReference(scene_reference);
            oot_4k_scene_frame = render_trace_frame;
            oot_4k_guest_present = VK_NULL_HANDLE;
        }
        oot_dlss_frame = render_trace_frame;
        // AMD's jitter phase count for FSR: 8 * (output / render width)^2, so 18 for 1440p -> 4K
        // and 8 for native AA.
        const f32 ratio = static_cast<f32>(output_extent.width) / static_cast<f32>(extent.width);
        oot_jitter_phases = std::max(1U, static_cast<u32>(8.0f * ratio * ratio));
        oot_fallback_reason = nullptr;
        return true;
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina temporal upscaling unavailable: {}", error.what());
        oot_live_failed = true;
        texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        return false;
    }
}

bool RasterizerVulkan::FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes) {
    static constexpr u32 BUFFER_CAP = 4U << 20;
    std::vector<FrameDump::BoundBuffer> buffers;
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto* host, u32 host_offset) {
        const u32 size = std::min(bound_size, BUFFER_CAP);
        const bool gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
        FrameDump::BoundBuffer& buffer = buffers.emplace_back(FrameDump::BoundBuffer{
            .stage = stage,
            .storage = storage,
            .index = index,
            .written = written,
            .gpu_addr = binding.gpu_addr,
            .device_addr = binding.device_addr,
            .bound_size = bound_size,
            .size = size,
            .gpu_modified = gpu_modified,
            .host = host ? host->Handle() : VK_NULL_HANDLE,
            .host_offset = host_offset,
            .guest = {},
        });
        if (!gpu_modified || !host) {
            buffer.guest.resize(size);
            device_memory.ReadBlockUnsafe(binding.device_addr, buffer.guest.data(), size);
        }
    });
    return frame_dump.NoteBuffers(compute, shader_hashes, buffers);
}

void RasterizerVulkan::FrameDumpPass(const Framebuffer* framebuffer) {
    std::array<ImageView*, VideoCommon::NUM_RT> colors{};
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot]) {
            colors[slot] = &texture_cache.GetImageView(ids[slot]);
        }
    }
    if (frame_dump.BeginPass(framebuffer, colors)) {
        scheduler.RequestRenderpass(framebuffer);
    }
}

void RasterizerVulkan::TickFrame() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Frame);
    if (VideoCore::Nx2Debug::HasRequests()) {
        if (auto capture = VideoCore::Nx2Debug::TakeCapture()) {
            frame_dump.Arm(std::move(capture->directory), capture->sampled, capture->buffers,
                           capture->id);
        }
        if (const auto change = VideoCore::Nx2Debug::TakeRejectChange()) {
            pipeline_cache.RefreshDebugRejections(change->unrejected);
            VideoCore::Nx2Debug::NoteRejectApplied(change->generation);
        }
        while (const auto read = VideoCore::Nx2Debug::TakeGpuRead()) {
            std::vector<Tegra::MemoryManager*> spaces;
            if (gpu_memory != nullptr) {
                spaces.push_back(gpu_memory);
            }
            {
                std::scoped_lock lock{config_mutex};
                for (const auto& [id, space] : address_spaces) {
                    if (space.gpu_memory != gpu_memory) {
                        spaces.push_back(space.gpu_memory);
                    }
                }
            }
            VideoCore::Nx2Debug::FinishGpuRead(read->id,
                                               ReadGpuMemory(spaces, read->gpu_address, read->size));
        }
    }
    if (fast_draw_on) {
        ReportFastDraws();
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Frame);
    }
    frame_dump.TickFrame();
    ++render_trace_frame;
    render_trace_draws = 0;
    render_trace_clears = 0;
    if (Common::DkbFrameTrace::Enabled()) {
        CollectIndirectTrace();
        const auto now = std::chrono::steady_clock::now();
        if (trace_cpu_rate.Ready(now)) {
            auto snapshot = Common::DkbCpuTrace::ReadSnapshot();
            if (trace_cpu_snapshot && trace_frame > trace_cpu_frame) {
                const u64 frames = trace_frame - trace_cpu_frame;
                std::string roles;
                for (size_t i = 0; i < Common::DkbCpuTrace::RoleCount; ++i) {
                    const auto name = Common::DkbCpuTrace::RoleNames[i];
                    const auto delta = Common::DkbCpuTrace::CpuTimeDelta(
                        trace_cpu_snapshot->threads[i], snapshot.threads[i]);
                    if (delta) {
                        const double cpu_ms = static_cast<double>(*delta) / 1e6;
                        roles += fmt::format(" {}_cpu_ms={:.3f} {}_cpu_ms_per_frame={:.3f}",
                                             name, cpu_ms, name, cpu_ms / frames);
                    } else {
                        roles += fmt::format(" {}={}", name,
                                             snapshot.threads[i].active ? "unavailable" : "inactive");
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "DKB cpu through_frame={} frames={} window_wall_ms={:.3f}{}",
                         trace_frame, frames,
                         std::chrono::duration<double, std::milli>(
                             snapshot.wall_time - trace_cpu_snapshot->wall_time).count(),
                         roles);
            }
            trace_cpu_snapshot = std::move(snapshot);
            trace_cpu_frame = trace_frame;
        }
        ++trace_window_frames;
        if (trace_sample_frame) {
            LOG_INFO(Render_Vulkan,
                     "DKB draws through_frame={} frames={} attempts={} skipped={} indirect={} "
                     "zero_direct={} capture_dropped={} attempts_per_frame={:.1f} "
                     "zero_direct_per_frame={:.1f} (totals since previous line)",
                     trace_frame, trace_window_frames, trace_draws, trace_skipped, trace_indirect,
                     trace_zero_direct, trace_capture_dropped,
                     static_cast<double>(trace_draws) / trace_window_frames,
                     static_cast<double>(trace_zero_direct) / trace_window_frames);
            trace_draws = trace_skipped = trace_indirect = trace_zero_direct = 0;
            trace_window_frames = 0;
            trace_capture_dropped = trace_failure_details = 0;
            trace_rate.next = now + std::chrono::seconds{1};
        }
        ++trace_frame;
        trace_captures = 0;
        trace_sample_frame = trace_frame < 4 || trace_rate.Ready(now);
    }
    ComputePipeline::TickUpscalerTrace();
    draw_counter = 0;
    pending_native_work = false;
    pipeline_cache.TickFrame();
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    if (back_ring) {
        descriptor_buffer_ring.PrepareFrame(scheduler);
    }
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    if (back_staging_pool) {
        back_compute_pass_descriptor_queue->TickFrame();
        back_staging_pool->TickFrame();
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::QueryCache);
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Copy);
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerVulkan::UsesNvidiaDlss() const {
    return texture_cache_runtime.UsesNvidiaDlss();
}

bool RasterizerVulkan::SupportsGameDlssNetwork() const {
    // Compute pipelines are created 32 threads wide where the device can be told to.
    return device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
}

std::optional<VideoCore::GameDlssProbe> RasterizerVulkan::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.ProbeGameDlssOutput(dst, src);
}

bool RasterizerVulkan::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history,
    std::optional<std::array<u32, 2>> render_region) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history, render_region);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Copy);
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    VideoCommon::DrawStages::NoteWrite(VideoCommon::DrawStages::Join::Inline, *cpu_addr,
                                       copy_size);
    // EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    bool uploaded;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size);
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Frame);
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.CapturePresent(*image_view);
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    info.anti_aliased = oot_dlss_frame == render_trace_frame && !oot_live_dlss->EvaluationFailed();
    if (oot_4k_scene_frame == render_trace_frame && info.image &&
        info.image == oot_4k_guest_present && info.image_view &&
        info.width == 2560 && info.height == 1440 && !scaled &&
        image_view->format == VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM &&
        !oot_live_dlss->EvaluationFailed() && !frame_dump.Capturing()) {
        try {
            oot_live_dlss->Composite4k(info.image_view, render_trace_frame);
            info.image = oot_live_dlss->CompositedImage();
            info.image_view = oot_live_dlss->CompositedView();
            info.scaled_width = 3840;
            info.scaled_height = 2160;
            info.already_upscaled = true;
        } catch (const vk::Exception& error) {
            LOG_WARNING(Render_Vulkan, "Ocarina temporal 4K presentation unavailable: {}", error.what());
            oot_live_failed = true;
            texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        }
    }
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr bool android = true;
#else
    static constexpr bool android = false;
#endif // __ANDROID__
    const bool native = gpu_memory->UsesNativeTiling();
    u32 budget = SubmissionPolicy::DefaultNativeBudget;
    if (native && !android) {
        static const std::optional<u32> configured_budget = []() -> std::optional<u32> {
            if (const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS")) {
                if (const auto parsed = SubmissionPolicy::ParseBudget(value)) {
                    return parsed;
                }
                LOG_WARNING(Render_Vulkan,
                            "Invalid EDEN_NX2_FLUSH_DRAWS; expected an integer from 1 to {}",
                            (std::numeric_limits<u32>::max)());
            }
            return std::nullopt;
        }();
        budget = configured_budget.value_or(SubmissionPolicy::DefaultNativeBudget);
    }
    u32 legacy_budget = 4096;
    if (!native && !android) {
        static const u32 forced_draws = [] {
            const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS");
            return value != nullptr
                       ? static_cast<u32>(std::max(32L, std::strtol(value, nullptr, 10)) & ~31L)
                       : 0U;
        }();
        legacy_budget = forced_draws != 0 ? forced_draws : (device.IsMoltenVK() ? 128U : 4096U);
    }
    switch (SubmissionPolicy::Advance(draw_counter, native, android, budget, legacy_budget)) {
    case SubmissionPolicy::Action::None:
        break;
    case SubmissionPolicy::Action::Dispatch:
        scheduler.DispatchWork();
        break;
    case SubmissionPolicy::Action::Flush:
        scheduler.Flush();
        pending_native_work = false;
        break;
    }
    if (native && !android) {
        pending_native_work = true;
    }
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    const bool copied = buffer_cache.DMACopy(src_address, dest_address, amount);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush();
    }
    buffer_cache.Nx2CollectListCopies(false);
    return copied;
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    VideoCommon::DrawStages::Count(VideoCommon::DrawStages::Join::DmaImageBuffer);
    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates(const GraphicsPipeline* pipeline,
                                           DynamicStatePacket& out) {
    VideoCommon::DrawStages::Expect(VideoCommon::DrawStages::Stage::A, "UpdateDynamicStates");
    using G = DynamicStatePacket;
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    // A mesh emulation pipeline has static input assembly and vertex input. Binding it replaces
    // that dynamic state, so leave it to the next regular draw to set again.
    const bool emulated_mesh{pipeline != nullptr && pipeline->IsEmulatedMesh()};
    if (emulated_mesh) {
        state_tracker.InvalidateInputState();
    }
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = !emulated_mesh && state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }
    DynamicStateInputs inputs{
        .device = &device,
        .is_rescaling = texture_cache.IsRescaling(),
        .topology = topology,
        .color_output_mask = pipeline ? pipeline->ColorOutputMask() : ~0U,
        .viewport_offset = oot_jitter_applied,
    };
    const auto take = [&](G::Group group, bool touched = true) {
        if (touched) {
            ReadDynamicGroup(group, regs, inputs, out);
            out.Set(group);
        }
    };

    if (state_tracker.TouchViewports()) {
        flags[Dirty::Scissors] = true;
        take(G::Viewports);
    }
    take(G::Scissors, state_tracker.TouchScissors());
    if (state_tracker.TouchDepthBias()) {
        inputs.d24_depth_bias_rescale = !device.SupportsD24DepthBuffer() &&
                                        std::ranges::find(NEEDS_D24, program_id) !=
                                            std::ranges::end(NEEDS_D24);
        take(G::DepthBias);
    }
    take(G::BlendConstants, state_tracker.TouchBlendConstants());
    take(G::DepthBounds, state_tracker.TouchDepthBounds());
    if (state_tracker.TouchStencilProperties()) {
        bool update_references = state_tracker.TouchStencilReference();
        bool update_write_mask = state_tracker.TouchStencilWriteMask();
        bool update_compare_masks = state_tracker.TouchStencilCompare();
        const bool two_sided = regs.stencil_two_side_enable != 0;
        if (state_tracker.TouchStencilSide(two_sided)) {
            update_references = true;
            update_write_mask = true;
            update_compare_masks = true;
        }
        // The back face takes the front's values unless two-sided; record what changed.
        if (update_references) {
            const bool front = state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
            const bool back = state_tracker.CheckStencilReferenceBack(
                two_sided ? regs.stencil_back_ref : regs.stencil_front_ref);
            take(G::StencilReference, front || back);
        }
        if (update_write_mask) {
            const bool front = state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
            const bool back = state_tracker.CheckStencilWriteMaskBack(
                two_sided ? regs.stencil_back_mask : regs.stencil_front_mask);
            take(G::StencilWriteMask, front || back);
        }
        if (update_compare_masks) {
            const bool front =
                state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
            const bool back = state_tracker.CheckStencilCompareMaskBack(
                two_sided ? regs.stencil_back_func_mask : regs.stencil_front_func_mask);
            take(G::StencilCompareMask, front || back);
        }
        state_tracker.ClearStencilReset();
    }
    take(G::LineWidth, state_tracker.TouchLineWidth());

    if (device.IsExtExtendedDynamicStateSupported()) {
        take(G::CullMode, state_tracker.TouchCullMode());
        take(G::DepthCompareOp, state_tracker.TouchDepthCompareOp());
        take(G::FrontFace, state_tracker.TouchFrontFace());
        take(G::StencilOp, state_tracker.TouchStencilOp());
        if (state_tracker.TouchStateEnable()) {
            if (state_tracker.TouchDepthBoundsTestEnable()) {
                if (regs.depth_bounds_enable && !device.IsDepthBoundsSupported()) {
                    LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
                }
                take(G::DepthBoundsTestEnable);
            }
            take(G::DepthTestEnable, state_tracker.TouchDepthTestEnable());
            take(G::DepthWriteEnable, state_tracker.TouchDepthWriteEnable());
            take(G::StencilTestEnable, state_tracker.TouchStencilTestEnable());
        }
        take(G::PrimitiveTopology, topology_changed);
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        if (!emulated_mesh) {
            take(G::PrimitiveRestartEnable, state_tracker.TouchPrimitiveRestartEnable());
        }
        take(G::RasterizerDiscardEnable, state_tracker.TouchRasterizerDiscardEnable());
        take(G::DepthBiasEnable, state_tracker.TouchDepthBiasEnable());
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        take(G::LogicOp, state_tracker.TouchLogicOp());
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        take(G::LogicOpEnable, state_tracker.TouchLogicOpEnable() &&
                                   device.SupportsDynamicState3LogicOpEnable());
        take(G::DepthClampEnable, state_tracker.TouchDepthClampEnable() &&
                                      device.SupportsDynamicState3DepthClampEnable());
        if (device.IsExtLineRasterizationSupported() &&
            state_tracker.TouchLineRasterizationMode()) {
            if (!device.SupportsDynamicState3LineRasterizationMode()) {
                static std::once_flag warn_missing_rect;
                std::call_once(warn_missing_rect, [] {
                    LOG_WARNING(Render_Vulkan,
                                "Driver lacks rectangular line rasterization support; skipping "
                                "dynamic line state updates");
                });
            } else {
                if (regs.line_anti_alias_enable != 0 && !device.SupportsSmoothLines()) {
                    static std::once_flag warn_missing_smooth;
                    std::call_once(warn_missing_smooth, [] {
                        LOG_WARNING(Render_Vulkan,
                                    "Line anti-aliasing requested but smoothLines feature "
                                    "unavailable; using rectangular rasterization");
                    });
                }
                take(G::LineRasterizationMode);
            }
        }
        take(G::LineStippleEnable, state_tracker.TouchLineStippleEnable() &&
                                       device.SupportsDynamicState3LineStippleEnable());
        take(G::ConservativeRasterization,
             state_tracker.TouchConservativeRasterizationMode() &&
                 device.SupportsDynamicState3ConservativeRasterizationMode());
        if (state_tracker.TouchAlphaToCoverageEnable() &&
            device.SupportsDynamicState3AlphaToCoverageEnable()) {
            inputs.lookup_pipeline = pipeline_cache.CurrentGraphicsPipeline();
            take(G::AlphaToCoverageEnable);
        }
        if (state_tracker.TouchAlphaToOneEnable()) {
            if (!device.SupportsDynamicState3AlphaToOneEnable()) {
                static std::once_flag warn_alpha_to_one;
                std::call_once(warn_alpha_to_one, [] {
                    LOG_WARNING(Render_Vulkan,
                                "Alpha-to-one is not supported on this device; forcing it "
                                "disabled");
                });
            } else {
                inputs.lookup_pipeline = pipeline_cache.CurrentGraphicsPipeline();
                take(G::AlphaToOneEnable);
            }
        }
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        // The write mask also excludes components the fragment shader doesn't output, so it has
        // to be recorded again whenever the bound pipeline outputs a different set of components.
        const bool color_output_changed = inputs.color_output_mask != last_color_output_mask;
        if (state_tracker.TouchBlending() || color_output_changed) {
            if (state_tracker.TouchColorMask() || color_output_changed) {
                last_color_output_mask = inputs.color_output_mask;
                take(G::ColorWriteMask);
            }
            take(G::BlendEnable, state_tracker.TouchBlendEnable());
            take(G::BlendEquations, state_tracker.TouchBlendEquations());
        }
    } else if (device.IsExtColorWriteEnableSupported()) {
        take(G::ColorWriteEnable, state_tracker.TouchColorMask());
    }

    if (device.IsExtVertexInputDynamicStateSupported() && !emulated_mesh) {
        // A draw passes the pipeline it looked up: a second lookup was ~2.5% of the back thread.
        const GraphicsPipeline* const gp =
            pipeline != nullptr ? pipeline : pipeline_cache.CurrentGraphicsPipeline();
        if (gp && gp->HasDynamicVertexInput() &&
            (flags[Dirty::VertexInput] || flags[VideoCommon::Dirty::VertexBuffers])) {
            flags[Dirty::VertexInput] = false;
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                flags[Dirty::VertexAttribute0 + index] = false;
            }
            for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
                flags[Dirty::VertexBinding0 + index] = false;
            }
            inputs.repacked_vertex_buffers = RepackedVertexBuffers();
            take(G::VertexInput);
        }
    }
}

void RasterizerVulkan::RecordDynamicStates(const DynamicStatePacket& state) {
    VideoCommon::DrawStages::Expect(VideoCommon::DrawStages::Stage::B, "RecordDynamicStates");
    using G = DynamicStatePacket;
    for (u64 mask = state.mask; mask != 0; mask &= mask - 1) {
        switch (static_cast<G::Group>(std::countr_zero(mask))) {
        case G::Viewports:
            scheduler.Record([this, viewports = state.viewports](vk::CommandBuffer cmdbuf) {
                const u32 num_viewports =
                    std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
                cmdbuf.SetViewport(0, vk::Span<VkViewport>(viewports.data(), num_viewports));
            });
            break;
        case G::Scissors:
            scheduler.Record([this, scissors = state.scissors](vk::CommandBuffer cmdbuf) {
                const u32 num_scissors =
                    std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
                cmdbuf.SetScissor(0, vk::Span<VkRect2D>(scissors.data(), num_scissors));
            });
            break;
        case G::DepthBias:
            scheduler.Record([this, bias = state.depth_bias](vk::CommandBuffer cmdbuf) {
                if (device.IsExtDepthBiasControlSupported()) {
                    static VkDepthBiasRepresentationInfoEXT bias_info{
                        .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                        .pNext = nullptr,
                        .depthBiasRepresentation =
                            VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                        .depthBiasExact = VK_FALSE,
                    };
                    cmdbuf.SetDepthBias(bias.constant, bias.clamp, bias.slope, &bias_info);
                } else {
                    cmdbuf.SetDepthBias(bias.constant, bias.clamp, bias.slope);
                }
            });
            break;
        case G::BlendConstants:
            scheduler.Record([blend_color = state.blend_constants](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetBlendConstants(blend_color.data());
            });
            break;
        case G::DepthBounds:
            scheduler.Record([bounds = state.depth_bounds](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthBounds(bounds[0], bounds[1]);
            });
            break;
        case G::StencilReference:
            scheduler.Record([values = state.stencil_reference](vk::CommandBuffer cmdbuf) {
                const bool set_back = values.two_sided && values.front != values.back;
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           values.front);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, values.back);
                }
            });
            break;
        case G::StencilWriteMask:
            scheduler.Record([values = state.stencil_write_mask](vk::CommandBuffer cmdbuf) {
                const bool set_back = values.two_sided && values.front != values.back;
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           values.front);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, values.back);
                }
            });
            break;
        case G::StencilCompareMask:
            scheduler.Record([values = state.stencil_compare_mask](vk::CommandBuffer cmdbuf) {
                const bool set_back = values.two_sided && values.front != values.back;
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             values.front);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, values.back);
                }
            });
            break;
        case G::LineWidth:
            scheduler.Record(
                [width = state.line_width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
            break;
        case G::CullMode:
            scheduler.Record([mode = state.cull_mode](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetCullModeEXT(mode);
            });
            break;
        case G::DepthCompareOp:
            scheduler.Record([op = state.depth_compare_op](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthCompareOpEXT(op);
            });
            break;
        case G::FrontFace:
            scheduler.Record([front_face = state.front_face](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetFrontFaceEXT(front_face);
            });
            break;
        case G::StencilOp:
            scheduler.Record([ops = state.stencil_op](vk::CommandBuffer cmdbuf) {
                const auto set = [&](VkStencilFaceFlags faces, const G::StencilFaceOps& face) {
                    cmdbuf.SetStencilOpEXT(faces, face.fail, face.pass, face.depth_fail,
                                           face.compare);
                };
                if (ops.two_sided) {
                    set(VK_STENCIL_FACE_FRONT_BIT, ops.front);
                    set(VK_STENCIL_FACE_BACK_BIT, ops.back);
                } else {
                    // Front face defines the stencil op of both faces
                    set(VK_STENCIL_FACE_FRONT_AND_BACK, ops.front);
                }
            });
            break;
        case G::DepthBoundsTestEnable:
            scheduler.Record([enable = state.depth_bounds_test_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthBoundsTestEnableEXT(enable);
            });
            break;
        case G::DepthTestEnable:
            scheduler.Record([enable = state.depth_test_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthTestEnableEXT(enable);
            });
            break;
        case G::DepthWriteEnable:
            scheduler.Record([enable = state.depth_write_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthWriteEnableEXT(enable);
            });
            break;
        case G::StencilTestEnable:
            scheduler.Record([enable = state.stencil_test_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetStencilTestEnableEXT(enable);
            });
            break;
        case G::PrimitiveTopology:
            scheduler.Record([topology = state.primitive_topology](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology);
            });
            break;
        case G::PrimitiveRestartEnable:
            scheduler.Record([enable = state.primitive_restart_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveRestartEnableEXT(enable);
            });
            break;
        case G::RasterizerDiscardEnable:
            scheduler.Record([discard = state.rasterizer_discard_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetRasterizerDiscardEnableEXT(discard);
            });
            break;
        case G::DepthBiasEnable:
            scheduler.Record([enable = state.depth_bias_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthBiasEnableEXT(enable);
            });
            break;
        case G::LogicOp:
            scheduler.Record(
                [op = state.logic_op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
            break;
        case G::LogicOpEnable:
            scheduler.Record([enable = state.logic_op_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetLogicOpEnableEXT(enable);
            });
            break;
        case G::DepthClampEnable:
            scheduler.Record([enable = state.depth_clamp_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetDepthClampEnableEXT(enable);
            });
            break;
        case G::LineRasterizationMode:
            scheduler.Record([mode = state.line_rasterization_mode](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetLineRasterizationModeEXT(mode);
            });
            break;
        case G::LineStippleEnable:
            scheduler.Record([enable = state.line_stipple_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetLineStippleEnableEXT(enable);
            });
            break;
        case G::ConservativeRasterization:
            scheduler.Record([enable = state.conservative_rasterization](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetConservativeRasterizationModeEXT(
                    enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                           : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
            });
            break;
        case G::AlphaToCoverageEnable:
            scheduler.Record([enable = state.alpha_to_coverage_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
            });
            break;
        case G::AlphaToOneEnable:
            scheduler.Record([enable = state.alpha_to_one_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
            });
            break;
        case G::ColorWriteMask:
            scheduler.Record([masks = state.color_write_mask](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetColorWriteMaskEXT(0, masks);
            });
            break;
        case G::BlendEnable:
            scheduler.Record([enables = state.blend_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetColorBlendEnableEXT(0, enables);
            });
            break;
        case G::BlendEquations:
            scheduler.Record([blends = state.blend_equations](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetColorBlendEquationEXT(0, blends);
            });
            break;
        case G::ColorWriteEnable:
            scheduler.Record([enables = state.color_write_enable](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetColorWriteEnableEXT(enables);
            });
            break;
        case G::VertexInput: {
            const auto& input = state.vertex_input;
            const boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32>
                bindings(input.bindings.begin(), input.bindings.begin() + input.num_bindings);
            const boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32>
                attributes(input.attributes.begin(),
                           input.attributes.begin() + input.num_attributes);
            scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetVertexInputEXT(bindings, attributes);
            });
            break;
        }
        case G::Count:
            break;
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback(bool enabled) {
    static std::once_flag warn_unsupported;

    if (!device.IsExtTransformFeedbackSupported()) {
        if (enabled) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount, enabled);
    // Log extension usage for transform feedback
    if (enabled && GPU::Logging::IsActive()) {
        GPU::Logging::GPULogger::GetInstance().LogExtensionUsage("VK_EXT_transform_feedback",
                                                                 "HandleTransformFeedback");
    }
}

void RasterizerVulkan::VerifyDrawPacket(const DrawPacket& packet) {
    using G = DynamicStatePacket;
    std::vector<std::string> differences;
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (packet.is_indirect) {
        const auto& now = maxwell3d->draw_manager.indirect_state;
        const auto& was = packet.indirect;
        if (now.is_byte_count != was.is_byte_count || now.is_indexed != was.is_indexed ||
            now.include_count != was.include_count ||
            now.count_start_address != was.count_start_address ||
            now.indirect_start_address != was.indirect_start_address ||
            now.buffer_size != was.buffer_size || now.max_draw_counts != was.max_draw_counts ||
            now.stride != was.stride) {
            differences.push_back(fmt::format("indirect parameters: {:#x} in the packet, {:#x} now",
                                              was.indirect_start_address,
                                              now.indirect_start_address));
        }
    } else {
        const DrawParams now =
            MakeDrawParams(draw_state, packet.params.num_instances, packet.is_indexed);
        const DrawParams& was = packet.params;
        if (now.base_instance != was.base_instance || now.base_vertex != was.base_vertex ||
            now.num_vertices != was.num_vertices || now.first_index != was.first_index ||
            now.is_indexed != was.is_indexed) {
            differences.push_back(fmt::format(
                "draw parameters: {} from {} (base {}) in the packet, {} from {} (base {}) now",
                was.num_vertices, was.first_index, was.base_vertex, now.num_vertices,
                now.first_index, now.base_vertex));
        }
    }
    const auto& buffers = packet.buffers;
    const auto describe = [](const VideoCommon::Binding& binding) {
        return fmt::format("{:#x}+{:#x}", binding.device_addr, binding.size);
    };
    const auto same = [](const VideoCommon::Binding& lhs, const VideoCommon::Binding& rhs) {
        return lhs.device_addr == rhs.device_addr && lhs.size == rhs.size;
    };
    if (buffers.index_dirty) {
        const VideoCommon::Binding now = buffer_cache.ResolveIndexBuffer();
        if (!same(buffers.index, now)) {
            differences.push_back(fmt::format("index buffer: {} in the packet, {} now",
                                              describe(buffers.index), describe(now)));
        }
    }
    for (const auto& stream : buffers.vertex_streams) {
        const u32 bit = 1U << stream.index;
        if ((buffers.vertex_resolve & bit) != 0) {
            const VideoCommon::Binding now = buffer_cache.ResolveVertexBuffer(stream.index);
            if (!same(stream.binding, now)) {
                differences.push_back(fmt::format("vertex buffer {}: {} in the packet, {} now",
                                                  stream.index, describe(stream.binding),
                                                  describe(now)));
            }
        }
        if ((buffers.vertex_rebind & bit) != 0 &&
            stream.stride != regs.vertex_streams[stream.index].stride) {
            differences.push_back(fmt::format("vertex buffer {} stride: {} in the packet, {} now",
                                              stream.index, stream.stride,
                                              regs.vertex_streams[stream.index].stride.Value()));
        }
    }
    if (buffers.transform_feedback != (regs.transform_feedback_enabled != 0)) {
        differences.push_back("transform feedback");
    }
    if (packet.zpass_pixel_count != (regs.zpass_pixel_count_enable != 0)) {
        differences.push_back("zpass pixel count");
    }
    packet.pipeline->VerifyPacket(packet, differences);

    DynamicStateInputs inputs{
        .device = &device,
        .is_rescaling = texture_cache.IsRescaling(),
        .topology = draw_state.topology,
        .color_output_mask = packet.pipeline->ColorOutputMask(),
        .d24_depth_bias_rescale = !device.SupportsD24DepthBuffer() &&
                                  std::ranges::find(NEEDS_D24, program_id) !=
                                      std::ranges::end(NEEDS_D24),
        .repacked_vertex_buffers = buffer_cache_runtime.RepackedVertexBuffers(),
        .viewport_offset = oot_jitter_applied,
    };
    if (packet.dynamic.Has(G::AlphaToCoverageEnable) || packet.dynamic.Has(G::AlphaToOneEnable)) {
        inputs.lookup_pipeline = pipeline_cache.CurrentGraphicsPipeline();
    }
    G now;
    for (u64 mask = packet.dynamic.mask; mask != 0; mask &= mask - 1) {
        const auto group = static_cast<G::Group>(std::countr_zero(mask));
        ReadDynamicGroup(group, regs, inputs, now);
        if (!SameDynamicGroup(packet.dynamic, now, group)) {
            differences.push_back(fmt::format("dynamic state group {}", static_cast<u32>(group)));
        }
    }

    auto& stats = packet_verify;
    ++stats.draws;
    stats.bytes += packet.PayloadBytes();
    if (!differences.empty()) {
        ++stats.differing;
    }
    for (const std::string& difference : differences) {
        if (stats.logged >= 16) {
            break;
        }
        ++stats.logged;
        LOG_ERROR(Render_Vulkan, "Draw packet verify: draw {} (vs {:016x}, ps {:016x}): {}",
                  packet.sequence, packet.pipeline->ShaderHashes()[1],
                  packet.pipeline->ShaderHashes()[5], difference);
    }
    const auto now_time = std::chrono::steady_clock::now();
    if (now_time - stats.last_report >= std::chrono::seconds{10}) {
        stats.last_report = now_time;
        LOG_INFO(Render_Vulkan,
                 "Draw packet verify: {} draws checked, {} differ; {} bytes per packet on average",
                 stats.draws, stats.differing, stats.draws != 0 ? stats.bytes / stats.draws : 0);
    }
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Channel);
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Channel);
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::Channel);
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
