// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/transform_feedback_query_state.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Framebuffer;
class GraphicsPipeline;

struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

/// The dynamic state groups whose dirty flags stage A took, with the values stage B records.
struct DynamicStatePacket {
    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    enum Group : u32 {
        Viewports,
        Scissors,
        DepthBias,
        BlendConstants,
        DepthBounds,
        StencilReference,
        StencilWriteMask,
        StencilCompareMask,
        LineWidth,
        CullMode,
        DepthCompareOp,
        FrontFace,
        StencilOp,
        DepthBoundsTestEnable,
        DepthTestEnable,
        DepthWriteEnable,
        StencilTestEnable,
        PrimitiveTopology,
        PrimitiveRestartEnable,
        RasterizerDiscardEnable,
        DepthBiasEnable,
        LogicOp,
        LogicOpEnable,
        DepthClampEnable,
        LineRasterizationMode,
        LineStippleEnable,
        ConservativeRasterization,
        AlphaToCoverageEnable,
        AlphaToOneEnable,
        ColorWriteMask,
        BlendEnable,
        BlendEquations,
        ColorWriteEnable,
        VertexInput,
        Count,
    };
    static_assert(Count <= 64);

    struct DepthBiasValues {
        f32 constant;
        f32 clamp;
        f32 slope;
    };
    /// A stencil reference or mask: the back face's value applies when two-sided and different.
    struct StencilValues {
        u32 front;
        u32 back;
        bool two_sided;
    };
    struct StencilFaceOps {
        VkStencilOp fail;
        VkStencilOp pass;
        VkStencilOp depth_fail;
        VkCompareOp compare;
    };
    /// The front ops apply to both faces unless two-sided.
    struct StencilOps {
        StencilFaceOps front;
        StencilFaceOps back;
        bool two_sided;
    };
    struct VertexInputState {
        std::array<VkVertexInputBindingDescription2EXT, Maxwell::NumVertexArrays> bindings;
        std::array<VkVertexInputAttributeDescription2EXT, Maxwell::NumVertexAttributes> attributes;
        u32 num_bindings;
        u32 num_attributes;
    };

    void Set(Group group) noexcept {
        mask |= u64{1} << group;
    }

    [[nodiscard]] bool Has(Group group) const noexcept {
        return ((mask >> group) & 1) != 0;
    }

    u64 mask{};
    std::array<VkViewport, Maxwell::NumViewports> viewports{};
    std::array<VkRect2D, Maxwell::NumViewports> scissors{};
    DepthBiasValues depth_bias{};
    std::array<f32, 4> blend_constants{};
    std::array<f32, 2> depth_bounds{};
    StencilValues stencil_reference{};
    StencilValues stencil_write_mask{};
    StencilValues stencil_compare_mask{};
    f32 line_width{};
    VkCullModeFlags cull_mode{};
    VkCompareOp depth_compare_op{};
    VkFrontFace front_face{};
    StencilOps stencil_op{};
    bool depth_bounds_test_enable{};
    bool depth_test_enable{};
    bool depth_write_enable{};
    bool stencil_test_enable{};
    VkPrimitiveTopology primitive_topology{};
    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    bool depth_bias_enable{};
    VkLogicOp logic_op{};
    bool logic_op_enable{};
    bool depth_clamp_enable{};
    VkLineRasterizationModeEXT line_rasterization_mode{};
    bool line_stipple_enable{};
    bool conservative_rasterization{};
    bool alpha_to_coverage_enable{};
    bool alpha_to_one_enable{};
    std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> color_write_mask{};
    std::array<VkBool32, Maxwell::NumRenderTargets> blend_enable{};
    std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> blend_equations{};
    std::array<VkBool32, Maxwell::NumRenderTargets> color_write_enable{};
    VertexInputState vertex_input{};
};

namespace DrawPacketDetail {

template <typename T>
[[nodiscard]] bool Same(const T& lhs, const T& rhs) noexcept {
    if constexpr (std::is_same_v<T, f32>) {
        return std::bit_cast<u32>(lhs) == std::bit_cast<u32>(rhs);
    } else {
        return lhs == rhs;
    }
}

[[nodiscard]] inline bool Same(const VkViewport& lhs, const VkViewport& rhs) noexcept {
    return Same(lhs.x, rhs.x) && Same(lhs.y, rhs.y) && Same(lhs.width, rhs.width) &&
           Same(lhs.height, rhs.height) && Same(lhs.minDepth, rhs.minDepth) &&
           Same(lhs.maxDepth, rhs.maxDepth);
}

[[nodiscard]] inline bool Same(const VkRect2D& lhs, const VkRect2D& rhs) noexcept {
    return lhs.offset.x == rhs.offset.x && lhs.offset.y == rhs.offset.y &&
           lhs.extent.width == rhs.extent.width && lhs.extent.height == rhs.extent.height;
}

[[nodiscard]] inline bool Same(const VkColorBlendEquationEXT& lhs,
                               const VkColorBlendEquationEXT& rhs) noexcept {
    return lhs.srcColorBlendFactor == rhs.srcColorBlendFactor &&
           lhs.dstColorBlendFactor == rhs.dstColorBlendFactor &&
           lhs.colorBlendOp == rhs.colorBlendOp &&
           lhs.srcAlphaBlendFactor == rhs.srcAlphaBlendFactor &&
           lhs.dstAlphaBlendFactor == rhs.dstAlphaBlendFactor &&
           lhs.alphaBlendOp == rhs.alphaBlendOp;
}

[[nodiscard]] inline bool Same(const DynamicStatePacket::DepthBiasValues& lhs,
                               const DynamicStatePacket::DepthBiasValues& rhs) noexcept {
    return Same(lhs.constant, rhs.constant) && Same(lhs.clamp, rhs.clamp) &&
           Same(lhs.slope, rhs.slope);
}

[[nodiscard]] inline bool Same(const DynamicStatePacket::StencilValues& lhs,
                               const DynamicStatePacket::StencilValues& rhs) noexcept {
    return lhs.front == rhs.front && lhs.back == rhs.back && lhs.two_sided == rhs.two_sided;
}

[[nodiscard]] inline bool Same(const DynamicStatePacket::StencilFaceOps& lhs,
                               const DynamicStatePacket::StencilFaceOps& rhs) noexcept {
    return lhs.fail == rhs.fail && lhs.pass == rhs.pass && lhs.depth_fail == rhs.depth_fail &&
           lhs.compare == rhs.compare;
}

[[nodiscard]] inline bool Same(const DynamicStatePacket::StencilOps& lhs,
                               const DynamicStatePacket::StencilOps& rhs) noexcept {
    return lhs.two_sided == rhs.two_sided && Same(lhs.front, rhs.front) &&
           (!lhs.two_sided || Same(lhs.back, rhs.back));
}

[[nodiscard]] inline bool Same(const DynamicStatePacket::VertexInputState& lhs,
                               const DynamicStatePacket::VertexInputState& rhs) noexcept {
    if (lhs.num_bindings != rhs.num_bindings || lhs.num_attributes != rhs.num_attributes) {
        return false;
    }
    for (u32 index = 0; index < lhs.num_bindings; ++index) {
        const auto& a = lhs.bindings[index];
        const auto& b = rhs.bindings[index];
        if (a.binding != b.binding || a.stride != b.stride || a.inputRate != b.inputRate ||
            a.divisor != b.divisor) {
            return false;
        }
    }
    for (u32 index = 0; index < lhs.num_attributes; ++index) {
        const auto& a = lhs.attributes[index];
        const auto& b = rhs.attributes[index];
        if (a.location != b.location || a.binding != b.binding || a.format != b.format ||
            a.offset != b.offset) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t N>
[[nodiscard]] bool Same(const std::array<T, N>& lhs, const std::array<T, N>& rhs) noexcept {
    for (size_t index = 0; index < N; ++index) {
        if (!Same(lhs[index], rhs[index])) {
            return false;
        }
    }
    return true;
}

struct GroupOps {
    u32 bytes;
    void (*copy)(DynamicStatePacket&, const DynamicStatePacket&);
    bool (*same)(const DynamicStatePacket&, const DynamicStatePacket&);
};

template <auto Member>
constexpr GroupOps MakeGroupOps() {
    using T = std::remove_cvref_t<decltype(std::declval<DynamicStatePacket&>().*Member)>;
    return {
        .bytes = static_cast<u32>(sizeof(T)),
        .copy = [](DynamicStatePacket& to, const DynamicStatePacket& from) {
            to.*Member = from.*Member;
        },
        .same = [](const DynamicStatePacket& lhs, const DynamicStatePacket& rhs) {
            return Same(lhs.*Member, rhs.*Member);
        },
    };
}

using P = DynamicStatePacket;
inline constexpr std::array<GroupOps, P::Count> GROUP_OPS{
    MakeGroupOps<&P::viewports>(),
    MakeGroupOps<&P::scissors>(),
    MakeGroupOps<&P::depth_bias>(),
    MakeGroupOps<&P::blend_constants>(),
    MakeGroupOps<&P::depth_bounds>(),
    MakeGroupOps<&P::stencil_reference>(),
    MakeGroupOps<&P::stencil_write_mask>(),
    MakeGroupOps<&P::stencil_compare_mask>(),
    MakeGroupOps<&P::line_width>(),
    MakeGroupOps<&P::cull_mode>(),
    MakeGroupOps<&P::depth_compare_op>(),
    MakeGroupOps<&P::front_face>(),
    MakeGroupOps<&P::stencil_op>(),
    MakeGroupOps<&P::depth_bounds_test_enable>(),
    MakeGroupOps<&P::depth_test_enable>(),
    MakeGroupOps<&P::depth_write_enable>(),
    MakeGroupOps<&P::stencil_test_enable>(),
    MakeGroupOps<&P::primitive_topology>(),
    MakeGroupOps<&P::primitive_restart_enable>(),
    MakeGroupOps<&P::rasterizer_discard_enable>(),
    MakeGroupOps<&P::depth_bias_enable>(),
    MakeGroupOps<&P::logic_op>(),
    MakeGroupOps<&P::logic_op_enable>(),
    MakeGroupOps<&P::depth_clamp_enable>(),
    MakeGroupOps<&P::line_rasterization_mode>(),
    MakeGroupOps<&P::line_stipple_enable>(),
    MakeGroupOps<&P::conservative_rasterization>(),
    MakeGroupOps<&P::alpha_to_coverage_enable>(),
    MakeGroupOps<&P::alpha_to_one_enable>(),
    MakeGroupOps<&P::color_write_mask>(),
    MakeGroupOps<&P::blend_enable>(),
    MakeGroupOps<&P::blend_equations>(),
    MakeGroupOps<&P::color_write_enable>(),
    MakeGroupOps<&P::vertex_input>(),
};

} // namespace DrawPacketDetail

/// Copies the groups of `from` into `to` and adds them to its mask (stage B's record of the
/// latest values).
inline void MergeDynamicState(DynamicStatePacket& to, const DynamicStatePacket& from) {
    for (u64 mask = from.mask; mask != 0; mask &= mask - 1) {
        DrawPacketDetail::GROUP_OPS[std::countr_zero(mask)].copy(to, from);
    }
    to.mask |= from.mask;
}

/// Whether group `group` holds the same values in both.
[[nodiscard]] inline bool SameDynamicGroup(const DynamicStatePacket& lhs,
                                           const DynamicStatePacket& rhs,
                                           DynamicStatePacket::Group group) {
    return DrawPacketDetail::GROUP_OPS[group].same(lhs, rhs);
}

/// Bytes of the values of the groups in the mask; the vertex input counts its used entries.
[[nodiscard]] inline u32 DynamicStatePayloadBytes(const DynamicStatePacket& state) {
    u32 bytes = sizeof(state.mask);
    for (u64 mask = state.mask; mask != 0; mask &= mask - 1) {
        const auto group = static_cast<DynamicStatePacket::Group>(std::countr_zero(mask));
        if (group == DynamicStatePacket::VertexInput) {
            bytes += 2 * sizeof(u32) +
                     state.vertex_input.num_bindings *
                         static_cast<u32>(sizeof(VkVertexInputBindingDescription2EXT)) +
                     state.vertex_input.num_attributes *
                         static_cast<u32>(sizeof(VkVertexInputAttributeDescription2EXT));
            continue;
        }
        bytes += DrawPacketDetail::GROUP_OPS[group].bytes;
    }
    return bytes;
}

/// An emulated mesh draw's compute dispatch, as stage A gave it to the pipeline for the draw.
struct MeshEmulationDispatch {
    struct Buffer {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };
    std::array<Buffer, 4> task_buffers{}; ///< Payload, offsets, arguments, arena counter
    u32 groups{};
    u32 capacity{}; ///< Words of the output arena (with a task program)
};

/// One draw, built by stage A from the registers, dirty flags and memory as they are at the
/// draw, and recorded by stage B from it alone (docs/nx2/cmdlist.md).
struct DrawPacket {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;
    using Range = std::pair<u32, u32>; ///< First entry and count

    u64 sequence{};
    GraphicsPipeline* pipeline{};
    const Framebuffer* framebuffer{};
    bool is_indexed{};
    bool is_indirect{};
    bool mesh_draw{};
    DrawParams params{};
    Tegra::Engines::Maxwell3D::DrawManager::IndirectParams indirect{};
    /// Written for mesh draws only.
    u32 mesh_emulation_groups{};
    u32 mesh_emulation_vertices_per_group{}; ///< 0 unless an emulated mesh draw
    VkBuffer mesh_emulation_indirect_buffer{};
    VkDeviceSize mesh_emulation_indirect_offset{};
    MeshEmulationDispatch mesh_dispatch{};
    /// With stage B on its own thread: the vertex streams the runtime repacks for the draw.
    u32 vertex_repack_mask{};

    VideoCommon::GraphicsBufferPacket buffers;

    /// A writes the draw's descriptor entries into the descriptor queue and leaves these, counted
    /// from its UpdateData, for B's storage and texel buffer entries of each stage.
    std::array<Range, NUM_STAGES> buffer_entries{};
    DescriptorUpdateEntry* descriptor_data{}; ///< A's UpdateData for the draw
    u32 descriptor_count{};                   ///< Entries A left there, B's included
    u32 stages{}; ///< Stages B binds descriptors for
    std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling{};
    RenderAreaPushConstant render_area{};
    bool is_rescaling{};
    bool samples_render_target{}; ///< A fragment barrier is pending and the draw samples a target
    bool zpass_pixel_count{};
    TransformFeedbackQueryState query_tfb{};

    DynamicStatePacket dynamic;

    /// Bytes stage A wrote into the packet for this draw: the fields the draw uses, its lists
    /// counted by their used entries.
    [[nodiscard]] u32 PayloadBytes() const {
        using Buffers = VideoCommon::GraphicsBufferPacket;
        size_t header = sizeof(DrawPacket) - sizeof(buffers) - sizeof(dynamic) +
                        sizeof(Buffers) - sizeof(buffers.inline_indices) -
                        sizeof(buffers.vertex_streams) - sizeof(buffers.storage_buffers) -
                        sizeof(buffers.texture_buffers);
        header -= is_indirect ? sizeof(params) : sizeof(indirect);
        if (!mesh_draw) {
            header -= sizeof(mesh_emulation_groups) + sizeof(mesh_emulation_vertices_per_group) +
                      sizeof(mesh_emulation_indirect_buffer) +
                      sizeof(mesh_emulation_indirect_offset) + sizeof(mesh_dispatch);
        }
        if (!buffers.is_indexed) {
            header -= sizeof(buffers.index_format) + sizeof(buffers.index_format_size) +
                      sizeof(buffers.index_first) + sizeof(buffers.index_count) +
                      sizeof(buffers.index_guest_size);
        }
        if (!buffers.index_dirty) {
            header -= sizeof(buffers.index);
        }
        if (!buffers.transform_feedback) {
            header -= sizeof(buffers.transform_feedback_layouts) +
                      sizeof(buffers.transform_feedback_buffers);
        }
        if (!buffers.indirect) {
            header -= sizeof(buffers.indirect_count) + sizeof(buffers.indirect_buffer) +
                      sizeof(buffers.count_buffer);
        }
        const size_t lists =
            buffers.inline_indices.size() +
            buffers.vertex_streams.size() * sizeof(Buffers::VertexStream) +
            buffers.storage_buffers.size() * sizeof(Buffers::StorageBuffer) +
            buffers.texture_buffers.size() * sizeof(Buffers::TextureBuffer);
        return static_cast<u32>(header + lists) + DynamicStatePayloadBytes(dynamic);
    }
};

} // namespace Vulkan
