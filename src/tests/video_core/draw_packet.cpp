// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/vk_draw_packet.h"

namespace {

using Vulkan::DynamicStatePacket;

TEST_CASE("Draw packet dynamic state merges and compares only the groups it carries",
          "[draw_packet]") {
    DynamicStatePacket taken;
    taken.Set(DynamicStatePacket::LineWidth);
    taken.line_width = 2.0f;
    taken.Set(DynamicStatePacket::DepthTestEnable);
    taken.depth_test_enable = true;
    taken.blend_constants = {1.0f, 1.0f, 1.0f, 1.0f};

    DynamicStatePacket recorded;
    Vulkan::MergeDynamicState(recorded, taken);
    REQUIRE(recorded.Has(DynamicStatePacket::LineWidth));
    REQUIRE(recorded.Has(DynamicStatePacket::DepthTestEnable));
    REQUIRE_FALSE(recorded.Has(DynamicStatePacket::BlendConstants));
    REQUIRE(recorded.line_width == 2.0f);
    REQUIRE(recorded.blend_constants[0] == 0.0f);
    REQUIRE(Vulkan::SameDynamicGroup(recorded, taken, DynamicStatePacket::LineWidth));

    // Values compare bit for bit: -0 is another value than +0.
    DynamicStatePacket zero;
    DynamicStatePacket negative_zero;
    negative_zero.line_width = -0.0f;
    REQUIRE_FALSE(Vulkan::SameDynamicGroup(zero, negative_zero, DynamicStatePacket::LineWidth));

    // The back face's stencil ops count only when two-sided.
    DynamicStatePacket one_sided;
    one_sided.stencil_op.back.fail = VK_STENCIL_OP_INVERT;
    REQUIRE(Vulkan::SameDynamicGroup(zero, one_sided, DynamicStatePacket::StencilOp));
    one_sided.stencil_op.two_sided = true;
    REQUIRE_FALSE(Vulkan::SameDynamicGroup(zero, one_sided, DynamicStatePacket::StencilOp));
}

TEST_CASE("Draw packet dynamic state counts the bytes of the groups it carries",
          "[draw_packet]") {
    DynamicStatePacket state;
    REQUIRE(Vulkan::DynamicStatePayloadBytes(state) == sizeof(u64));
    state.Set(DynamicStatePacket::LineWidth);
    state.Set(DynamicStatePacket::Viewports);
    REQUIRE(Vulkan::DynamicStatePayloadBytes(state) ==
            sizeof(u64) + sizeof(f32) + sizeof(state.viewports));

    // The vertex input counts its used entries; entries past them don't change the values.
    DynamicStatePacket input;
    input.Set(DynamicStatePacket::VertexInput);
    input.vertex_input.num_bindings = 2;
    input.vertex_input.num_attributes = 1;
    REQUIRE(Vulkan::DynamicStatePayloadBytes(input) ==
            sizeof(u64) + 2 * sizeof(u32) + 2 * sizeof(VkVertexInputBindingDescription2EXT) +
                sizeof(VkVertexInputAttributeDescription2EXT));
    DynamicStatePacket stale = input;
    stale.vertex_input.bindings[5].stride = 64;
    REQUIRE(Vulkan::SameDynamicGroup(input, stale, DynamicStatePacket::VertexInput));
    stale.vertex_input.bindings[1].stride = 64;
    REQUIRE_FALSE(Vulkan::SameDynamicGroup(input, stale, DynamicStatePacket::VertexInput));
}

TEST_CASE("Draw packets count their lists by the entries they use", "[draw_packet]") {
    Vulkan::DrawPacket packet;
    const u32 empty = packet.PayloadBytes();
    packet.buffers.storage_buffers.reserve(64);
    REQUIRE(packet.PayloadBytes() == empty);
    packet.buffers.storage_buffers.emplace_back();
    REQUIRE(packet.PayloadBytes() ==
            empty + sizeof(VideoCommon::GraphicsBufferPacket::StorageBuffer));
}

} // namespace
