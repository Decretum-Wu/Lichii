// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Vulkan {

struct TransformFeedbackQueryState {
    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    struct Buffer {
        bool enabled{};
        u32 stream{};
        u32 stride{};
    };

    bool enabled{};
    Maxwell::PrimitiveTopology output_topology{};
    u32 patch_vertices{};
    std::array<Buffer, Maxwell::NumTransformFeedbackBuffers> buffers{};

    [[nodiscard]] static TransformFeedbackQueryState FromRegs(
        const Maxwell& regs, Maxwell::PrimitiveTopology topology) {
        TransformFeedbackQueryState state{
            .enabled = regs.transform_feedback_enabled != 0,
            .output_topology = topology,
            .patch_vertices = (std::max)(regs.patch_vertices, 1U),
        };
        if (topology == Maxwell::PrimitiveTopology::Patches) {
            switch (regs.tessellation.params.output_primitives.Value()) {
            case Maxwell::Tessellation::OutputPrimitives::Points:
                state.output_topology = Maxwell::PrimitiveTopology::Points;
                break;
            case Maxwell::Tessellation::OutputPrimitives::Lines:
                state.output_topology = Maxwell::PrimitiveTopology::LineStrip;
                break;
            case Maxwell::Tessellation::OutputPrimitives::Triangles_CW:
            case Maxwell::Tessellation::OutputPrimitives::Triangles_CCW:
                state.output_topology = Maxwell::PrimitiveTopology::TriangleStrip;
                break;
            }
        }
        for (size_t index = 0; index < state.buffers.size(); ++index) {
            state.buffers[index] = {
                .enabled = regs.transform_feedback.buffers[index].enable != 0,
                .stream = regs.transform_feedback.controls[index].stream,
                .stride = regs.transform_feedback.controls[index].stride,
            };
        }
        return state;
    }
};

} // namespace Vulkan
