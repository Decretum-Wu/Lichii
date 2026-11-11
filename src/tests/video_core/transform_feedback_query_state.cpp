// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/transform_feedback_query_state.h"

namespace {

using State = Vulkan::TransformFeedbackQueryState;
using Maxwell = State::Maxwell;

TEST_CASE("Transform feedback query state captures independent buffer settings", "[draw_packet]") {
    Maxwell regs{};
    regs.transform_feedback_enabled = 1;
    regs.transform_feedback.buffers[1].enable = 1;
    regs.transform_feedback.controls[1].stream = 2;
    regs.transform_feedback.controls[1].stride = 48;
    regs.transform_feedback.buffers[3].enable = 1;
    regs.transform_feedback.controls[3].stream = 2;
    regs.transform_feedback.controls[3].stride = 96;
    const State state = State::FromRegs(regs, Maxwell::PrimitiveTopology::Triangles);
    regs = {};

    REQUIRE(state.enabled);
    REQUIRE(state.patch_vertices == 1);
    REQUIRE(state.output_topology == Maxwell::PrimitiveTopology::Triangles);
    REQUIRE_FALSE(state.buffers[0].enabled);
    REQUIRE(state.buffers[1].enabled);
    REQUIRE(state.buffers[1].stream == 2);
    REQUIRE(state.buffers[1].stride == 48);
    REQUIRE(state.buffers[3].enabled);
    REQUIRE(state.buffers[3].stream == 2);
    REQUIRE(state.buffers[3].stride == 96);
    REQUIRE_FALSE(State::FromRegs(regs, Maxwell::PrimitiveTopology::Points).enabled);
}

TEST_CASE("Transform feedback query state captures tessellation output topology", "[draw_packet]") {
    Maxwell regs{};
    regs.patch_vertices = 7;
    using Output = Maxwell::Tessellation::OutputPrimitives;
    for (const auto& [output, topology] : {
             std::pair{Output::Points, Maxwell::PrimitiveTopology::Points},
             std::pair{Output::Lines, Maxwell::PrimitiveTopology::LineStrip},
             std::pair{Output::Triangles_CW, Maxwell::PrimitiveTopology::TriangleStrip},
             std::pair{Output::Triangles_CCW, Maxwell::PrimitiveTopology::TriangleStrip},
         }) {
        regs.tessellation.params.output_primitives.Assign(output);
        const State state = State::FromRegs(regs, Maxwell::PrimitiveTopology::Patches);
        REQUIRE(state.output_topology == topology);
        REQUIRE(state.patch_vertices == 7);
        REQUIRE(State::FromRegs(regs, Maxwell::PrimitiveTopology::LineLoop).output_topology ==
                Maxwell::PrimitiveTopology::LineLoop);
    }
}

} // namespace
