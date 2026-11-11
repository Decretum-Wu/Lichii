// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_draw_stages.h"

namespace {

namespace DrawStages = VideoCommon::DrawStages;

TEST_CASE("Draw stages: writes join only over ranges the packets in flight read",
          "[draw_stages]") {
    DrawStages::OverlapRing ring;
    ring.Read(DrawStages::IndexSlot, 0x10000, 0x100, 0);
    ring.Read(DrawStages::VertexSlot(0), 0x40000, 0x1000, 1);
    REQUIRE(ring.Overlaps(0x10080, 4, 2));
    REQUIRE(ring.Overlaps(0x40ffc, 8, 2));
    REQUIRE_FALSE(ring.Overlaps(0x10100, 0x100, 2));
    REQUIRE_FALSE(ring.Overlaps(0x20000, 0x100, 2));
    // Once Window more packets completed, the first packet is no longer in flight.
    REQUIRE_FALSE(ring.Overlaps(0x10080, 4, DrawStages::OverlapRing::Window + 1));
    REQUIRE(ring.Overlaps(0x40000, 4, DrawStages::OverlapRing::Window + 1));
    // A binding read again by a later packet stays in flight.
    ring.Read(DrawStages::IndexSlot, 0x10000, 0x100, 100);
    REQUIRE(ring.Overlaps(0x10000, 4, 120));
    REQUIRE(ring.Live(120) == 1);
}

TEST_CASE("Draw stages: a full overlap ring conservatively joins until the lost read expires",
          "[draw_stages]") {
    DrawStages::OverlapRing ring;
    for (size_t i = 0; i <= DrawStages::OverlapRing::Capacity; ++i) {
        ring.Read(DrawStages::IndexSlot, 0x10000 + i * 0x10000, 4, 0);
    }
    REQUIRE(ring.overflows == 1);
    REQUIRE(ring.Overlaps(0x80000000, 4, DrawStages::OverlapRing::Window));
    REQUIRE_FALSE(ring.Overlaps(0x80000000, 0, DrawStages::OverlapRing::Window));
    REQUIRE_FALSE(ring.Overlaps(0x80000000, 4, DrawStages::OverlapRing::Window + 1));
}

TEST_CASE("Draw stages: owner checks, join counts and the profile line", "[draw_stages]") {
    DrawStages::g_counting = true;
    DrawStages::g_checking = true;
    for (auto& counter : DrawStages::g_counters.joins) {
        counter = 0;
    }
    DrawStages::g_counters.violations = 0;

    DrawStages::Enter(DrawStages::Stage::A);
    DrawStages::Expect(DrawStages::Stage::A, "test A");
    REQUIRE(DrawStages::g_counters.violations == 0);
    DrawStages::Expect(DrawStages::Stage::B, "test B");
    REQUIRE(DrawStages::g_counters.violations == 1);
    {
        // A join holds both stages' state.
        const DrawStages::JoinScope join{DrawStages::Join::ASubmit};
        DrawStages::Expect(DrawStages::Stage::B, "test join");
        DrawStages::CountInA(DrawStages::Join::ASchedulerRead);
    }
    REQUIRE(DrawStages::g_counters.violations == 1);
    REQUIRE(DrawStages::In(DrawStages::Stage::A));
    DrawStages::CountInA(DrawStages::Join::ASchedulerRead);
    const auto joins = [](DrawStages::Join site) {
        return DrawStages::g_counters.joins[static_cast<size_t>(site)].load();
    };
    REQUIRE(joins(DrawStages::Join::ASubmit) == 1);
    REQUIRE(joins(DrawStages::Join::ASchedulerRead) == 1);

    DrawStages::Enter(DrawStages::Stage::B);
    DrawStages::NoteRead(DrawStages::IndexSlot, 0x80000, 0x40);
    DrawStages::EndPacket();
    REQUIRE_FALSE(DrawStages::In(DrawStages::Stage::B));
    DrawStages::Expect(DrawStages::Stage::A, "between draws");
    REQUIRE(DrawStages::g_counters.violations == 1);
    DrawStages::NoteWrite(DrawStages::Join::CbData, 0x80020, 4);
    DrawStages::NoteWrite(DrawStages::Join::CbData, 0x90000, 4);
    REQUIRE(joins(DrawStages::Join::CbData) == 1);

    const std::string line = DrawStages::FormatJoins(1'000'000'000);
    REQUIRE(line.find("CB_DATA 1 of 2") != std::string::npos);
    REQUIRE(line.find("A submits 1") != std::string::npos);
    REQUIRE(line.find("owner violations 1") != std::string::npos);
    REQUIRE(joins(DrawStages::Join::CbData) == 0);

    DrawStages::g_counting = false;
    DrawStages::g_checking = false;
    DrawStages::t_stage = DrawStages::Stage::Outside;
}

} // namespace
