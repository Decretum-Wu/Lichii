// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>
#include <utility>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include "video_core/host1x/vic_geometry.h"

using Tegra::Host1x::VicGeometry::BlendInput;
using Tegra::Host1x::VicGeometry::BlendRects;
using Tegra::Host1x::VicGeometry::FitBlend;

namespace {

/// Blend's rectangles as base computed them, before any clamp to the slot surface.
BlendRects BaseRects(const BlendInput& in) {
    BlendRects r{};
    r.rect_left = (std::max)(in.target_left, in.dest_left);
    r.rect_right = (std::min)(in.target_right, in.dest_right);
    r.source_left = (std::max)(in.source_left, r.rect_left);
    r.source_right = (std::min)({in.source_right, r.rect_right, in.out_width});
    r.source_top = (std::max)({in.source_top, in.target_top, in.dest_top});
    r.source_bottom =
        (std::min)({in.source_bottom, in.target_bottom, in.dest_bottom, in.out_height});
    return r;
}

BlendInput Whole(u32 width, u32 height, u32 slot_width, u32 slot_rows) {
    return {.source_left = 0, .source_right = width, .source_top = 0, .source_bottom = height,
            .dest_left = 0, .dest_right = width, .dest_top = 0, .dest_bottom = height,
            .target_left = 0, .target_right = width, .target_top = 0, .target_bottom = height,
            .out_width = width, .out_height = height,
            .slot_width = slot_width, .slot_rows = slot_rows};
}

} // namespace

TEST_CASE("Vic draws a slot that fits its surfaces as it always has",
          "[video_core][vic_geometry]") {
    const auto fit = FitBlend(Whole(1280, 720, 1280, 720));
    REQUIRE(fit.has_value());
    CHECK(fit->source_left == 0);
    CHECK(fit->source_right == 1280);
    CHECK(fit->source_top == 0);
    CHECK(fit->source_bottom == 720);
    CHECK(fit->matrix_right == 1280);
}

TEST_CASE("Vic draws the part of a slot that overlaps a smaller slot surface",
          "[video_core][vic_geometry]") {
    // A 1920x1080 output from a 1280x720 slot: base copied 1:1 and read past the slot surface.
    // The overlapping part is drawn, nothing outside the slot surface is read.
    const auto fit = FitBlend(Whole(1920, 1080, 1280, 720));
    REQUIRE(fit.has_value());
    CHECK(fit->source_right == 1280);
    CHECK(fit->source_bottom == 720);
}

TEST_CASE("Vic draws nothing for an empty or inverted slot", "[video_core][vic_geometry]") {
    auto inverted = Whole(640, 480, 640, 480);
    inverted.source_left = 500;
    inverted.source_right = 100;
    CHECK_FALSE(FitBlend(inverted).has_value());
    auto past = Whole(640, 480, 320, 240);
    past.source_left = 400; // Left edge past the slot surface's right edge.
    past.target_left = 400;
    past.dest_left = 400;
    CHECK_FALSE(FitBlend(past).has_value());
    auto empty = Whole(640, 480, 640, 480);
    empty.source_top = empty.source_bottom = 10;
    CHECK_FALSE(FitBlend(empty).has_value());
}

TEST_CASE("Every fitted slot stays within both surfaces on both blend paths",
          "[video_core][vic_geometry]") {
    std::mt19937 rng{20261126};
    const auto below = [&](u32 limit) { return static_cast<u32>(rng() % limit); };
    u32 drawn = 0;
    u32 compared = 0;
    u32 matrix_compared = 0;
    u32 refused = 0;
    constexpr int Configurations = 200'000;
    for (int i = 0; i < Configurations; ++i) {
        const u32 out_w = 1 + below(2048), out_h = 1 + below(2048);
        const u32 slot_w = 1 + below(2048), slot_rows = 1 + below(2048);
        // Three in four configurations are a video's: destination and target near the origin, the
        // source about the slot's size. The others let every edge range over its surface.
        // Each of the inversion classes is generated, and 1 in 8 sources take the 30-bit fields'
        // large values.
        const bool video = below(4) != 0;
        const auto edges = [&](u32& lo, u32& hi, u32 size) {
            if (video) {
                lo = below(16);
                hi = lo + below(size + size / 4 + 1);
            } else {
                lo = below(size + 1);
                hi = lo + below(size - lo + size / 4 + 1);
            }
        };
        BlendInput in{.out_width = out_w, .out_height = out_h, .slot_width = slot_w,
                      .slot_rows = slot_rows};
        edges(in.dest_left, in.dest_right, out_w);
        edges(in.dest_top, in.dest_bottom, out_h);
        edges(in.target_left, in.target_right, out_w);
        edges(in.target_top, in.target_bottom, out_h);
        edges(in.source_left, in.source_right, slot_w);
        edges(in.source_top, in.source_bottom, slot_rows);
        if (below(8) == 0) {
            in.source_right = below(1U << 30);
            in.source_bottom = below(1U << 30);
        }
        // One rectangle's one axis inverted, in 1 of 4 configurations.
        switch (below(24)) {
        case 0: std::swap(in.source_left, in.source_right); break;
        case 1: std::swap(in.source_top, in.source_bottom); break;
        case 2: std::swap(in.dest_left, in.dest_right); break;
        case 3: std::swap(in.dest_top, in.dest_bottom); break;
        case 4: std::swap(in.target_left, in.target_right); break;
        case 5: std::swap(in.target_top, in.target_bottom); break;
        default: break;
        }
        const auto fit = FitBlend(in);
        const auto base = BaseRects(in);
        if (!fit) {
            // Nothing drawn only where base's rectangle, kept to the slot surface, is empty or
            // inverted.
            ++refused;
            const u32 right = (std::min)(base.source_right, slot_w);
            const u32 bottom = (std::min)(base.source_bottom, slot_rows);
            REQUIRE((base.source_left >= right || base.source_top >= bottom));
            continue;
        }
        ++drawn;
        const auto& r = *fit;
        REQUIRE(r.source_left < r.source_right);
        REQUIRE(r.source_top < r.source_bottom);
        // Reads of the slot surface, rows and columns.
        REQUIRE(r.source_right <= slot_w);
        REQUIRE(r.source_bottom <= slot_rows);
        REQUIRE(r.source_bottom <= out_h);
        // The copy path writes min(source width, target width) pixels from rect_left.
        const u32 copy = (std::min)(r.source_right - r.source_left, r.rect_right - r.rect_left);
        REQUIRE(u64{r.rect_left} + copy <= out_w);
        // The colour-matrix path reads at source_left + x and writes at rect_left + x, x from
        // source_left below matrix_right.
        if (r.source_left < r.matrix_right) {
            REQUIRE(u64{r.source_left} + r.matrix_right <= slot_w);
            REQUIRE(u64{r.rect_left} + r.matrix_right <= out_w);
        }
        // Where base drew within the slot surface, the slot is drawn exactly as before, and the
        // colour-matrix path ends where base's did where base's stayed within both rows.
        if (base.source_right <= slot_w && base.source_bottom <= slot_rows) {
            REQUIRE(r.source_left == base.source_left);
            REQUIRE(r.source_right == base.source_right);
            REQUIRE(r.source_top == base.source_top);
            REQUIRE(r.source_bottom == base.source_bottom);
            REQUIRE(r.rect_left == base.rect_left);
            REQUIRE(r.rect_right == base.rect_right);
            ++compared;
            if (u64{base.source_left} + base.source_right <= slot_w &&
                u64{base.rect_left} + base.source_right <= out_w) {
                REQUIRE(r.matrix_right == base.source_right);
                ++matrix_compared;
            }
        }
    }
    // Most configurations are drawn, and many are compared with base on both paths.
    CHECK(drawn > Configurations / 2);
    CHECK(compared > Configurations / 4);
    CHECK(matrix_compared > Configurations / 10);
    CHECK(refused > 0);
}
