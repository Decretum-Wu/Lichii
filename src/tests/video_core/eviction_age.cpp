// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/eviction_age.h"

namespace {

constexpr u64 MiB = u64{1} << 20;

} // namespace

TEST_CASE("The collector keeps recently used images until the device runs short",
          "[video_core][texture_cache][eviction_age]") {
    using VideoCommon::EvictionAge;
    using VideoCommon::MinEvictionAge;
    // Mario Kart World's race: 6268 MiB in use, past the 5120 MiB critical threshold, under the
    // 7678 MiB budget. Aggressive (10), high priority (25) and normal (50) modes all wait.
    for (const u64 mode_age : {u64{10}, u64{25}, u64{50}}) {
        REQUIRE(EvictionAge(mode_age, 6268 * MiB, 7678 * MiB, false) == MinEvictionAge);
    }
    REQUIRE(EvictionAge(600, 6268 * MiB, 7678 * MiB, false) == 600);
    // At the budget, after a failed image, or with no budget to compare: the mode's own age.
    REQUIRE(EvictionAge(10, 7678 * MiB, 7678 * MiB, false) == 10);
    REQUIRE(EvictionAge(10, 7900 * MiB, 7678 * MiB, false) == 10);
    REQUIRE(EvictionAge(25, 6268 * MiB, 7678 * MiB, true) == 25);
    REQUIRE(EvictionAge(50, 6268 * MiB, 0, false) == 50);
}
