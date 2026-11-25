// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_view_slots.h"

TEST_CASE("An erased image view stays stale when its slot and address are reused",
          "[video_core][image_view_slots]") {
    struct View {
        GPUVAddr gpu_addr;
        u32 handle;
    };
    constexpr GPUVAddr address = 0x1000;
    VideoCommon::ImageViewSlots<View> views;
    const auto saved = views.insert(View{address, 1});
    const auto generation = views.Generation(saved);
    const auto other = views.insert(View{address + 0x1000, 2});
    const auto other_generation = views.Generation(other);
    const auto* const original = views.GetIfCurrent(saved, generation);
    REQUIRE(original != nullptr);
    CHECK(original->handle == 1);

    views.erase(saved);
    CHECK(views.GetIfCurrent(saved, generation) == nullptr);
    CHECK(views.GetIfCurrent(other, other_generation) != nullptr);

    const auto replacement = views.insert(View{address, 3});
    REQUIRE(replacement == saved);
    REQUIRE(views[replacement].gpu_addr == address);
    CHECK(views.GetIfCurrent(saved, generation) == nullptr);
    const auto* const current = views.GetIfCurrent(replacement, views.Generation(replacement));
    REQUIRE(current != nullptr);
    CHECK(current->handle == 3);
}
