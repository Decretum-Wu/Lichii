// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

#include "common/common_types.h"

/// TEST (EGLA-146, never merge): EDEN_NX2_TEST_MICRO=<hex mask> keeps only the back micro-cuts
/// whose bit is set; 0 restores the old behaviour of all five. Unset: all on.
/// 1 locks and gathers, 2 vertex input cache, 4 prepare clock, 8 uniform streak, 10 break skip.
namespace VideoCommon::TestMicro {

inline const u32 g_mask = [] {
    const char* const value = std::getenv("EDEN_NX2_TEST_MICRO");
    return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0x1fU;
}();

[[nodiscard]] inline bool On(u32 bit) {
    return (g_mask & bit) != 0;
}

} // namespace VideoCommon::TestMicro
