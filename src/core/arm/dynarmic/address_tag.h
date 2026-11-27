// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {

/// Top Byte Ignore: with TCR_ELx.TBI set, loads and stores ignore bits 63:56 of a user address
/// (bit 55 is 0 there), so guest code may keep a tag in them.
[[nodiscard]] constexpr u64 StripAddressTag(u64 vaddr) {
    constexpr u64 top_byte = 0xffULL << 56;
    return (vaddr >> 55) & 1 ? vaddr : vaddr & ~top_byte;
}

} // namespace Core
