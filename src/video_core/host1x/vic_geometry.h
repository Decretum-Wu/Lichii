// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

/// Vic's pure geometry, apart from the device so tests can run it over hand-made surfaces.
namespace Tegra::Host1x::VicGeometry {

/// Block-linear swizzle of a pitch surface for a block height of one (log2), as Vic writes it.
void SwizzleSurface(std::span<u8> output, u32 out_stride, std::span<const u8> input, u32 in_stride,
                    u32 height) noexcept;

} // namespace Tegra::Host1x::VicGeometry
