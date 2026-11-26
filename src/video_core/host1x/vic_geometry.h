// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/common_types.h"

/// Vic's pure geometry, apart from the device so tests can run it over hand-made surfaces.
namespace Tegra::Host1x::VicGeometry {

/// A slot's rectangles as Vic's guest configs give them, each edge as Blend's add_one leaves it
/// (a value v becomes v + 1, except 0, which stays 0; left and top edges included), and the sizes
/// of the surfaces Blend reads and writes.
struct BlendInput {
    u32 source_left, source_right, source_top, source_bottom;
    u32 dest_left, dest_right, dest_top, dest_bottom;
    u32 target_left, target_right, target_top, target_bottom;
    u32 out_width, out_height;   ///< The output surface's
    u32 slot_width, slot_rows;   ///< The slot surface's
};

/// What Blend draws: the source rectangle, the target's left and right edges, and the right edge
/// the colour-matrix path may go to (it reads at source_left + x and writes at rect_left + x for x
/// from source_left, so it is bounded by both surfaces' rows).
struct BlendRects {
    u32 source_left, source_right, source_top, source_bottom;
    u32 rect_left, rect_right;
    u32 matrix_right;
};

/// The source rectangle as Blend has always intersected it with the slot's destination, the
/// output's target and the output surface, now clamped to the slot surface too, so nothing is
/// read outside it and the part that overlaps is drawn as before. Nothing when it is empty or
/// inverted after that. Within the result, the copy path's min(source width, target width)
/// pixels from rect_left stay within the output row, since rect_left <= source_left.
std::optional<BlendRects> FitBlend(const BlendInput& in) noexcept;

/// Block-linear swizzle of a pitch surface for a block height of one (log2), as Vic writes it.
void SwizzleSurface(std::span<u8> output, u32 out_stride, std::span<const u8> input, u32 in_stride,
                    u32 height) noexcept;

} // namespace Tegra::Host1x::VicGeometry
