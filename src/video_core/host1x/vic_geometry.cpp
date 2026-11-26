// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "video_core/host1x/vic_geometry.h"

namespace Tegra::Host1x::VicGeometry {

void SwizzleSurface(std::span<u8> output, u32 out_stride, std::span<const u8> input, u32 in_stride, u32 height) noexcept {
    //// Taken from https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/hwcontext_nvtegra.c#L949
    /// Can only handle block height == 1.
    u32 const x_mask = 0xFFFFFFD2u, y_mask = 0x2Cu;
    u32 offs_x = 0, offs_y = 0;
    for (size_t y = 0; y < height; y += 2) {
        auto dst_line = output.data() + offs_y * 16;
        auto const src_line = input.data() + u32(y) * (in_stride / 16) * 16;

        auto offs_line = offs_x;
        for (u32 x = 0; x < in_stride; x += 16) {
            std::memcpy(&dst_line[offs_line * 16], &src_line[x], 16);
            std::memcpy(&dst_line[offs_line * 16 + 16], &src_line[x + in_stride], 16);
            offs_line = (offs_line - x_mask) & x_mask;
        }
        offs_y = (offs_y - y_mask) & y_mask;
        offs_x += offs_y ? 0 : out_stride; // Wrap into next tile row
    }
}

} // namespace Tegra::Host1x::VicGeometry
