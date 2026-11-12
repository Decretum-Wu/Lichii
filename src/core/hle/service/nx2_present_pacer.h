// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

#include "common/game_presets.h"

namespace Service {

/// Which GPU fences a present waits for before it returns.
enum class Nx2PresentWait {
    /// This frame's fence (EDEN_NX2_FRAMES_AHEAD=0).
    Current,
    /// The previous frame's fence, and this frame's too while the GPU is behind
    /// (EDEN_NX2_FRAMES_AHEAD=1).
    Previous,
    /// This frame's fence, or the previous frame's once a GPU channel is suspended.
    CurrentUnlessGpuWaits,
    /// Only the previous frame's fence.
    PreviousOnly,
};

/**
 * Returns how a present waits for the GPU.
 *
 * @param frames_ahead_setting EDEN_NX2_FRAMES_AHEAD: 0, 1, or -1 when unset.
 * @param program_id           The running application's program id.
 */
[[nodiscard]] inline Nx2PresentWait SelectPresentWait(int frames_ahead_setting,
                                                      std::uint64_t program_id) {
    // Donkey Kong Bananza's channels wait on each other every frame, so it always ran a frame
    // ahead, where it rewrote memory the GPU thread hadn't read yet (misread commands, then
    // a freeze or crash; rubberbanding while moving). Title 1 runs ahead without either.
    constexpr std::uint64_t DonkeyKongBananzaProgramId = 0x040074A01BF12000ULL;
    // Ocarina waited ~12 ms a frame on its previous frame's fence here (#161).
    constexpr std::uint64_t OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;
    if (frames_ahead_setting == 0) {
        return Nx2PresentWait::Current;
    }
    if (frames_ahead_setting == 1) {
        return Nx2PresentWait::Previous;
    }
    switch (program_id) {
    case DonkeyKongBananzaProgramId:
        return Nx2PresentWait::Current;
    case OcarinaOfTimeProgramId:
        return Nx2PresentWait::Previous;
    // Mario Kart World's channel is suspended at every present, so waiting for this frame gave
    // up after ~4 ms each frame and the fence passed ~31 ms later (EGLA-146).
    case Settings::MarioKartWorldProgramId:
        return Nx2PresentWait::PreviousOnly;
    default:
        return Nx2PresentWait::CurrentUnlessGpuWaits;
    }
}

/**
 * Returns the interval between paced presents: the vblank the game asked for, lengthened when an
 * explicit FPS limit is below the rate that vblank allows.
 *
 * @param vblank_interval The interval the game's present word asks for, already scaled by the
 *                        speed limit.
 * @param fps_limit       Frames per second to hold, or 0 to leave the game's interval alone.
 * @return The longer of the two intervals.
 */
[[nodiscard]] inline std::chrono::nanoseconds PacedInterval(std::chrono::nanoseconds vblank_interval,
                                                            std::uint32_t fps_limit) {
    if (fps_limit == 0) {
        return vblank_interval;
    }
    const auto limited_interval = std::chrono::nanoseconds{1'000'000'000LL / fps_limit};
    return (std::max)(vblank_interval, limited_interval);
}

class Nx2PresentPacer {
public:
    using Clock = std::chrono::steady_clock;

    struct Step {
        Clock::time_point deadline;
        bool reset;
    };

    /**
     * Returns the deadline for the present arriving at @p now: the previous deadline plus the
     * interval, re-anchored to @p now once the caller has fallen more than one interval behind.
     *
     * @param now      When the present reached the pacer.
     * @param interval Present interval, normally the value from PacedInterval.
     * @return The deadline to hold until, and whether the pacer re-anchored because it was late.
     */
    Step Advance(Clock::time_point now, std::chrono::nanoseconds interval) {
        const auto deadline = last_deadline ? *last_deadline + interval : now;
        const bool reset = !last_deadline || now > deadline + interval;
        last_deadline = reset ? now : deadline;
        return {*last_deadline, reset};
    }

private:
    std::optional<Clock::time_point> last_deadline;
};

} // namespace Service
