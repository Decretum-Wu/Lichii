// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

/// The GPU back thread's work outside draw preparation: locks and passes it skips where a check
/// without the lock shows the locked work would change nothing (ported from jpkto/Lichii
/// 7a5e5357cb and f5edeabdbb). Guest CPU write traps hold the buffer cache's lock across page
/// protection calls, so every acquisition the back makes can meet one and spin.
namespace VideoCommon::BackX {

/// EDEN_NX2_BACKX_VERIFY=1: every skip also runs the locked check it replaces, logs
/// "BackX verify mismatch" where that check would have done something, and then does it.
inline bool g_verify{};

/// Called by VideoCore::CreateGPU, before the caches exist.
inline void ReadSwitch() {
    const char* const value = std::getenv("EDEN_NX2_BACKX_VERIFY");
    g_verify = value != nullptr && value[0] == '1';
    if (g_verify) {
        LOG_INFO(HW_GPU, "BackX verify on");
    }
}

/// Verify: a skip the locked check contradicts (logged 64 times at most).
inline void Mismatch(const char* what, u64 address, u64 size) {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_ERROR(HW_GPU, "BackX verify mismatch: {} at {:#x}+{:#x}", what, address, size);
    }
}

} // namespace VideoCommon::BackX
