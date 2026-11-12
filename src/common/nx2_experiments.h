// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

// Test-only switches for the Ocarina of Time flying-trees bisect (EGLA-154). Each turns one change
// merged after build-165 back to its build-165 behaviour. Never merge.
namespace Common::Nx2Experiments {

inline bool Enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

inline bool Revert1528() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1528");
    return on;
}

inline bool Revert1479() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1479");
    return on;
}

inline bool Revert1462Flush() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1462_FLUSH");
    return on;
}

inline bool Revert1462Spirv() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1462_SPIRV");
    return on;
}

inline bool Revert1462Lower() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1462_LOWER");
    return on;
}

inline bool Revert1523() {
    static const bool on = Enabled("EDEN_NX2_EXP_REVERT_1523");
    return on;
}

} // namespace Common::Nx2Experiments
