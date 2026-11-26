// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include "common/fs/path_util.h"
#include "common/settings.h"
#include "tests/common/restore_eden_path.h"
#include "tests/common/temp_dir.h"

class ScopedSystemTestContext {
public:
    ScopedSystemTestContext() {
        Common::FS::SetEdenPath(Common::FS::EdenPath::NANDDir, directory.Path());
        Settings::values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_4Gb);
        Settings::values.use_multi_core.SetValue(false);
    }

    ~ScopedSystemTestContext() {
        Tests::RestoreEdenPath(Common::FS::EdenPath::NANDDir, previous_nand);
        Settings::values.current_user = previous_user;
        Settings::values.memory_layout_mode.SetValue(previous_memory);
        Settings::values.use_multi_core.SetValue(previous_multicore);
    }

    ScopedSystemTestContext(const ScopedSystemTestContext&) = delete;
    ScopedSystemTestContext& operator=(const ScopedSystemTestContext&) = delete;

    const std::filesystem::path& Directory() const {
        return directory.Path();
    }

private:
    const std::filesystem::path previous_nand =
        Common::FS::GetEdenPath(Common::FS::EdenPath::NANDDir);
    const s32 previous_user = Settings::values.current_user.GetValue();
    const Settings::MemoryLayout previous_memory = Settings::values.memory_layout_mode.GetValue();
    const bool previous_multicore = Settings::values.use_multi_core.GetValue();
    // A random name made with an exclusive create, retried while taken (Tests::TempDir). A name
    // from steady_clock was taken twice where the clock moves in steps (about 42 ns with macOS's
    // libc++): by contexts made back to back, and by tests processes started together, and the
    // second context failed. Declared last, so it is removed after the destructor has restored
    // the paths.
    const Tests::TempDir directory{"hwacod-system-test"};
};
