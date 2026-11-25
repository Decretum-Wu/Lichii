// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "video_core/texture_cache/types.h"

namespace VideoCommon {

template <typename View>
class ImageViewSlots : private Common::SlotVector<View> {
    using Base = Common::SlotVector<View>;

public:
    using Base::insert;
    using Base::operator[];

    void erase(ImageViewId id) {
        if (id.Value() >= generations.size()) {
            generations.resize(id.Value() + 1);
        }
        ++generations[id.Value()];
        Base::erase(id);
    }

    [[nodiscard]] u32 Generation(ImageViewId id) const noexcept {
        return id.Value() < generations.size() ? generations[id.Value()] : 0;
    }

    [[nodiscard]] View* GetIfCurrent(ImageViewId id, u32 generation) noexcept {
        return Base::contains(id) && Generation(id) == generation ? &(*this)[id] : nullptr;
    }

private:
    std::vector<u32> generations;
};

} // namespace VideoCommon
