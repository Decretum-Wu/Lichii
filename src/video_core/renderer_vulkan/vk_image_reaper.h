// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class ImageReaper {
public:
    explicit ImageReaper(u64 cap_bytes);
    ~ImageReaper();

    void Push(vk::Image image, u64 bytes);
    void Drain();

private:
    struct Entry {
        vk::Image image;
        u64 bytes;
    };

    void Run(std::stop_token stop);

    const u64 cap_bytes;
    std::mutex mutex;
    std::condition_variable_any work;
    std::condition_variable drained;
    std::deque<Entry> queue;
    u64 pending_bytes{};
    u64 peak_bytes{};
    u64 reaped{};
    u64 reaped_ns{};
    u64 refused{};
    u64 drained_on_owner{};
    std::jthread worker;
};

} // namespace Vulkan
