// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <utility>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_image_reaper.h"

namespace Vulkan {

ImageReaper::ImageReaper(u64 cap_bytes_)
    : cap_bytes{cap_bytes_}, worker{[this](std::stop_token stop) { Run(stop); }} {}

ImageReaper::~ImageReaper() {
    worker.request_stop();
    worker.join();
    Drain();
    LOG_INFO(Render_Vulkan,
             "Vulkan image reaper: {} images destroyed in {:.3f} ms on worker, {} refused, "
             "{} drained on owner, peak {} KiB, {} bytes pending at shutdown",
             reaped, static_cast<double>(reaped_ns) / 1e6, refused, drained_on_owner,
             peak_bytes >> 10, pending_bytes);
}

void ImageReaper::Push(vk::Image image, u64 bytes) {
    {
        std::scoped_lock lock{mutex};
        if (bytes > cap_bytes - pending_bytes) {
            ++refused;
            return;
        }
        queue.push_back({std::move(image), bytes});
        pending_bytes += bytes;
        peak_bytes = (std::max)(peak_bytes, pending_bytes);
    }
    work.notify_one();
}

void ImageReaper::Drain() {
    std::unique_lock lock{mutex};
    if (pending_bytes == 0) {
        return;
    }
    while (!queue.empty()) {
        Entry entry = std::move(queue.front());
        queue.pop_front();
        lock.unlock();
        entry.image.reset();
        lock.lock();
        pending_bytes -= entry.bytes;
        ++drained_on_owner;
    }
    drained.wait(lock, [this] { return pending_bytes == 0; });
}

void ImageReaper::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("VulkanReaper");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    std::unique_lock lock{mutex};
    while (work.wait(lock, stop, [this] { return !queue.empty(); }) && !stop.stop_requested()) {
        Entry entry = std::move(queue.front());
        queue.pop_front();
        lock.unlock();
        const auto start = std::chrono::steady_clock::now();
        entry.image.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        lock.lock();
        pending_bytes -= entry.bytes;
        ++reaped;
        reaped_ns += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        if (pending_bytes == 0) {
            drained.notify_one();
        }
    }
}

} // namespace Vulkan
