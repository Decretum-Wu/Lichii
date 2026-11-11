// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>

#include <fmt/format.h>

#include "common/thread.h"
#include "video_core/nx2_back_stages.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace VideoCommon::DrawStages {
namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

} // namespace

BackRing::BackRing() = default;

BackRing::~BackRing() {
    Stop();
}

void BackRing::Start(Consume consume_, MarkReached mark_reached_,
                     std::function<void()> thread_start_) {
    consume = std::move(consume_);
    mark_reached = std::move(mark_reached_);
    thread_start = std::move(thread_start_);
    thread = std::jthread([this](std::stop_token stop_token) { Loop(stop_token); });
}

void BackRing::Stop() {
    if (!thread.joinable()) {
        return;
    }
    published_local = committed;
    published.store(committed, std::memory_order_seq_cst);
    thread.request_stop();
    b_wake.fetch_add(1, std::memory_order_seq_cst);
    b_wake.notify_one();
    thread.join();
    RunMarks();
}

u32 BackRing::Acquire() {
    if (committed - consumed_cache >= Capacity) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (committed - consumed_cache >= Capacity) {
            stats.full_waits.fetch_add(1, std::memory_order_relaxed);
            Publish();
            WaitConsumed(committed - Capacity + 1);
        }
    }
    return static_cast<u32>(committed % Capacity);
}

void BackRing::Publish() {
    if (published_local == committed) {
        return;
    }
    published_local = committed;
    stats.batches.fetch_add(1, std::memory_order_relaxed);
    published.store(committed, std::memory_order_seq_cst);
    if (b_waiting.load(std::memory_order_seq_cst)) {
        b_wake.fetch_add(1, std::memory_order_seq_cst);
        b_wake.notify_one();
    }
}

bool BackRing::Join() {
    if (consumed.load(std::memory_order_acquire) == committed) {
        return false;
    }
    stats.joins.fetch_add(1, std::memory_order_relaxed);
    Publish();
    WaitConsumed(committed);
    return true;
}

void BackRing::WaitConsumed(u64 target) {
    const u64 start = NowNs();
    for (;;) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            break;
        }
        if (NowNs() - start < SpinNs) {
            Pause();
            continue;
        }
        // B checks a_waiting after each store of consumed, A checks consumed after setting it.
        a_waiting.store(true, std::memory_order_seq_cst);
        const u32 seen = a_wake.load(std::memory_order_seq_cst);
        if (consumed.load(std::memory_order_seq_cst) < target) {
            a_wake.wait(seen, std::memory_order_seq_cst);
        }
        a_waiting.store(false, std::memory_order_relaxed);
    }
    stats.a_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
}

void BackRing::Mark(u64 value) {
    {
        std::scoped_lock lock{mark_mutex};
        if (mark_count == MaxMarks) {
            // B is far behind: the oldest marks are the ones it reaches first anyway.
            mark_head = (mark_head + 1) % MaxMarks;
            --mark_count;
            marks_pending.fetch_sub(1, std::memory_order_seq_cst);
        }
        marks[(mark_head + mark_count) % MaxMarks] = {committed, value};
        ++mark_count;
        marks_pending.fetch_add(1, std::memory_order_seq_cst);
    }
    RunMarks();
}

void BackRing::RunMarks() {
    std::scoped_lock lock{mark_mutex};
    const u64 done = consumed.load(std::memory_order_seq_cst);
    bool reached = false;
    u64 value = 0;
    while (mark_count != 0 && marks[mark_head].packets <= done) {
        value = marks[mark_head].value;
        mark_head = (mark_head + 1) % MaxMarks;
        --mark_count;
        marks_pending.fetch_sub(1, std::memory_order_seq_cst);
        reached = true;
    }
    if (reached && mark_reached) {
        mark_reached(value);
    }
}

void BackRing::Loop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPUStageB");
    t_stage = Stage::B;
    t_stage_b_thread = true;
    if (thread_start) {
        thread_start();
    }
    u64 next = consumed.load(std::memory_order_relaxed);
    for (;;) {
        u64 available = published.load(std::memory_order_acquire);
        if (available == next) {
            const u64 spin_start = NowNs();
            while (available == next && NowNs() - spin_start < SpinNs) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
        }
        if (available == next) {
            if (stop_token.stop_requested()) {
                break;
            }
            // Publish bumps b_wake when it sees b_waiting: read it before the last look at
            // `published`, so a publish in between ends the wait at once.
            b_waiting.store(true, std::memory_order_seq_cst);
            const u32 seen = b_wake.load(std::memory_order_seq_cst);
            available = published.load(std::memory_order_seq_cst);
            if (available == next && !stop_token.stop_requested()) {
                stats.b_sleeps.fetch_add(1, std::memory_order_relaxed);
                b_wake.wait(seen, std::memory_order_seq_cst);
            }
            b_waiting.store(false, std::memory_order_relaxed);
            continue;
        }
        const u64 busy_start = NowNs();
        const u64 first = next;
        while (next != available) {
            consume(static_cast<u32>(next % Capacity));
            ++next;
            consumed.store(next, std::memory_order_seq_cst);
            if (a_waiting.load(std::memory_order_seq_cst)) {
                a_wake.fetch_add(1, std::memory_order_seq_cst);
                a_wake.notify_one();
            }
            if (marks_pending.load(std::memory_order_seq_cst) != 0) {
                RunMarks();
            }
        }
        stats.packets.fetch_add(next - first, std::memory_order_relaxed);
        stats.b_busy_ns.fetch_add(NowNs() - busy_start, std::memory_order_relaxed);
    }
}

void ClaimRing() {
    static std::atomic<u64> epochs{};
    const u64 epoch = epochs.fetch_add(1, std::memory_order_relaxed) + 1;
    t_ring_epoch = epoch;
    g_ring_epoch.store(epoch, std::memory_order_release);
}

void ReleaseRing() {
    g_ring_epoch.store(0, std::memory_order_release);
}

void JoinNow(Join site) {
    BackRing* const ring = g_ring.load(std::memory_order_acquire);
    if (ring == nullptr || !OwnsRing()) {
        return;
    }
    if (ring->InFlight()) {
        Count(site);
        ring->Join();
    }
    if (g_after_join != nullptr) {
        g_after_join(g_after_join_argument);
    }
}

bool PacketsInFlight() {
    BackRing* const ring = g_ring.load(std::memory_order_acquire);
    return ring != nullptr && ring->InFlight();
}

u64 RingConsumed() {
    BackRing* const ring = g_ring.load(std::memory_order_acquire);
    return ring != nullptr ? ring->Consumed() : 0;
}

std::string FormatRing(u64 window_ns) {
    BackRing* const ring = g_ring.load(std::memory_order_acquire);
    if (ring == nullptr || !OwnsRing() || window_ns == 0) {
        return {};
    }
    BackRing::Stats& stats = ring->stats;
    const auto rate = [window_ns](std::atomic<u64>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) * 1e9 /
               static_cast<double>(window_ns);
    };
    const auto ms = [window_ns](std::atomic<u64>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) * 1e3 /
               static_cast<double>(window_ns);
    };
    const double packets = rate(stats.packets);
    const double batches = rate(stats.batches);
    const double full = rate(stats.full_waits);
    const double joins = rate(stats.joins);
    const double a_wait = ms(stats.a_wait_ns);
    const double b_busy = ms(stats.b_busy_ns);
    const double sleeps = rate(stats.b_sleeps);
    return fmt::format(" | stage B thread per second: {:.0f} packets in {:.0f} batches, B busy "
                       "{:.1f} ms, B sleeps {:.0f}; A waited {:.1f} ms ({:.0f} joins, {:.0f} "
                       "full ring)",
                       packets, batches, b_busy, sleeps, a_wait, joins, full);
}

} // namespace VideoCommon::DrawStages
