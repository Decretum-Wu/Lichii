// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "common/common_types.h"
#include "video_core/nx2_draw_stages.h"

// Stage 4 of the draw packets (docs/nx2/cmdlist.md): stage B on its own thread, behind the test
// switch EDEN_NX2_BACK_STAGES=1. Stage A (the thread that draws) fills packet slots and hands them
// over in batches; B records them in order. Joins wait until B consumed every packet A committed.
namespace VideoCommon::DrawStages {

/// A single producer (stage A) and single consumer (stage B's thread) ring of packet slots. A
/// hands packets over in batches of Batch, or at once at a join or Publish. B spins a few
/// microseconds when it runs dry, then blocks on a futex (WaitOnAddress on Windows); A does the
/// same when the ring is full or it joins.
class BackRing {
public:
    /// At most Capacity packets in flight: the overlap ring keeps the ranges of the last
    /// OverlapRing::Window packets.
    static constexpr u32 Capacity = static_cast<u32>(OverlapRing::Window);
    static constexpr u32 Batch = 32;
    static constexpr u32 MaxMarks = 1024;
    /// How long either side spins before it blocks.
    static constexpr u64 SpinNs = 3'000;

    using Consume = std::function<void(u32 slot)>;
    using MarkReached = std::function<void(u64 value)>;

    BackRing();
    ~BackRing();

    BackRing(const BackRing&) = delete;
    BackRing& operator=(const BackRing&) = delete;

    /// Starts B's thread, which calls `consume` for each packet in order. `mark_reached` gets the
    /// value of each Mark once B consumed the packets committed before it (on either thread, in
    /// order, under a lock).
    void Start(Consume consume, MarkReached mark_reached, std::function<void()> thread_start);

    /// Hands over every committed packet, lets B consume them and stops its thread.
    void Stop();

    [[nodiscard]] bool Running() const noexcept {
        return thread.joinable();
    }

    // Stage A, one thread.

    /// The slot of the next packet; blocks while the ring is full.
    [[nodiscard]] u32 Acquire();

    /// The packet in the slot Acquire returned is complete: hands the batch over once full.
    void Commit() {
        ++committed;
        if (committed - published_local >= Batch) {
            Publish();
        }
    }

    /// Hands every committed packet over now.
    void Publish();

    /// Publishes and waits until B consumed every committed packet. False when nothing was in
    /// flight.
    bool Join();

    /// `value` reaches MarkReached once B consumed every packet committed so far (at once when it
    /// has).
    void Mark(u64 value);

    [[nodiscard]] bool InFlight() const noexcept {
        return consumed.load(std::memory_order_acquire) != committed;
    }

    /// Packets committed: the sequence of the next one (stage A).
    [[nodiscard]] u64 Committed() const noexcept {
        return committed;
    }

    [[nodiscard]] u64 Consumed() const noexcept {
        return consumed.load(std::memory_order_acquire);
    }

    struct Stats {
        std::atomic<u64> batches{};    ///< Hand-overs
        std::atomic<u64> full_waits{}; ///< A blocked on a full ring
        std::atomic<u64> joins{};      ///< Joins that waited
        std::atomic<u64> a_wait_ns{};  ///< A's time waiting (joins and full ring)
        std::atomic<u64> b_sleeps{};   ///< B blocked after spinning
        std::atomic<u64> b_busy_ns{};  ///< B's time consuming packets
        std::atomic<u64> packets{};
    };
    Stats stats;

private:
    struct MarkEntry {
        u64 packets;
        u64 value;
    };

    void Loop(std::stop_token stop_token);
    void WaitConsumed(u64 target);
    void RunMarks();

    Consume consume;
    MarkReached mark_reached;
    std::function<void()> thread_start;

    // Stage A's side.
    u64 committed{};
    u64 published_local{};
    u64 consumed_cache{};

    alignas(64) std::atomic<u64> published{};
    std::atomic<bool> b_waiting{};
    std::atomic<u32> b_wake{};

    alignas(64) std::atomic<u64> consumed{};
    std::atomic<bool> a_waiting{};
    std::atomic<u32> a_wake{};

    alignas(64) std::mutex mark_mutex;
    std::array<MarkEntry, MaxMarks> marks{};
    u32 mark_head{};
    u32 mark_count{};
    std::atomic<u32> marks_pending{};

    std::jthread thread;
};

/// The ring the thread that draws hands its packets to, while stage B runs on its own thread.
inline std::atomic<BackRing*> g_ring{};
/// Called by joins once B is idle: records the commands A captured since its last packet.
inline void (*g_after_join)(void*){};
inline void* g_after_join_argument{};
/// Called on the split's back thread as it exits: the ring stops on the thread that owns it.
inline void (*g_back_exit)(void*){};
inline void* g_back_exit_argument{};

/// On the thread that draws: hands the packets in flight to B and waits until B consumed them (a
/// join at `site`), then records what A captured since. Anything there that uses B's state calls
/// it first.
inline void JoinB(Join site) {
    if (OwnsRing()) {
        JoinNow(site);
    }
}

} // namespace VideoCommon::DrawStages
