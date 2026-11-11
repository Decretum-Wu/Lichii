// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/nx2_back_stages.h"
#include "video_core/nx2_stage_snapshot.h"

namespace {

namespace DrawStages = VideoCommon::DrawStages;
using DrawStages::BackRing;
using DrawStages::CleanPages;
using DrawStages::StageSnapshot;

constexpr u64 Page = CleanPages::PageSize;
constexpr u64 Base = u64{16} << 22;

class Tracked {
public:
    void UpdatePagesCachedCount(DAddr, size_t, s32) {}
    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>>, s32) {}
};
using MemoryTracker = VideoCommon::MemoryTrackerBase<Tracked>;

/// Guest memory from Base, a byte per address.
struct Guest {
    std::vector<u8> bytes = std::vector<u8>(64 * Page);
    void Read(u64 addr, u8* out, u64 size) const {
        std::memcpy(out, bytes.data() + (addr - Base), size);
    }
    void Fill(u64 addr, u64 size, u8 value) {
        std::memset(bytes.data() + (addr - Base), value, size);
    }
};

std::vector<std::pair<u64, u64>> Unclean(const CleanPages& clean, u64 addr, u64 size) {
    std::vector<std::pair<u64, u64>> runs;
    clean.ForEachUnclean(addr, size, [&](u64 begin, u64 end) { runs.emplace_back(begin, end); });
    return runs;
}

/// Stage B's side of the buffer cache, as BufferCache::SettleSnapshot drives it.
struct Settled {
    std::vector<std::pair<u64, u64>> uploads;
    std::vector<std::pair<u64, u64>> marks;
};
Settled Settle(const StageSnapshot& snapshot, u64 addr, u64 size, CleanPages& clean,
               MemoryTracker& tracker, const Guest& guest) {
    Settled settled;
    std::vector<u8> scratch;
    DrawStages::SettleSnapshot(
        snapshot, addr, size, clean,
        [&](u64 begin, u64 length, auto&& found) {
            tracker.ForEachUploadRange(begin, length, found);
        },
        [&](u64 begin, u64 length) { settled.uploads.emplace_back(begin, length); },
        [&](u64 begin, u8* out, u64 length) { guest.Read(begin, out, length); },
        [&](u64 begin, u64 length) {
            tracker.MarkRegionAsCpuModified(begin, length);
            settled.marks.emplace_back(begin, length);
        },
        scratch);
    return settled;
}

} // Anonymous namespace

TEST_CASE("Stage snapshot: clean pages report the runs a CPU mark cleared", "[stage_snapshot]") {
    CleanPages clean;
    clean.Enable();
    REQUIRE(Unclean(clean, Base + 10, 3 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base, Base + 4 * Page}});
    clean.Set(Base, 200 * Page);
    REQUIRE(Unclean(clean, Base, 200 * Page).empty());
    clean.Clear(Base + 63 * Page + 5, Page);
    clean.Clear(Base + 130 * Page, 1);
    REQUIRE(Unclean(clean, Base, 200 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base + 63 * Page, Base + 65 * Page},
                                             {Base + 130 * Page, Base + 131 * Page}});
    REQUIRE(Unclean(clean, Base + 64 * Page, Page) ==
            std::vector<std::pair<u64, u64>>{{Base + 64 * Page, Base + 65 * Page}});
}

TEST_CASE("Stage snapshot: A copies the pages that aren't clean, merged and found by address",
          "[stage_snapshot]") {
    CleanPages clean;
    clean.Enable();
    Guest guest;
    for (u64 page = 0; page < 64; ++page) {
        guest.Fill(Base + page * Page, Page, static_cast<u8>(page));
    }
    clean.Set(Base, 64 * Page);
    clean.Clear(Base + 2 * Page, 2 * Page);
    clean.Clear(Base + 10 * Page, Page);
    const std::array<StageSnapshot::Range, 4> ranges{{
        {Base + Page, 4 * Page, false},
        {Base + 3 * Page + 8, 100, false},
        {Base + 9 * Page, 3 * Page, false},
        {Base + 20 * Page + 16, 32, true},
    }};
    StageSnapshot snapshot;
    snapshot.Capture(ranges, clean, [&](u64 a, u8* out, u64 size) { guest.Read(a, out, size); },
                     true);
    REQUIRE(snapshot.chunks.size() == 3);
    REQUIRE(snapshot.chunks[0].begin == Base + 2 * Page);
    REQUIRE(snapshot.chunks[0].size == 2 * Page);
    REQUIRE(snapshot.chunks[1].begin == Base + 10 * Page);
    REQUIRE(snapshot.chunks[2].begin == Base + 20 * Page);
    REQUIRE(snapshot.Bytes() == 4 * Page);
    REQUIRE(snapshot.Find(Base + 3 * Page + 8, 100)[0] == 3);
    REQUIRE(snapshot.Find(Base + 20 * Page + 16, 32)[0] == 20);
    REQUIRE(snapshot.Find(Base + Page, 2 * Page) == nullptr);
    REQUIRE(snapshot.Find(Base + 3 * Page, 2 * Page) == nullptr);
    for (const auto& chunk : snapshot.chunks) {
        REQUIRE(snapshot.HashChunk(chunk) == chunk.hash);
    }
    clean.Set(Base, 64 * Page);
    snapshot.Capture(ranges, clean, [&](u64 a, u8* out, u64 size) { guest.Read(a, out, size); },
                     false);
    REQUIRE(snapshot.chunks.size() == 1);
    REQUIRE(snapshot.chunks[0].begin == Base + 20 * Page);
}

TEST_CASE("Stage snapshot: a CPU mark after A's copy survives B's upload",
          "[stage_snapshot]") {
    Tracked tracked;
    MemoryTracker tracker{tracked};
    CleanPages clean;
    clean.Enable();
    tracker.SetCleanPages(&clean);
    Guest guest;
    guest.Fill(Base, 8 * Page, 1);
    tracker.UnmarkRegionAsCpuModified(Base, 8 * Page);
    clean.Set(Base, 8 * Page);

    tracker.MarkRegionAsCpuModified(Base + Page, 2 * Page);
    guest.Fill(Base + Page, 2 * Page, 2);
    REQUIRE(Unclean(clean, Base, 8 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base + Page, Base + 3 * Page}});

    const std::array<StageSnapshot::Range, 1> ranges{{{Base, 8 * Page, false}}};
    StageSnapshot snapshot;
    snapshot.Capture(ranges, clean, [&](u64 a, u8* out, u64 size) { guest.Read(a, out, size); },
                     false);

    // After A: a write on a page A copied (already CPU-modified, so no new mark), and one on a
    // page that was clean at A, which marks it.
    guest.Fill(Base + 2 * Page, 16, 3);
    tracker.MarkRegionAsCpuModified(Base + 5 * Page, Page);
    guest.Fill(Base + 5 * Page, Page, 3);

    const Settled settled = Settle(snapshot, Base, 8 * Page, clean, tracker, guest);
    REQUIRE(settled.uploads == std::vector<std::pair<u64, u64>>{{Base + Page, 2 * Page}});
    REQUIRE(snapshot.Find(Base + 2 * Page, 16)[0] == 2);
    REQUIRE(settled.marks == std::vector<std::pair<u64, u64>>{{Base + 2 * Page, Page}});
    REQUIRE(tracker.IsRegionCpuModified(Base + 5 * Page, Page));
    REQUIRE(tracker.IsRegionCpuModified(Base + 2 * Page, Page));
    REQUIRE_FALSE(tracker.IsRegionCpuModified(Base + Page, Page));
    REQUIRE(Unclean(clean, Base, 8 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base + 2 * Page, Base + 3 * Page},
                                             {Base + 5 * Page, Base + 6 * Page}});
}

TEST_CASE("Stage snapshot: a region B's scan creates with a snapshot pending",
          "[stage_snapshot]") {
    // A range no buffer held before: its tracker region starts CPU-modified when B's buffer
    // creation or upload scan first reaches it, and no page of it was ever clean.
    Tracked tracked;
    MemoryTracker tracker{tracked};
    CleanPages clean;
    clean.Enable();
    tracker.SetCleanPages(&clean);
    Guest guest;
    guest.Fill(Base, 8 * Page, 7);
    const std::array<StageSnapshot::Range, 1> ranges{{{Base + 2 * Page, 3 * Page, false}}};
    StageSnapshot snapshot;
    snapshot.Capture(ranges, clean, [&](u64 a, u8* out, u64 size) { guest.Read(a, out, size); },
                     false);
    REQUIRE(snapshot.Bytes() == 3 * Page);
    guest.Fill(Base + 3 * Page, Page, 9);
    const Settled settled = Settle(snapshot, Base + 2 * Page, 3 * Page, clean, tracker, guest);
    REQUIRE(settled.uploads == std::vector<std::pair<u64, u64>>{{Base + 2 * Page, 3 * Page}});
    REQUIRE(settled.marks == std::vector<std::pair<u64, u64>>{{Base + 3 * Page, Page}});
    REQUIRE(tracker.IsRegionCpuModified(Base, 2 * Page));
    REQUIRE(tracker.IsRegionCpuModified(Base + 3 * Page, Page));
    REQUIRE_FALSE(tracker.IsRegionCpuModified(Base + 2 * Page, Page));
    REQUIRE(Unclean(clean, Base + 2 * Page, 3 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base + 3 * Page, Base + 4 * Page}});
}

TEST_CASE("Stage snapshot: a cached CPU write and its flush clear the pages",
          "[stage_snapshot]") {
    Tracked tracked;
    MemoryTracker tracker{tracked};
    CleanPages clean;
    clean.Enable();
    tracker.SetCleanPages(&clean);
    tracker.UnmarkRegionAsCpuModified(Base, 8 * Page);
    clean.Set(Base, 8 * Page);
    tracker.CachedCpuWrite(Base + Page, 16);
    REQUIRE(Unclean(clean, Base, 8 * Page) ==
            std::vector<std::pair<u64, u64>>{{Base + Page, Base + 2 * Page}});
    clean.Set(Base, 8 * Page);
    tracker.FlushCachedWrites();
    REQUIRE_FALSE(Unclean(clean, Base, 8 * Page).empty());
}

TEST_CASE("Stage snapshot: a slot's copy stays A's until B consumed the packet",
          "[stage_snapshot]") {
    CleanPages clean;
    std::array<StageSnapshot, BackRing::Capacity> slots;
    std::array<u64, BackRing::Capacity> sequence{};
    std::atomic<u64> wrong{};
    std::atomic<u64> consumed{};
    const auto run = [&](u64 first, u64 count) {
        BackRing ring;
        ring.Start(
            [&](u32 slot) {
                const StageSnapshot& snapshot = slots[slot];
                if ((consumed.load() & 7) == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds{30});
                }
                for (const auto& chunk : snapshot.chunks) {
                    const u8* bytes = snapshot.Find(chunk.begin, chunk.size);
                    if (snapshot.HashChunk(chunk) != chunk.hash ||
                        bytes[0] != static_cast<u8>(sequence[slot]) ||
                        bytes[chunk.size - 1] != static_cast<u8>(sequence[slot])) {
                        wrong.fetch_add(1);
                    }
                }
                consumed.fetch_add(1);
            },
            {}, {});
        for (u64 packet = first; packet < first + count; ++packet) {
            const u32 slot = ring.Acquire();
            sequence[slot] = packet;
            const std::array<StageSnapshot::Range, 1> ranges{
                {{Base + (packet % 5) * Page, (1 + packet % 3) * Page + 1, false}}};
            slots[slot].Capture(
                ranges, clean,
                [&](u64, u8* out, u64 size) { std::memset(out, static_cast<u8>(packet), size); },
                true);
            ring.Commit();
            if (packet % 101 == 0) {
                ring.Join();
            }
            if (packet % 333 == 0) {
                ring.Publish();
            }
        }
        ring.Stop();
    };
    run(0, 1000);
    run(1000, 3 * BackRing::Capacity + 7);
    REQUIRE(consumed.load() == 1000 + 3 * BackRing::Capacity + 7);
    REQUIRE(wrong.load() == 0);
}

TEST_CASE("Stage snapshot: A reads clean pages while marks and B change them",
          "[stage_snapshot]") {
    // A mark that happened before A looks (here, handed over through `round`) is always seen.
    CleanPages clean;
    clean.Enable();
    constexpr u64 Rounds = 2000;
    std::atomic<u64> round{};
    std::thread marker([&] {
        for (u64 next = 1; next <= Rounds; ++next) {
            clean.Set(Base, 64 * Page);
            clean.Clear(Base + (next % 64) * Page, Page);
            round.store(next, std::memory_order_release);
            while (round.load(std::memory_order_acquire) == next) {
                std::this_thread::yield();
            }
        }
    });
    u64 processed = 0;
    u64 found = 0;
    while (processed < Rounds) {
        const u64 current = round.load(std::memory_order_acquire);
        if (current == 0) {
            std::this_thread::yield();
            continue;
        }
        const u64 page = Base + (current % 64) * Page;
        bool seen = false;
        clean.ForEachUnclean(Base, 64 * Page, [&](u64 begin, u64 end) {
            seen |= begin <= page && page < end;
        });
        found += seen ? 1 : 0;
        ++processed;
        round.store(0, std::memory_order_release);
    }
    marker.join();
    REQUIRE(processed == Rounds);
    REQUIRE(found == Rounds);
}
