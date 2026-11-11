// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <optional>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_draw_stages.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::DrawStages {
namespace {

constexpr u64 Golden = 0x9E3779B97F4A7C15ULL;

/// The thread that draws: the packets in flight and the last write's page translation.
struct DrawThread {
    OverlapRing ring;
    u64 packets{}; ///< Packets completed (the ring's clock)
    const Tegra::MemoryManager* translated_memory{};
    u64 translated_serial{};
    u64 translated_page{~u64{0}};
    u64 translated_base{};
};

DrawThread& Thread() {
    static DrawThread thread;
    return thread;
}

std::optional<u64> Translate(DrawThread& thread, const Tegra::MemoryManager& memory,
                             GPUVAddr address) {
    const u64 page = address >> 12;
    const u64 serial = memory.MapSerial();
    if (thread.translated_memory == &memory && thread.translated_serial == serial &&
        thread.translated_page == page) {
        return thread.translated_base + (address & 0xfff);
    }
    const std::optional<DAddr> device = memory.GpuToCpuAddress(address);
    if (!device) {
        return std::nullopt;
    }
    thread.translated_memory = &memory;
    thread.translated_serial = serial;
    thread.translated_page = page;
    thread.translated_base = *device - (address & 0xfff);
    return *device;
}

const char* StageName(Stage stage) {
    switch (stage) {
    case Stage::A:
        return "stage A";
    case Stage::B:
        return "stage B";
    default:
        return "between draws";
    }
}

} // namespace

void Configure() {
    g_checking = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    g_counting = g_checking || Nx2Profile::Enabled();
}

bool BackStagesRequested() {
    static const bool requested = [] {
        const char* const value = std::getenv("EDEN_NX2_BACK_STAGES");
        return value != nullptr && value[0] == '1';
    }();
    return requested;
}

bool BackStagesVerifyRequested() {
    static const bool requested = [] {
        const char* const value = std::getenv("EDEN_NX2_BACK_STAGES_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return requested;
}

void Violation(Stage owner, const char* where) {
    if (g_counters.violations.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "Draw stage owner check: {} belongs to {} but runs in {}", where,
                  StageName(owner), StageName(t_stage));
    }
}

OverlapRing::OverlapRing()
    : index(IndexSize), filter((size_t{1} << FilterBits) / 64), memo(SlotCount) {
    entries.reserve(Capacity);
}

size_t OverlapRing::HashSlot(u64 begin, u64 end) const {
    const u64 hash = (begin * Golden) ^ (end * 0xC2B2AE3D27D4EB4FULL);
    return static_cast<size_t>(hash >> 32) & (IndexSize - 1);
}

void OverlapRing::Mark(u64 begin, u64 end) {
    const u64 first = begin >> 16;
    const u64 last = (end - 1) >> 16;
    if (last - first >= 4096) {
        filter_all = true;
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        const u64 bit = (page * Golden) >> (64 - FilterBits);
        filter[bit >> 6] |= u64{1} << (bit & 63);
    }
}

void OverlapRing::Read(u32 slot, u64 begin, u64 size, u64 packet) {
    if (size == 0 || begin == 0) {
        return;
    }
    const u64 end = begin + size;
    Memo& slot_memo = memo[(std::min)(slot, SlotCount - 1)];
    if (slot_memo.begin == begin && slot_memo.end == end && slot_memo.entry < entries.size()) {
        Entry& entry = entries[slot_memo.entry];
        if (entry.begin == begin && entry.end == end) {
            entry.last = packet;
            return;
        }
    }
    size_t hash = HashSlot(begin, end);
    for (; index[hash] != 0; hash = (hash + 1) & (IndexSize - 1)) {
        const u32 stored = index[hash] - 1;
        Entry& entry = entries[stored];
        if (entry.begin == begin && entry.end == end) {
            entry.last = packet;
            slot_memo = {begin, end, stored};
            return;
        }
    }
    if (entries.size() == Capacity) {
        Compact(packet);
        if (entries.size() == Capacity) {
            ++overflows;
            last_overflow = packet;
            return;
        }
        hash = HashSlot(begin, end);
        while (index[hash] != 0) {
            hash = (hash + 1) & (IndexSize - 1);
        }
    }
    entries.push_back({begin, end, packet});
    index[hash] = static_cast<u32>(entries.size());
    Mark(begin, end);
    slot_memo = {begin, end, static_cast<u32>(entries.size() - 1)};
}

bool OverlapRing::Overlaps(u64 begin, u64 size, u64 packets) {
    if (size != 0 && last_overflow && packets - *last_overflow <= Window) {
        return true;
    }
    if (entries.empty() || size == 0) {
        return false;
    }
    const u64 end = begin + size;
    const u64 first = begin >> 16;
    const u64 last = (end - 1) >> 16;
    if (!filter_all && last - first < 64) {
        bool marked = false;
        for (u64 page = first; page <= last && !marked; ++page) {
            const u64 bit = (page * Golden) >> (64 - FilterBits);
            marked = ((filter[bit >> 6] >> (bit & 63)) & 1) != 0;
        }
        if (!marked) {
            return false;
        }
    }
    return std::ranges::any_of(entries, [&](const Entry& entry) {
        return packets - entry.last <= Window && entry.begin < end && begin < entry.end;
    });
}

void OverlapRing::Compact(u64 packets) {
    std::erase_if(entries,
                  [packets](const Entry& entry) { return packets - entry.last > Window; });
    std::ranges::fill(index, 0U);
    std::ranges::fill(filter, u64{0});
    filter_all = false;
    for (size_t position = 0; position < entries.size(); ++position) {
        const Entry& entry = entries[position];
        size_t hash = HashSlot(entry.begin, entry.end);
        while (index[hash] != 0) {
            hash = (hash + 1) & (IndexSize - 1);
        }
        index[hash] = static_cast<u32>(position + 1);
        Mark(entry.begin, entry.end);
    }
}

size_t OverlapRing::Live(u64 packets) const {
    return static_cast<size_t>(std::ranges::count_if(
        entries, [packets](const Entry& entry) { return packets - entry.last <= Window; }));
}

void NoteReadImpl(u32 slot, u64 begin, u64 size) {
    if (!t_draw_thread) {
        return;
    }
    DrawThread& thread = Thread();
    thread.ring.Read(slot, begin, size, thread.packets);
}

void NoteReadAtImpl(u32 slot, u64 begin, u64 size, u64 packet) {
    Thread().ring.Read(slot, begin, size, packet);
}

void EndPacketImpl() {
    t_stage = Stage::Outside;
    ++Thread().packets;
}

void NoteWriteImpl(Join site, u64 begin, u64 size) {
    const auto index = static_cast<size_t>(site);
    g_counters.checked[index].fetch_add(1, std::memory_order_relaxed);
    if (!t_draw_thread) {
        return;
    }
    if (g_threaded) {
        // Stage A noted the ranges at each packet's sequence: those B has not consumed are
        // live. Only the thread that hands the packets over holds them.
        if (OwnsRing() && PacketsInFlight() &&
            Thread().ring.Overlaps(begin, size, RingConsumed() + OverlapRing::Window)) {
            JoinNow(site);
        }
        return;
    }
    DrawThread& thread = Thread();
    if (thread.ring.Overlaps(begin, size, thread.packets)) {
        Count(site);
    }
}

void NoteGpuWriteImpl(Join site, const Tegra::MemoryManager& memory, GPUVAddr address,
                      u64 size) {
    if (!t_draw_thread || (g_threaded && !OwnsRing())) {
        g_counters.checked[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const std::optional<u64> device = Translate(Thread(), memory, address);
    if (!device) {
        g_counters.checked[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    NoteWriteImpl(site, *device, size);
}

std::string FormatJoins(u64 window_ns) {
    if (!(g_counting || g_threaded) || !t_draw_thread || (g_threaded && !OwnsRing()) ||
        window_ns == 0) {
        return {};
    }
    const auto rate = [window_ns](std::atomic<u64>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) * 1e9 /
               static_cast<double>(window_ns);
    };
    std::string line = " | draw stage joins per second:";
    for (size_t site = 0; site < g_counters.front_waits.size(); ++site) {
        const double waits = rate(g_counters.front_waits[site]);
        if (waits != 0.0) {
            line += fmt::format(" front {} {:.0f},", drain_site_names[site], waits);
        }
    }
    for (size_t site = 0; site < JoinCount; ++site) {
        const double joins = rate(g_counters.joins[site]);
        const double checked = rate(g_counters.checked[site]);
        if (checked != 0.0) {
            line += fmt::format(" {} {:.0f} of {:.0f},", join_names[site], joins, checked);
        } else if (joins != 0.0) {
            line += fmt::format(" {} {:.0f},", join_names[site], joins);
        }
    }
    DrawThread& thread = Thread();
    const u64 clock = g_threaded ? RingConsumed() + OverlapRing::Window : thread.packets;
    line += fmt::format(" packets in flight hold {} ranges ({} overflows) | owner violations {}",
                        thread.ring.Live(clock), thread.ring.overflows,
                        g_counters.violations.load(std::memory_order_relaxed));
    return line + FormatRing(window_ns);
}

} // namespace VideoCommon::DrawStages
