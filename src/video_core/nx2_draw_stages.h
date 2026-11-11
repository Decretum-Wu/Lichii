// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"

namespace Tegra {
class MemoryManager;
}

// Stage A and stage B of a draw (docs/nx2/cmdlist.md), still on one thread. The thread that draws
// marks which stage runs; with EDEN_NX2_SPLIT_CHECK each stage's entry points check it. Join
// points are where stage 4's A waits until B consumed every packet A sent: here they only count,
// with EDEN_NX2_GPU_PROFILE (the back's profile line names each site) or EDEN_NX2_SPLIT_CHECK.
namespace VideoCommon::DrawStages {

enum class Stage : u8 {
    Outside, ///< Between draws: methods, clears, compute (either stage's state)
    A,       ///< From the draw's start to its command list splice
    B,       ///< From the splice to the draw's end
};

enum class Join : u32 {
    Dma,            ///< MaxwellDMA launches whose destination overlaps a packet in flight
    DmaImageBuffer, ///< DMA image<->buffer copies (the texture cache calls the buffer cache)
    Inline,         ///< I2M and inline writes that overlap
    CbData,         ///< CB_DATA writes that overlap
    Query,          ///< Query, report and semaphore writes that overlap
    GuestFlushRead, ///< Guest reads of GPU-written memory that ask for a flush
    BufferFlush,    ///< Buffer cache flushes (FlushRegion, any thread)
    Present,
    ASubmit,        ///< Stage A submits or dispatches the scheduler's work
    AListFull,      ///< Stage A's command list filled up
    ARenderPassEnd, ///< Stage A ends the scheduler's render pass
    ADeferredClear, ///< Stage A defers or realizes a clear
    ASchedulerRead, ///< Stage A reads scheduler state (render pass, fragment barrier, refresh)
    ABufferChange,  ///< Stage A creates or deletes a buffer (uniform resolve)
    ABufferUpload,  ///< Stage A uploads buffer data (uniform sync)
    BDynamicState,  ///< Stage B takes the dynamic state from the registers after a submit
    // With stage B on its own thread (EDEN_NX2_BACK_STAGES=1), where the thread that draws waits
    // for B:
    ABufferLock,     ///< It takes the buffer cache's lock (B uses the cache under it)
    ASchedulerUse,   ///< It uses the scheduler outside a packet's command list
    ADescriptorWrap, ///< Its descriptor payload wraps or grows while B may still read it
    Clear,
    Compute,
    QueryCache, ///< Query cache work: reports, counters, conditional rendering
    Fence,      ///< Fences, sync points and synchronous operations
    Flush,      ///< FlushAll, FlushCommands, WaitForIdle
    Frame,      ///< TickFrame
    Copy,       ///< Surface copies, DrawTexture, inline uploads, DMA buffer operations
    Channel,
    Barrier,     ///< Fragment and tiled cache barriers
    IdleHook,    ///< The split's idle hook (deferred submits)
    Diagnostics, ///< Draws a diagnostic reads B's result of (frame dump, indirect trace)
    Framebuffer, ///< The texture cache creates or removes a framebuffer (packets point at them)
    Count,
};

inline constexpr size_t JoinCount = static_cast<size_t>(Join::Count);

inline constexpr std::array<const char*, JoinCount> join_names{
    "DMA",           "DMA image<->buffer", "I2M",
    "CB_DATA",       "query",              "guest flush reads",
    "buffer flushes", "present",           "A submits",
    "A list full",   "A render pass ends", "A deferred clears",
    "A scheduler reads", "A buffer changes", "A buffer uploads",
    "B dynamic state after a submit",
    "A buffer cache lock", "A scheduler use", "A descriptor wrap",
    "clears",        "compute",            "query cache",
    "fences",        "flushes",            "frames",
    "copies",        "channels",           "barriers",
    "idle hook",     "diagnostics",        "framebuffer changes",
};

/// Whether the stage marker and the join counters run, as of Configure.
inline bool g_counting{};
/// Stage B runs on its own thread (EDEN_NX2_BACK_STAGES=1, test only): set while the
/// rasterizer's ring runs. Joins then wait for B, and the owner checks are thread checks.
inline std::atomic<bool> g_threaded{};
/// EDEN_NX2_BACK_STAGES_VERIFY=1 with the ring running.
inline std::atomic<bool> g_threaded_verify{};
/// Owner checks (EDEN_NX2_SPLIT_CHECK), as of Configure.
inline bool g_checking{};
inline thread_local Stage t_stage{};
/// A thread that draws (the overlap ring of the stage counters is its own).
inline thread_local bool t_draw_thread{};
/// Stage 4: nonzero while a packet ring runs. The thread that started it, the one that draws,
/// holds the same value. Any other thread owns no ring, and neither does that thread once the
/// ring stopped, whichever thread stopped it.
inline std::atomic<u64> g_ring_epoch{};
inline thread_local u64 t_ring_epoch{};
/// Stage B's own thread.
inline thread_local bool t_stage_b_thread{};

/// Whether the calling thread hands packets to the running ring. Only it may join B, and only
/// its captured command list is live.
[[nodiscard]] inline bool OwnsRing() {
    const u64 epoch = g_ring_epoch.load(std::memory_order_relaxed);
    return epoch != 0 && t_ring_epoch == epoch;
}
/// The calling thread starts handing packets to a ring.
void ClaimRing();
/// The ring stops (any thread): no thread owns it from here on.
void ReleaseRing();

/// Reads the switches (when the split starts and when the rasterizer is made).
void Configure();

/// EDEN_NX2_BACK_STAGES=1 and EDEN_NX2_BACK_STAGES_VERIFY=1, read once.
[[nodiscard]] bool BackStagesRequested();
[[nodiscard]] bool BackStagesVerifyRequested();

/// The thread that draws enters a stage.
inline void Enter(Stage stage) {
    if (g_counting || g_threaded) {
        t_stage = stage;
        t_draw_thread = true;
    }
}

/// Stage 4: hands the packets in flight to B and waits until B consumed them, counting `site`
/// when it waits, then records what A captured since its last packet while B is idle
/// (nx2_back_stages.cpp; JoinB is the inline check).
void JoinNow(Join site);
/// Stage 4: packets A committed that B has not consumed yet.
[[nodiscard]] bool PacketsInFlight();
/// Stage 4: packets B consumed.
[[nodiscard]] u64 RingConsumed();
/// Stage 4: the ring's part of the GPU profile line (empty while B runs inline).
std::string FormatRing(u64 window_ns);

[[nodiscard]] inline bool In(Stage stage) {
    return g_counting && t_stage == stage;
}

void Violation(Stage owner, const char* where);

/// `where` belongs to `owner`'s state: with EDEN_NX2_SPLIT_CHECK, reports it running in the other
/// stage (between draws either may run).
/// With stage B on its own thread, B's state between draws also needs B idle.
inline void Expect(Stage owner, const char* where) {
    if (!g_checking) {
        return;
    }
    if (t_stage != Stage::Outside ? t_stage != owner
                                  : owner == Stage::B && OwnsRing() && PacketsInFlight()) {
        Violation(owner, where);
    }
}

struct Counters {
    std::array<std::atomic<u64>, JoinCount> joins{};
    std::array<std::atomic<u64>, JoinCount> checked{}; ///< Writes checked for overlap, by site
    std::array<std::atomic<u64>, static_cast<size_t>(DrainSite::Count)> front_waits{};
    std::atomic<u64> violations{};
};
inline Counters g_counters;

/// A join point: in stage 4 the caller waits for stage B. Here it counts.
inline void Count(Join site) {
    if (g_counting || g_threaded) {
        g_counters.joins[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// A join point that only stage A reaches (stage B and between draws own the state anyway).
inline void CountInA(Join site) {
    if (In(Stage::A)) {
        Count(site);
    }
}

/// A join point whose operation then uses both stages' state: B is idle while it holds (with B on
/// its own thread the thread that draws waits for it), so the owner checks treat it as between
/// draws.
class JoinScope {
public:
    explicit JoinScope(Join site, bool when = true) {
        if (!when || !(g_counting || g_threaded)) {
            return;
        }
        if (!g_threaded) {
            Count(site);
        } else if (OwnsRing()) {
            JoinNow(site);
        }
        previous = std::exchange(t_stage, Stage::Outside);
        active = true;
    }
    ~JoinScope() {
        if (active) {
            t_stage = previous;
        }
    }
    JoinScope(const JoinScope&) = delete;
    JoinScope& operator=(const JoinScope&) = delete;

private:
    Stage previous{};
    bool active{};
};

/// The split's front waited for the back at `site`: stage 4's back must have run B as well.
inline void NoteFrontWait(DrainSite site) {
    if (g_counting && site != DrainSite::RingFull) {
        g_counters.front_waits[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// Read slots, so a binding that stays the same costs one compare.
inline constexpr u32 IndexSlot = 0;
inline constexpr u32 UniformSlots = 5 * 18;
inline constexpr u32 IndirectSlot = 1 + 32 + 5 * 64 + 5 * 32 + UniformSlots;
inline constexpr u32 IndirectCountSlot = IndirectSlot + 1;
inline constexpr u32 SlotCount = IndirectCountSlot + 1;
[[nodiscard]] constexpr u32 VertexSlot(u32 index) {
    return 1 + index;
}
[[nodiscard]] constexpr u32 StorageSlot(size_t stage, u32 index) {
    return 33 + static_cast<u32>(stage) * 64 + index;
}
[[nodiscard]] constexpr u32 TexelSlot(size_t stage, u32 index) {
    return 33 + 5 * 64 + static_cast<u32>(stage) * 32 + index;
}
/// Stage 4: the uniform buffers B binds from the cache, by their order in the draw.
[[nodiscard]] constexpr u32 UniformSlot(u32 order) {
    return 33 + 5 * 64 + 5 * 32 + (order < UniformSlots ? order : UniformSlots - 1);
}

/// The device ranges stage B read for the last Window packets, standing in for the packets in
/// flight in stage 4: index buffers, vertex streams, storage and texel buffers. A 64 KiB page
/// filter keeps the write checks cheap. The stage 0 probe's ring (EDEN_NX2_STAGE_PROBE).
class OverlapRing {
public:
    static constexpr u64 Window = 64;
    static constexpr size_t Capacity = 1024;

    OverlapRing();

    /// Packet number `packet` read [begin, begin + size).
    void Read(u32 slot, u64 begin, u64 size, u64 packet);

    /// Whether [begin, begin + size) overlaps a range one of the last Window packets read,
    /// `packets` packets having completed.
    [[nodiscard]] bool Overlaps(u64 begin, u64 size, u64 packets);

    [[nodiscard]] size_t Live(u64 packets) const;

    u64 overflows{}; ///< Reads not kept: more than Capacity ranges in the window

private:
    struct Entry {
        u64 begin;
        u64 end;
        u64 last;
    };
    struct Memo {
        u64 begin{};
        u64 end{};
        u32 entry{};
    };

    static constexpr size_t IndexSize = 2 * Capacity;
    static constexpr u32 FilterBits = 18;

    void Compact(u64 packets);
    void Mark(u64 begin, u64 end);
    [[nodiscard]] size_t HashSlot(u64 begin, u64 end) const;

    std::vector<Entry> entries;
    std::vector<u32> index;  ///< Entry + 1, or 0
    std::vector<u64> filter; ///< One bit per hashed 64 KiB page
    std::vector<Memo> memo;
    bool filter_all{}; ///< A range too large to mark is in the table
    std::optional<u64> last_overflow;
};

void NoteReadImpl(u32 slot, u64 begin, u64 size);
void NoteReadAtImpl(u32 slot, u64 begin, u64 size, u64 packet);
void EndPacketImpl();
void NoteWriteImpl(Join site, u64 begin, u64 size);
void NoteGpuWriteImpl(Join site, const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

/// Stage B read this device range for the current packet (thread that draws).
inline void NoteRead(u32 slot, u64 begin, u64 size) {
    if (g_counting) {
        NoteReadImpl(slot, begin, size);
    }
}

/// Stage 4: stage A notes a range B reads for packet `packet` (the ring's sequence), so writes
/// over it join while the packet is in flight.
inline void NoteReadAt(u32 slot, u64 begin, u64 size, u64 packet) {
    NoteReadAtImpl(slot, begin, size, packet);
}

/// The current draw ends: its packet joins the ones in flight.
inline void EndPacket() {
    if (g_counting) {
        EndPacketImpl();
    }
}

/// A method writes guest memory at this device range: with B on its own thread, a write over a
/// range a packet in flight reads waits for B.
inline void NoteWrite(Join site, u64 begin, u64 size) {
    if (g_counting || g_threaded) {
        NoteWriteImpl(site, begin, size);
    }
}

/// A method writes guest memory at this GPU range.
inline void NoteGpuWrite(Join site, const Tegra::MemoryManager& memory, GPUVAddr address,
                         u64 size) {
    if (g_counting || g_threaded) {
        NoteGpuWriteImpl(site, memory, address, size);
    }
}

/// The bytes a MaxwellDMA launch writes from regs.offset_out (block linear: width x height x
/// depth x bytes per pixel, without the block padding).
template <typename DmaRegs>
[[nodiscard]] u64 DmaDestinationBytes(const DmaRegs& regs) {
    const auto& launch = regs.launch_dma;
    using Layout = typename std::remove_cvref_t<decltype(launch)>::MemoryLayout;
    const u64 pixel = launch.remap_enable != 0
                          ? u64{regs.remap_const.num_dst_components_minus_one + 1U} *
                                (regs.remap_const.component_size_minus_one + 1U)
                          : 1;
    const u64 line = u64{regs.line_length_in} * pixel;
    if (launch.multi_line_enable == 0) {
        return line;
    }
    if (launch.dst_memory_layout == Layout::PITCH) {
        const u64 lines = regs.line_count != 0 ? regs.line_count : 1;
        const s64 pitch = static_cast<s64>(regs.pitch_out);
        return (lines - 1) * static_cast<u64>(pitch < 0 ? -pitch : pitch) + line;
    }
    const auto& dst = regs.dst_params;
    return u64{dst.width} * (dst.height != 0 ? dst.height : 1U) *
           (dst.depth != 0 ? dst.depth : 1U) * pixel;
}

/// A MaxwellDMA launch starts.
template <typename DmaRegs>
void NoteDma(const Tegra::MemoryManager& memory, const DmaRegs& regs) {
    if (g_counting || g_threaded) {
        NoteGpuWrite(Join::Dma, memory, regs.offset_out, DmaDestinationBytes(regs));
    }
}

/// The joins since the last call, per second of `window_ns`, for the GPU profile line of the
/// thread that draws; empty on other threads.
std::string FormatJoins(u64 window_ns);

} // namespace VideoCommon::DrawStages
