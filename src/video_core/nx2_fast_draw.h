// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings. Buffers are resolved and bound again, except small streamed uniforms whose ranges
/// were not rebound or written: those can keep the previous stream allocation in the same command
/// buffer. This assumes no unsynchronized guest CPU writes unknown to the caches between draws;
/// EDEN_NX2_UBO_KEEP_VERIFY checks that assumption. Kept image descriptors and rescaling words are
/// replayed while the views remain valid; EDEN_NX2_DELTA_DESCRIPTORS_VERIFY checks the replay.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left. A draw of another pipeline (or after a
/// draw that can't be followed) takes it with what that pipeline kept at its own last draw while
/// the memory serial, the map, the command buffer, the render pass and the texture pool registers
/// are the ones that draw left, reading every texture handle again (RevisitDraws).
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    Map,
    /// Around a macro: what it runs breaks for itself (HLE programs write registers directly).
    Macro,
    /// 3D methods and deferred calls that write guest memory only where its readers learn of it
    /// otherwise (barriers, waits, clears, host cache invalidations, fence work, releases).
    HostWork,
    /// Texture downloads into guest memory (BumpPlanMemorySerial only).
    Download,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "memory map",    "macro",     "host work",            "download",
};

inline std::atomic<bool> g_enabled{true};

/// Whether vertex stream strides and enables are dynamic state (extended dynamic state and
/// dynamic vertex input): then no pipeline depends on them and they are draw parameters. Set
/// by the pipeline cache.
inline bool g_vertex_streams_dynamic{};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// On their own cache lines, away from what every thread reads (g_enabled).
alignas(64) inline std::atomic<u64> g_serial{1};
/// Bumped by every break but a 3D register change: register writes change no memory, so draws
/// with only those in between can keep uniforms as delta draws do (MemorySerial).
inline std::atomic<u64> g_memory_serial{1};
/// The reason of the break that bumped g_serial last and, for Register and Method, its 3D method
/// (diagnostics; on the thread that draws, the first break since it read the serial).
inline std::atomic<u32> g_last_break{};
/// As g_memory_serial, for what plan workers' reads of guest memory hold under
/// (PlanMemorySerial): only the breaks BumpsPlanMemory names, and fence releases that download
/// buffers or images and texture downloads (BumpPlanMemorySerial).
alignas(64) inline std::atomic<u64> g_plan_memory_serial{1};

/// The thread that draws reads the serials before each draw (Serial, MemorySerial). Once one of
/// its breaks bumped a serial since it last read it, more change nothing it compares: they skip
/// the atomic add (~1.4 million other engine methods a second in Ocarina of Time's water). Other
/// threads always bump.
inline thread_local bool t_reads_serials{};
inline thread_local bool t_serial_bumped{};
inline thread_local bool t_memory_serial_bumped{};

/// The split's method lane (EDEN_NX2_PREP_LANE, gpu_split.h) runs the 3D engine ahead of the
/// rasterizer: there a break waits until the lane sends the state it changed (TakePendingBreak),
/// so it lands in order before the draws after it.
inline thread_local bool t_defer_breaks{};
inline thread_local bool t_break_pending{};
inline thread_local bool t_memory_break_pending{};
inline thread_local bool t_plan_memory_break_pending{};
inline thread_local u32 t_pending_break{};

/// In a break sent on (TakePendingBreak): a break other than a register change was among them.
inline constexpr u32 MemoryBreakBit = 0x8000'0000U;
/// ...and one other than a fence release (g_plan_memory_serial).
inline constexpr u32 PlanMemoryBreakBit = 0x4000'0000U;

/// Breaks after which guest memory a plan worker read may differ in ways nothing else tells:
/// 3D methods with side effects and inline uploads, deferred calls, GPU tick work (flushes),
/// invalidations, indirect resets and unmaps. Not register changes, constant buffer data (the
/// workers lay it over memory), fence releases (their downloads bump the serial themselves and
/// query results land on pages the query cache marks), other engines (their memory writes reach
/// the buffer cache, which logs them for the plans), macros, host work (buffer downloads are
/// logged, texture and fence downloads bump the serial, query results land on marked pages) or
/// frames.
[[nodiscard]] constexpr bool BumpsPlanMemory(Reason reason) {
    switch (reason) {
    case Reason::Method:
    case Reason::InlineData:
    case Reason::Call:
    case Reason::TickWork:
    case Reason::Invalidation:
    case Reason::IndirectReset:
    case Reason::Map:
        return true;
    default:
        return false;
    }
}

/// TEST: plan memory serial bumps by reason, and by 3D method for Method breaks.
inline std::array<std::atomic<u64>, static_cast<size_t>(Reason::Count)> g_plan_bumps{};
inline std::array<std::atomic<u32>, 0xe00> g_plan_bump_methods{};
inline void CountPlanBump(u32 last) {
    const u32 reason = last & 0xff;
    if (reason < g_plan_bumps.size()) {
        g_plan_bumps[reason].fetch_add(1, std::memory_order_relaxed);
    }
    const u32 method = (last >> 8) & 0x3fffff;
    if (reason == static_cast<u32>(Reason::Method) && method < g_plan_bump_methods.size()) {
        g_plan_bump_methods[method].fetch_add(1, std::memory_order_relaxed);
    }
}

inline void Break(Reason reason, u32 method = 0) {
    if (Enabled()) {
        const u32 last = static_cast<u32>(reason) | (method << 8);
        if (t_defer_breaks) {
            t_break_pending = true;
            t_memory_break_pending |= reason != Reason::Register;
            if (BumpsPlanMemory(reason)) {
                t_plan_memory_break_pending = true;
                CountPlanBump(last);
            }
            t_pending_break = last;
            return;
        }
        if (reason != Reason::Register && !(t_reads_serials && t_memory_serial_bumped)) {
            g_memory_serial.fetch_add(1, std::memory_order_relaxed);
            t_memory_serial_bumped = t_reads_serials;
        }
        if (BumpsPlanMemory(reason)) {
            g_plan_memory_serial.fetch_add(1, std::memory_order_seq_cst);
            CountPlanBump(last);
        }
        if (!(t_reads_serials && t_serial_bumped)) {
            g_serial.fetch_add(1, std::memory_order_relaxed);
            g_last_break.store(last, std::memory_order_relaxed);
            t_serial_bumped = t_reads_serials;
        }
    }
}

/// A break that landed elsewhere (TakePendingBreak), as Break with its reason and method.
inline void BreakAs(u32 last) {
    if ((last & MemoryBreakBit) != 0) {
        g_memory_serial.fetch_add(1, std::memory_order_relaxed);
    }
    if ((last & PlanMemoryBreakBit) != 0) {
        g_plan_memory_serial.fetch_add(1, std::memory_order_seq_cst);
    }
    g_serial.fetch_add(1, std::memory_order_relaxed);
    g_last_break.store(last & ~(MemoryBreakBit | PlanMemoryBreakBit), std::memory_order_relaxed);
}

/// A fence release or texture download that writes guest memory other than query results,
/// before and after it writes.
inline void BumpPlanMemorySerial(Reason counted_as = Reason::Fence) {
    g_plan_memory_serial.fetch_add(1, std::memory_order_seq_cst);
    CountPlanBump(static_cast<u32>(counted_as));
}

[[nodiscard]] inline u64 PlanMemorySerial() {
    return g_plan_memory_serial.load(std::memory_order_seq_cst);
}

/// Method lane: the break waiting since the last call, if any.
[[nodiscard]] inline bool TakePendingBreak(u32& last) {
    if (!t_break_pending) {
        return false;
    }
    t_break_pending = false;
    last = t_pending_break | (t_memory_break_pending ? MemoryBreakBit : 0) |
           (t_plan_memory_break_pending ? PlanMemoryBreakBit : 0);
    t_memory_break_pending = false;
    t_plan_memory_break_pending = false;
    return true;
}

/// The thread that draws: what it compares fast draws with (and, from then on, its next break
/// bumps the serial again).
inline u64 Serial() {
    t_reads_serials = true;
    t_serial_bumped = false;
    return g_serial.load(std::memory_order_relaxed);
}

inline u64 MemorySerial() {
    t_reads_serials = true;
    t_memory_serial_bumped = false;
    return g_memory_serial.load(std::memory_order_relaxed);
}

/// Whether full draws keep uniforms when only 3D register changes came since the previous draw.
/// EDEN_NX2_TEST_FULL_KEEPS=0 (TEST): only delta draws keep them.
[[nodiscard]] inline bool FullDrawKeeps() {
    static const bool keeps = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_FULL_KEEPS");
        return value == nullptr || value[0] != '0';
    }();
    return keeps;
}

/// Whether a draw after only 3D register changes (or another draw kind or vertex attribute
/// types) looks its pipeline up and keeps the previous draw's views, samplers and image
/// descriptors when that pipeline resolves them alike. EDEN_NX2_TEST_RELOOKUP=0 (TEST): such
/// draws resolve in full.
[[nodiscard]] inline bool RelookupDraws() {
    static const bool relookup = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_RELOOKUP");
        return value == nullptr || value[0] != '0';
    }();
    return relookup;
}

/// Whether a draw after texture handle writes or rebinds (or computed handle changes) reads the
/// handles again and resolves only the ones that changed, keeping the rest as a delta draw does.
/// EDEN_NX2_TEST_RETEXTURE=0 (TEST): such draws resolve in full.
[[nodiscard]] inline bool RetextureDraws() {
    static const bool retexture = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_RETEXTURE");
        return value == nullptr || value[0] != '0';
    }();
    return retexture;
}

/// Whether a draw whose pipeline differs from the previous draw's (or follows a draw that can't
/// be followed by delta draws) keeps what that pipeline resolved at its own last recorded draw,
/// while no memory, mapping, texture pool register, command buffer or render pass changed since
/// (the rasterizer's RevisitDraw). EDEN_NX2_TEST_REVISIT=0 (TEST): such draws resolve in full.
[[nodiscard]] inline bool RevisitDraws() {
    static const bool revisit = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_REVISIT");
        return value == nullptr || value[0] != '0';
    }();
    return revisit;
}

/// Whether RetextureDraws covers pipelines with texture handles computed at draw time or read
/// from two words (the views of the ones that changed are resolved instead of every view).
/// EDEN_NX2_TEST_RETEXTURE_COMPUTED=0 (TEST): only pipelines whose handles are all single
/// constant buffer words.
/// TEST: EDEN_NX2_TEST_TOGGLE_FILE=<path> turns it off while that file exists (the drawing thread
/// looks every 1024 draws), so one run compares both.
inline bool g_test_toggled{};
[[nodiscard]] inline bool RetextureComputedDraws() {
    static const bool computed = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_RETEXTURE_COMPUTED");
        return value == nullptr || value[0] != '0';
    }();
    return computed && !g_test_toggled;
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

    /// The GPU address the range starts at.
    [[nodiscard]] GPUVAddr Address() const noexcept {
        return address;
    }

    /// Whether the range maps to one run of device memory in GPU order, starting at LinearBase.
    [[nodiscard]] bool IsLinear() const noexcept {
        return linear;
    }
    [[nodiscard]] DAddr LinearBase() const noexcept {
        return linear_base;
    }

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    bool linear{};
    DAddr linear_base{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the split's back); with its preparation lane
/// the method lane collects them and sends them on with its draws (MergeDelta).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};
    /// Set when the ranges were widened (AppendRange) or reached a buffer of texture handles
    /// between its handles: the draw keeps no uniforms, as when such writes ended draw reuse.
    bool no_uniform_keeps{};

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }

    /// Reaches for only the bytes [base + first, base + second) of each of words, all inside the
    /// range spans was updated with. Device writes are compared word by word when that range maps
    /// linearly, else with the whole range.
    [[nodiscard]] bool ReachesWords(GPUVAddr base, std::span<const std::pair<u32, u32>> words,
                                    const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (!spans.Intersects(device_begin[write], device_end[write])) {
                continue;
            }
            if (!spans.IsLinear()) {
                return true;
            }
            for (const auto& [first, second] : words) {
                const DAddr begin = spans.LinearBase() + (base + first - spans.Address());
                const DAddr end = begin + (second - first);
                if (device_begin[write] < end && begin < device_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline thread_local Delta g_delta;

/// Moves g_delta into `out` and clears it, copying only the ranges in use (Delta is ~0.5 KiB and
/// taken for every draw).
inline void TakeDelta(Delta& out) {
    Delta& delta = g_delta;
    out.binds = delta.binds;
    out.num_writes = delta.num_writes;
    std::copy_n(delta.write_begin.begin(), delta.num_writes, out.write_begin.begin());
    std::copy_n(delta.write_end.begin(), delta.num_writes, out.write_end.begin());
    out.num_device_writes = delta.num_device_writes;
    std::copy_n(delta.device_begin.begin(), delta.num_device_writes, out.device_begin.begin());
    std::copy_n(delta.device_end.begin(), delta.num_device_writes, out.device_end.begin());
    out.no_uniform_keeps = delta.no_uniform_keeps;
    delta.binds = {};
    delta.num_writes = 0;
    delta.num_device_writes = 0;
    delta.no_uniform_keeps = false;
}

/// Adds [begin, end) to a delta's range list: extends a range it touches or overlaps, else
/// appends it. A full list widens the range that grows least instead (true then), which only
/// makes Reaches more conservative.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (Enabled()) {
        Delta& delta = g_delta;
        delta.no_uniform_keeps |=
            AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end);
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (Enabled()) {
        Delta& delta = g_delta;
        delta.no_uniform_keeps |= AppendRange(delta.device_begin, delta.device_end,
                                              delta.num_device_writes, begin, end);
    }
}

/// Inline constant buffer writes left out of the delta because they wrote the words memory
/// held (EDEN_NX2_GPU_PROFILE logs them).
inline std::atomic<u64> g_unchanged_writes{};

/// Whether inline constant buffer data written with the words memory holds is left out of the
/// delta (memory, and so every handle and uniform read from it, stays the same).
/// EDEN_NX2_TEST_CB_SAME=0 (TEST): it is noted as any write.
[[nodiscard]] inline bool SkipsUnchangedWrites() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_TEST_CB_SAME");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

/// Notes data about to be written to [begin, begin + bytes) in GPU memory: inline constant buffer
/// `words`, unless memory holds those words already (on the thread that writes them, after
/// everything sent before), or with `words` null data not known yet (reports). Texture and sampler
/// headers are checked by the caller.
void NoteWrittenData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words, u64 bytes);
/// NoteWrittenData for inline constant buffer `words` the caller is about to write: true when
/// the write changes nothing and can be left out (memory holds the words, and neither the GPU
/// caches nor a download for a guest read may hold others there). Fast draws off: false.
[[nodiscard]] bool WriteData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words,
                             u64 bytes);

/// Adds the changes another thread collected (the method lane's) to this thread's.
inline void MergeDelta(const Delta& other) {
    Delta& delta = g_delta;
    for (size_t stage = 0; stage < delta.binds.size(); ++stage) {
        delta.binds[stage] |= other.binds[stage];
    }
    for (u32 i = 0; i < other.num_writes; ++i) {
        delta.no_uniform_keeps |= AppendRange(delta.write_begin, delta.write_end,
                                              delta.num_writes, other.write_begin[i],
                                              other.write_end[i]);
    }
    for (u32 i = 0; i < other.num_device_writes; ++i) {
        delta.no_uniform_keeps |= AppendRange(delta.device_begin, delta.device_end,
                                              delta.num_device_writes, other.device_begin[i],
                                              other.device_end[i]);
    }
    delta.no_uniform_keeps |= other.no_uniform_keeps;
}

} // namespace VideoCommon::Nx2FastDraw
