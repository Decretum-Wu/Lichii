// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Tegra {
class MemoryManager;
}

namespace VideoCommon {

/// Plan workers (EDEN_NX2_PREP_WORKERS=n, with the preparation lane): threads that follow the
/// method lane's changes to each 3D engine's state on mirrors of their own and plan the draws
/// ahead of the preparation lane. The method lane never waits for them: it writes what it sends
/// the preparation lane into a ring of their own too, and a worker that falls a ring behind drops
/// its mirrors until the next checkpoint. The preparation lane never waits for them either: a
/// plan that is not there when it reaches the draw is not used.
///
/// A worker's mirror lacks what the preparation lane does to its own besides the method lane's
/// records: the registers macros it runs write (Maxwell3DLane::RunMacro) are unknown to the
/// worker until the method lane writes them again, and a plan names them.
struct PlanMirror {
    using Maxwell3D = Tegra::Engines::Maxwell3D;

    Maxwell3D::Regs regs{};
    Maxwell3D::State state{};
    std::array<std::array<u32, 64>, 8> root_tables{};
    u32 root_selected{};
    u32 root_offset{};
    u32 root_loaded{};
    /// The last draw's DrawState words (prep_lane.cpp) and inline index byte count.
    std::vector<u32> draw;
    u32 inline_index_bytes{};
    u32 topology{}; ///< The draw state's (its first word)
    u32 engine_state{};
    /// The engine's GPU memory and whether it runs native (Switch 2) programs, as of the last
    /// checkpoint.
    const Tegra::MemoryManager* memory{};
    bool native{};
    /// Registers a macro run on the preparation lane's mirror wrote since the method lane last
    /// wrote them (few at a time: the macros that read memory write a handful).
    std::vector<u16> unknown;
    bool valid{}; ///< Since the last checkpoint

    void SetUnknown(u32 reg) {
        if (std::find(unknown.begin(), unknown.end(), static_cast<u16>(reg)) == unknown.end()) {
            unknown.push_back(static_cast<u16>(reg));
        }
    }
    /// The method lane wrote [first, first + count): their values are known again.
    void Known(u32 first, u32 count) {
        if (!unknown.empty()) {
            std::erase_if(unknown, [&](u16 reg) { return reg >= first && reg < first + count; });
        }
    }
};

/// What a worker leaves for the preparation lane about one draw.
struct DrawPlan {
    static constexpr size_t MaxUnknown = 8;
    u64 digest{}; ///< EDEN_NX2_PREP_PLAN_VERIFY: the mirror's (MirrorDigest)
    u32 num_unknown{};
    std::array<u16, MaxUnknown> unknown{};

    // The graphics pipeline (the renderer's DrawPlanner), with what it holds while unchanged.
    const void* pipeline{}; ///< The renderer's entry for it; null: none planned
    u64 key_hash{};         ///< EDEN_NX2_PREP_PLAN_VERIFY: the key it looked up (0: none)
    u64 pipeline_generation{};
    u64 shader_generation{};
    u64 map_serial{};
    const Tegra::MemoryManager* memory{};
    bool native{};
    std::array<const void*, 6> shader_infos{};
};

/// The renderer's part of planning: what it can work out for a draw from a worker's mirror
/// alone, with lookups that only read what it publishes for the workers.
class DrawPlanner {
public:
    virtual ~DrawPlanner() = default;
    /// A worker thread, any time: fills in `plan` for the draw `mirror` is at.
    virtual void Plan(const PlanMirror& mirror, DrawPlan& plan) = 0;
};

class PlanWorkers {
public:
    using Maxwell3D = Tegra::Engines::Maxwell3D;

    [[nodiscard]] static PlanWorkers& Get();

    /// EDEN_NX2_PREP_WORKERS, read once (0: none).
    [[nodiscard]] static u32 Configured();
    /// EDEN_NX2_PREP_PLAN_VERIFY=1, read once.
    [[nodiscard]] static bool Verify();

    /// GpuSplit::Start with the preparation lane, Stop.
    void Start(u32 count);
    void Stop();

    /// Any thread: whether workers run (the method lane sends them its changes).
    [[nodiscard]] bool Active() const noexcept {
        return active.load(std::memory_order_relaxed);
    }

    // Method lane.
    /// A Maxwell3DLane record (prep_lane.cpp's tags) for `engine`'s mirror.
    void PushRecord(const void* engine, std::span<const u32> record);
    /// A macro run on the preparation lane's mirror of `engine` writes these registers.
    void PushMacro(const void* engine, std::span<const u32> writes);
    /// The sequence number of the next draw (any engine), sent with it to the preparation lane.
    [[nodiscard]] u64 NextSeq();
    /// Draw `seq` of `engine`: the records sent so far are its state. Publishes.
    void PushDraw(const void* engine, u64 seq);
    /// Whether a checkpoint is due for `engine` (PushCheckpoint).
    [[nodiscard]] bool CheckpointDue(const void* engine);
    /// Any thread: every engine's next record comes with a checkpoint.
    void RequestCheckpoints() noexcept {
        checkpoint_wanted.store(true, std::memory_order_relaxed);
    }
    /// The whole state a worker's mirror of `engine` starts from (`regs` with MacroOwned values
    /// for registers a macro owns), with the draw state words of the last draw.
    void PushCheckpoint(const void* engine, const Maxwell3D::Regs& regs,
                        const Maxwell3D::State& state,
                        const std::array<std::array<u32, 64>, 8>& root_tables, u32 root_selected,
                        u32 root_offset, bool root_loaded, std::span<const u32> draw,
                        u32 inline_index_bytes, u32 engine_state,
                        const Tegra::MemoryManager* memory, bool native,
                        std::span<const u16> unknown);

    // Preparation lane.
    /// The plan for draw `seq`, if a worker finished it; copied into `plan`.
    [[nodiscard]] bool TakePlan(u64 seq, DrawPlan& plan);
    /// Counts a draw: whether its plan was there and, with EDEN_NX2_PREP_PLAN_VERIFY, whether its
    /// mirror digest matched the preparation lane's. Logs every 10 s.
    void NoteDraw(bool found, bool matched);

    /// The plan of the draw the preparation lane runs now (Maxwell3DLane::CheckPlan), or null.
    [[nodiscard]] static const DrawPlan* CurrentPlan() noexcept {
        return t_current_plan;
    }
    static void SetCurrentPlan(const DrawPlan* plan) noexcept {
        t_current_plan = plan;
    }

    enum class PipelineOutcome : u32 {
        Used,     ///< The planned pipeline stood for the lookup
        Stale,    ///< Something it depends on changed since it was planned
        Matched,  ///< EDEN_NX2_PREP_PLAN_VERIFY: the lookup found the planned one
        Differed, ///< EDEN_NX2_PREP_PLAN_VERIFY: the lookup found another one
        SameKeyUnplanned,  ///< EDEN_NX2_PREP_PLAN_VERIFY: none planned, the key was the lookup's
        OtherKeyUnplanned, ///< EDEN_NX2_PREP_PLAN_VERIFY: none planned, another key
        Count,
    };
    /// Preparation lane: what became of a draw's planned pipeline (logged with NoteDraw).
    void NotePipeline(PipelineOutcome outcome) noexcept {
        ++pipeline_outcomes[static_cast<size_t>(outcome)];
    }

    /// Why a worker planned no pipeline (or Found).
    enum class PlannerResult : u32 {
        Found,
        NoMemory,
        UnknownRegisters,
        Mesh,
        Unmapped,
        NotRegistered,
        NotInIndex,
        Count,
    };
    /// Worker threads.
    void NotePlanner(PlannerResult result) noexcept {
        planner_results[static_cast<size_t>(result)].fetch_add(1, std::memory_order_relaxed);
    }

    /// The renderer's planner (null: plans carry only the mirror's digest). Setting null waits
    /// until no worker is inside the old one.
    void SetPlanner(DrawPlanner* planner);

private:
    static constexpr size_t RingWords = size_t{1} << 21; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    /// A worker more than this behind the published position may read words being rewritten.
    static constexpr size_t SafeLag = RingWords / 2;
    static constexpr size_t NumSlots = size_t{1} << 14;

    enum Kind : u32 {
        KindWrap = 1,
        KindRecord,
        KindMacro,
        KindDraw,
        KindCheckpoint,
    };

    struct alignas(64) Slot {
        /// Odd while a worker writes the slot; the sequence it holds is version / 2 - 1.
        std::atomic<u64> version{};
        DrawPlan plan;
    };

    struct Worker {
        u32 index{};
        u64 read_position{};
        std::vector<std::pair<const void*, std::unique_ptr<PlanMirror>>> mirrors;
        std::jthread thread;
    };

    u32* Reserve(size_t words);
    void Publish();
    void WorkerLoop(Worker& worker, std::stop_token stop_token);
    /// Returns the words the entry at `position` takes, or 0 if the worker fell behind.
    size_t RunEntry(Worker& worker, u64 position);
    PlanMirror* FindMirror(Worker& worker, const void* engine);
    void WritePlan(u64 seq, const PlanMirror& mirror);

    std::atomic<bool> active{};
    u32 count{};
    std::unique_ptr<u32[]> ring;
    std::unique_ptr<Slot[]> slots;
    std::vector<std::unique_ptr<Worker>> workers;

    // Method lane side.
    u64 write_position{};
    u64 next_seq{};
    std::vector<std::pair<const void*, u64>> checkpoint_draws; ///< Draws since each one's last

    alignas(64) std::atomic<u64> published{};
    alignas(64) std::atomic<u32> sleepers{};
    std::atomic<u32> wake{};
    /// Set by a worker that dropped its mirrors: the next draw of every engine has a checkpoint.
    alignas(64) std::atomic<bool> checkpoint_wanted{};

    // Statistics (relaxed counters, logged with the preparation lane's verify counts).
    std::atomic<u64> overruns{};
    std::atomic<u64> plans_written{};
    std::atomic<u64> plans_unsynced{};
    u64 window_start{};
    u64 draws_seen{};
    u64 plans_found{};
    u64 plans_matched{};
    u64 plans_differed{};
    std::array<u64, static_cast<size_t>(PipelineOutcome::Count)> pipeline_outcomes{};
    std::array<std::atomic<u64>, static_cast<size_t>(PlannerResult::Count)> planner_results{};

    std::atomic<DrawPlanner*> planner{};
    std::atomic<u32> planning{}; ///< Workers inside the planner
    static inline thread_local const DrawPlan* t_current_plan{};
};

/// A digest of a 3D engine's state as a plan sees it: registers (those in `unknown` and the
/// rasterizer's own logic op enable left out), constant buffer bindings, root tables and the
/// last draw's state words.
[[nodiscard]] u64 MirrorDigest(const Tegra::Engines::Maxwell3D::Regs& regs,
                               const Tegra::Engines::Maxwell3D::State& state,
                               const std::array<std::array<u32, 64>, 8>& root_tables,
                               std::span<const u32> draw, std::span<const u16> unknown);

/// Applies a Maxwell3DLane record to a worker's mirror (prep_lane.cpp).
void ApplyPlanRecord(PlanMirror& mirror, std::span<const u32> record);

} // namespace VideoCommon
