// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/format.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/thread.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/plan_workers.h"

namespace VideoCommon {

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

/// Draws between an engine's checkpoints, so a worker that dropped its mirrors catches up.
constexpr u64 CheckpointEvery = 4096;

void PutEngine(u32* out, const void* engine) {
    const u64 value = reinterpret_cast<u64>(engine);
    out[1] = static_cast<u32>(value);
    out[2] = static_cast<u32>(value >> 32);
}

const void* GetEngine(const u32* in) {
    return reinterpret_cast<const void*>(u64{in[1]} | (u64{in[2]} << 32));
}

} // Anonymous namespace

u64 MirrorDigest(const Tegra::Engines::Maxwell3D::Regs& regs,
                 const Tegra::Engines::Maxwell3D::State& state,
                 const std::array<std::array<u32, 64>, 8>& root_tables, std::span<const u32> draw,
                 std::span<const u16> unknown) {
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    auto copy = std::make_unique<Maxwell3D::Regs>(regs);
    for (const u16 reg : unknown) {
        copy->reg_array[reg] = 0;
    }
    // The rasterizer's own change (UpdateDynamicStates' AMD logic op workaround).
    copy->reg_array[MAXWELL3D_REG_INDEX(logic_op.enable)] = 0;
    u64 hash = Common::CityHash64(reinterpret_cast<const char*>(copy->reg_array.data()),
                                  sizeof(copy->reg_array));
    // Field by field: the bindings' padding is not part of the state.
    for (const auto& stage : state.shader_stages) {
        for (const auto& binding : stage.const_buffers) {
            const std::array<u64, 2> words{binding.address,
                                           u64{binding.size} | (u64{binding.enabled} << 32)};
            hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(words.data()),
                                              sizeof(words), hash);
        }
    }
    hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(root_tables.data()),
                                      sizeof(root_tables), hash);
    return Common::CityHash64WithSeed(reinterpret_cast<const char*>(draw.data()),
                                      draw.size_bytes(), hash);
}

PlanWorkers& PlanWorkers::Get() {
    static PlanWorkers workers;
    return workers;
}

u32 PlanWorkers::Configured() {
    static const u32 configured = [] {
        const char* value = std::getenv("EDEN_NX2_PREP_WORKERS");
        return value != nullptr ? static_cast<u32>(std::clamp(std::atoi(value), 0, 8)) : 0U;
    }();
    return configured;
}

bool PlanWorkers::Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PREP_PLAN_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

void PlanWorkers::Start(u32 count_) {
    if (count_ == 0 || active.load(std::memory_order_relaxed)) {
        return;
    }
    count = count_;
    ring = std::make_unique<u32[]>(RingWords);
    slots = std::make_unique<Slot[]>(NumSlots);
    write_position = 0;
    next_seq = 0;
    next_write_id = 0;
    processed_seq.store(0, std::memory_order_relaxed);
    applied_writes.store(0, std::memory_order_relaxed);
    local_processed = 0;
    local_applied = 0;
    checkpoint_draws.clear();
    published.store(0, std::memory_order_relaxed);
    checkpoint_wanted.store(true, std::memory_order_relaxed);
    workers.clear();
    for (u32 index = 0; index < count; ++index) {
        auto worker = std::make_unique<Worker>();
        worker->index = index;
        Worker* const pointer = worker.get();
        worker->thread =
            std::jthread([this, pointer](std::stop_token stop) { WorkerLoop(*pointer, stop); });
        workers.push_back(std::move(worker));
    }
    active.store(true, std::memory_order_release);
    LOG_INFO(HW_GPU, "Plan workers: {} threads plan draws ahead of the preparation lane{}", count,
             Verify() ? " (EDEN_NX2_PREP_PLAN_VERIFY)" : "");
}

void PlanWorkers::Stop() {
    if (!active.load(std::memory_order_relaxed)) {
        return;
    }
    active.store(false, std::memory_order_relaxed);
    for (auto& worker : workers) {
        worker->thread.request_stop();
    }
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_all();
    workers.clear(); // Joins
    ring.reset();
    slots.reset();
}

u32* PlanWorkers::Reserve(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: the rest of the ring is skipped.
        ring[index] = KindWrap;
        write_position += RingWords - index;
        index = 0;
    }
    write_position += words;
    return &ring[index];
}

void PlanWorkers::Publish() {
    published.store(write_position, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) != 0) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_all();
    }
}

void PlanWorkers::PushRecord(const void* engine, std::span<const u32> record) {
    const size_t words = 3 + record.size();
    u32* const out = Reserve(words);
    out[0] = KindRecord | static_cast<u32>(words << 8);
    PutEngine(out, engine);
    std::memcpy(out + 3, record.data(), record.size_bytes());
}

void PlanWorkers::PushMacro(const void* engine, std::span<const u32> writes) {
    const size_t words = 3 + writes.size();
    u32* const out = Reserve(words);
    out[0] = KindMacro | static_cast<u32>(words << 8);
    PutEngine(out, engine);
    std::memcpy(out + 3, writes.data(), writes.size_bytes());
}

u64 PlanWorkers::NextSeq() {
    return ++next_seq;
}

void PlanWorkers::PushDraw(const void* engine, u64 seq) {
    u32* const out = Reserve(5);
    out[0] = KindDraw | (5U << 8);
    PutEngine(out, engine);
    out[3] = static_cast<u32>(seq);
    out[4] = static_cast<u32>(seq >> 32);
    for (auto& [known, draws] : checkpoint_draws) {
        if (known == engine) {
            ++draws;
            break;
        }
    }
    Publish();
}

bool PlanWorkers::CheckpointDue(const void* engine) {
    if (checkpoint_wanted.exchange(false, std::memory_order_relaxed)) {
        for (auto& entry : checkpoint_draws) {
            entry.second = CheckpointEvery;
        }
    }
    for (auto& [known, draws] : checkpoint_draws) {
        if (known == engine) {
            if (draws < CheckpointEvery) {
                return false;
            }
            draws = 0;
            return true;
        }
    }
    checkpoint_draws.emplace_back(engine, 0);
    return true;
}

void PlanWorkers::PushCheckpoint(const void* engine, const Maxwell3D::Regs& regs,
                                 const Maxwell3D::State& state,
                                 const std::array<std::array<u32, 64>, 8>& root_tables,
                                 u32 root_selected, u32 root_offset, bool root_loaded,
                                 std::span<const u32> draw, u32 inline_index_bytes,
                                 u32 engine_state, const Tegra::MemoryManager* memory,
                                 bool native, std::span<const u16> unknown) {
    constexpr size_t RegWords = Maxwell3D::Regs::NUM_REGS;
    constexpr size_t BindingWords = Maxwell3D::Regs::MaxShaderStage *
                                    Maxwell3D::Regs::MaxConstBuffers * 4;
    constexpr size_t RootWords = 8 * 64;
    const size_t words =
        3 + RegWords + BindingWords + RootWords + 3 + 2 + 4 + 1 + unknown.size() + draw.size();
    u32* const out = Reserve(words);
    out[0] = KindCheckpoint | static_cast<u32>(words << 8);
    PutEngine(out, engine);
    u32* at = out + 3;
    std::memcpy(at, regs.reg_array.data(), RegWords * sizeof(u32));
    at += RegWords;
    for (const auto& stage : state.shader_stages) {
        for (const auto& binding : stage.const_buffers) {
            at[0] = static_cast<u32>(binding.address);
            at[1] = static_cast<u32>(binding.address >> 32);
            at[2] = binding.size;
            at[3] = binding.enabled ? 1U : 0U;
            at += 4;
        }
    }
    std::memcpy(at, root_tables.data(), RootWords * sizeof(u32));
    at += RootWords;
    at[0] = root_selected;
    at[1] = root_offset;
    at[2] = root_loaded ? 1U : 0U;
    at[3] = static_cast<u32>(draw.size());
    at[4] = inline_index_bytes;
    at += 5;
    const u64 memory_bits = reinterpret_cast<u64>(memory);
    at[0] = engine_state;
    at[1] = static_cast<u32>(memory_bits);
    at[2] = static_cast<u32>(memory_bits >> 32);
    at[3] = native ? 1U : 0U;
    at += 4;
    *at++ = static_cast<u32>(unknown.size());
    for (const u16 reg : unknown) {
        *at++ = reg;
    }
    std::memcpy(at, draw.data(), draw.size_bytes());
    Publish();
}

PlanMirror* PlanWorkers::FindMirror(Worker& worker, const void* engine) {
    for (auto& [known, mirror] : worker.mirrors) {
        if (known == engine) {
            return mirror.get();
        }
    }
    worker.mirrors.emplace_back(engine, std::make_unique<PlanMirror>());
    return worker.mirrors.back().second.get();
}

void PlanWorkers::WorkerLoop(Worker& worker, std::stop_token stop_token) {
    Common::SetCurrentThreadName(fmt::format("GPUPlan{}", worker.index).c_str());
    t_worker = &worker;
    worker.read_position = published.load(std::memory_order_acquire);
    while (!stop_token.stop_requested()) {
        u64 available = published.load(std::memory_order_acquire);
        if (available == worker.read_position) {
            for (u32 spins = 0; spins < 4096 && available == worker.read_position; ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == worker.read_position) {
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                if (available == worker.read_position && !stop_token.stop_requested()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                sleepers.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
        }
        while (worker.read_position != available) {
            size_t words = 0;
            if (available - worker.read_position < SafeLag) {
                words = RunEntry(worker, worker.read_position);
            }
            if (words == 0) {
                // Fell behind: what it read may be rewritten. Start again from checkpoints.
                overruns.fetch_add(1, std::memory_order_relaxed);
                for (auto& [engine, mirror] : worker.mirrors) {
                    mirror->valid = false;
                }
                ClearOverlay(worker);
                worker.overlay_gap = true;
                worker.read_position = published.load(std::memory_order_acquire);
                checkpoint_wanted.store(true, std::memory_order_relaxed);
                break;
            }
            worker.read_position += words;
        }
    }
}

size_t PlanWorkers::RunEntry(Worker& worker, u64 position) {
    const size_t index = position & RingMask;
    const u32* const entry = &ring[index];
    const u32 head = entry[0];
    const u32 kind = head & 0xff;
    if (kind == KindWrap) {
        return RingWords - index;
    }
    const size_t words = head >> 8;
    if (words < 3 || index + words > RingWords) {
        return 0;
    }
    // The entry's words were written before the position was published; a worker far enough
    // behind may see them rewritten while it reads (checked after).
    const auto fell_behind = [&] {
        return published.load(std::memory_order_acquire) - position >= SafeLag;
    };
    PlanMirror& mirror = *FindMirror(worker, GetEngine(entry));
    switch (kind) {
    case KindRecord:
        if (mirror.valid) {
            ApplyPlanRecord(mirror, std::span<const u32>{entry + 3, words - 3});
        }
        break;
    case KindMacro:
        if (mirror.valid) {
            for (size_t i = 3; i < words; ++i) {
                if (entry[i] < Maxwell3D::Regs::NUM_REGS) {
                    mirror.SetUnknown(entry[i]);
                }
            }
        }
        break;
    case KindCheckpoint: {
        constexpr size_t RegWords = Maxwell3D::Regs::NUM_REGS;
        const u32* at = entry + 3;
        std::memcpy(mirror.regs.reg_array.data(), at, RegWords * sizeof(u32));
        at += RegWords;
        for (auto& stage : mirror.state.shader_stages) {
            for (auto& binding : stage.const_buffers) {
                binding.address = u64{at[0]} | (u64{at[1]} << 32);
                binding.size = at[2];
                binding.enabled = at[3] != 0;
                at += 4;
            }
        }
        std::memcpy(mirror.root_tables.data(), at, sizeof(mirror.root_tables));
        at += 8 * 64;
        mirror.root_selected = at[0];
        mirror.root_offset = at[1];
        mirror.root_loaded = at[2];
        const u32 draw_words = at[3];
        mirror.inline_index_bytes = at[4];
        at += 5;
        mirror.engine_state = at[0];
        mirror.memory =
            reinterpret_cast<const Tegra::MemoryManager*>(u64{at[1]} | (u64{at[2]} << 32));
        mirror.native = at[3] != 0;
        at += 4;
        mirror.unknown.assign(at + 1, at + 1 + at[0]);
        at += 1 + at[0];
        mirror.draw.assign(at, at + draw_words);
        mirror.topology = draw_words != 0 ? at[0] : 0;
        mirror.valid = !fell_behind();
        break;
    }
    case KindWrite: {
        const u64 id = u64{entry[3]} | (u64{entry[4]} << 32);
        const GPUVAddr address = u64{entry[5]} | (u64{entry[6]} << 32);
        if (worker.overlay_gap) {
            worker.overlay_gap = false;
            worker.overlay_complete_after = id - 1;
        }
        const size_t first = worker.overlay_words.size();
        worker.overlay_words.insert(worker.overlay_words.end(), entry + 7, entry + words);
        const u32 count = static_cast<u32>(words - 7);
        const u64 position = worker.overlay_base + worker.overlay.size();
        worker.overlay.push_back(OverlayWrite{id, GetEngine(entry), address, first, count});
        if (count != 0) {
            const u64 first_block = address >> Worker::OverlayBlockBits;
            const u64 last_block = (address + u64{count} * sizeof(u32) - 1) >> Worker::OverlayBlockBits;
            if (last_block - first_block >= Worker::MaxBlocksPerWrite) {
                worker.overlay_wide.push_back(position);
            } else {
                for (u64 block = first_block; block <= last_block; ++block) {
                    const u32 bucket = static_cast<u32>(block % Worker::OverlayBuckets);
                    auto& positions = worker.overlay_buckets[bucket];
                    if (positions.empty()) {
                        worker.overlay_touched.push_back(bucket);
                        if (worker.overlay_touched.size() > 4 * Worker::OverlayBuckets) {
                            std::ranges::sort(worker.overlay_touched);
                            worker.overlay_touched.erase(
                                std::unique(worker.overlay_touched.begin(),
                                            worker.overlay_touched.end()),
                                worker.overlay_touched.end());
                        }
                    }
                    positions.push_back(position);
                }
            }
        }
        break;
    }
    case KindDraw: {
        const u64 seq = u64{entry[3]} | (u64{entry[4]} << 32);
        if (fell_behind()) {
            return 0;
        }
        if (seq % count != worker.index) {
            break;
        }
        if (!mirror.valid || mirror.unknown.size() > DrawPlan::MaxUnknown) {
            plans_unsynced.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        worker.planning_engine = GetEngine(entry);
        WritePlan(worker, seq, mirror);
        break;
    }
    default:
        return 0;
    }
    return fell_behind() ? 0 : words;
}

void PlanWorkers::WritePlan(Worker& worker, u64 seq, const PlanMirror& mirror) {
    // The slot's last plan must be one the preparation lane is done with.
    if (seq >= processed_seq.load(std::memory_order_acquire) + NumSlots) {
        plans_unsynced.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    RetireOverlay(worker);
    DrawPlan plan;
    for (const u16 reg : mirror.unknown) {
        plan.unknown[plan.num_unknown++] = reg;
    }
    std::sort(plan.unknown.begin(), plan.unknown.begin() + plan.num_unknown);
    if (Verify()) {
        plan.digest = MirrorDigest(mirror.regs, mirror.state, mirror.root_tables, mirror.draw,
                                   std::span<const u16>{plan.unknown.data(), plan.num_unknown});
    }
    // Before any guest memory is read: a write the overlay doesn't hold bumps it after.
    plan.memory_serial = Nx2FastDraw::PlanMemorySerial();
    plan.processed_at_read = processed_seq.load(std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_acquire);
    planning.fetch_add(1, std::memory_order_seq_cst);
    if (DrawPlanner* const current = planner.load(std::memory_order_seq_cst)) {
        current->Plan(mirror, plan);
    }
    planning.fetch_sub(1, std::memory_order_release);
    Slot& slot = slots[seq & (NumSlots - 1)];
    const u64 done = (seq + 1) * 2;
    slot.version.store(done + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.plan = plan;
    slot.version.store(done, std::memory_order_release);
    plans_written.fetch_add(1, std::memory_order_relaxed);
}

const DrawPlan* PlanWorkers::TakePlan(u64 seq) const noexcept {
    const Slot& slot = slots[seq & (NumSlots - 1)];
    return slot.version.load(std::memory_order_acquire) == (seq + 1) * 2 ? &slot.plan : nullptr;
}

void PlanWorkers::PushWrite(const void* engine, u64 id, GPUVAddr address,
                            std::span<const u32> words) {
    const size_t size = 7 + words.size();
    u32* const out = Reserve(size);
    out[0] = KindWrite | static_cast<u32>(size << 8);
    PutEngine(out, engine);
    out[3] = static_cast<u32>(id);
    out[4] = static_cast<u32>(id >> 32);
    out[5] = static_cast<u32>(address);
    out[6] = static_cast<u32>(address >> 32);
    std::memcpy(out + 7, words.data(), words.size_bytes());
}

void PlanWorkers::ClearOverlay(Worker& worker) {
    worker.overlay_base += worker.overlay.size();
    worker.overlay.clear();
    worker.overlay_words.clear();
    for (const u32 bucket : worker.overlay_touched) {
        worker.overlay_buckets[bucket].clear();
    }
    worker.overlay_touched.clear();
    worker.overlay_wide.clear();
}

void PlanWorkers::RetireOverlay(Worker& worker) {
    const u64 applied = applied_writes.load(std::memory_order_acquire);
    size_t done = 0;
    while (done < worker.overlay.size() && worker.overlay[done].id <= applied) {
        ++done;
    }
    if (done == worker.overlay.size()) {
        ClearOverlay(worker);
        return;
    }
    // The buckets keep the retired positions until they are read (ReadMemory skips them).
    worker.overlay.erase(worker.overlay.begin(), worker.overlay.begin() + done);
    worker.overlay_base += done;
    std::erase_if(worker.overlay_wide,
                  [base = worker.overlay_base](u64 position) { return position < base; });
    // The words of retired writes go once they are most of the store.
    const size_t first_live = worker.overlay.front().first;
    if (first_live > (size_t{1} << 20) && first_live * 2 > worker.overlay_words.size()) {
        worker.overlay_words.erase(worker.overlay_words.begin(),
                                   worker.overlay_words.begin() + first_live);
        for (OverlayWrite& write : worker.overlay) {
            write.first -= first_live;
        }
    }
}

bool PlanWorkers::ReadMemory(const PlanMirror& mirror, GPUVAddr address, std::span<u8> out) {
    Worker* const worker = t_worker;
    if (worker == nullptr || mirror.memory == nullptr || out.empty()) {
        return false;
    }
    // Writes the overlay may have missed (it fell a ring behind) must be in memory first.
    const u64 applied = applied_writes.load(std::memory_order_acquire);
    if (worker->overlay_gap || applied < worker->overlay_complete_after) {
        return false;
    }
    if (!mirror.memory->IsFullyMappedRange(address, out.size())) {
        return false;
    }
    mirror.memory->ReadBlockUnsafe(address, out.data(), out.size());
    if (worker->overlay.empty()) {
        return true;
    }
    const GPUVAddr end = address + out.size();
    // The writes that may reach the range, in the order they were sent.
    auto& hits = worker->overlay_hits;
    hits.clear();
    const u64 base = worker->overlay_base;
    const u64 first_block = address >> Worker::OverlayBlockBits;
    const u64 last_block = (end - 1) >> Worker::OverlayBlockBits;
    for (u64 block = first_block; block <= last_block; ++block) {
        auto& bucket = worker->overlay_buckets[block % Worker::OverlayBuckets];
        // Positions are oldest first: drop the retired ones here.
        const auto live = std::ranges::lower_bound(bucket, base);
        bucket.erase(bucket.begin(), live);
        hits.insert(hits.end(), bucket.begin(), bucket.end());
        if (block - first_block >= Worker::OverlayBuckets) {
            break; // Every bucket seen
        }
    }
    for (const u64 position : worker->overlay_wide) {
        if (position >= base) {
            hits.push_back(position);
        }
    }
    std::ranges::sort(hits);
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
    for (const u64 position : hits) {
        const OverlayWrite& write = worker->overlay[position - base];
        if (write.id <= applied || write.engine != worker->planning_engine) {
            continue;
        }
        const GPUVAddr write_end = write.address + u64{write.count} * sizeof(u32);
        if (write.address >= end || write_end <= address) {
            continue;
        }
        const GPUVAddr begin = (std::max)(write.address, address);
        const GPUVAddr finish = (std::min)(write_end, end);
        std::memcpy(out.data() + (begin - address),
                    reinterpret_cast<const u8*>(worker->overlay_words.data() + write.first) +
                        (begin - write.address),
                    finish - begin);
    }
    return true;
}
void PlanWorkers::SetPlanner(DrawPlanner* planner_) {
    planner.store(planner_, std::memory_order_seq_cst);
    if (planner_ == nullptr) {
        while (planning.load(std::memory_order_seq_cst) != 0) {
            std::this_thread::yield();
        }
    }
}

void PlanWorkers::NoteDraw(bool found, bool matched) {
    // Counted only for the log (EDEN_NX2_GPU_PROFILE or EDEN_NX2_PREP_PLAN_VERIFY), and the
    // clock read once every 4096 draws.
    if (!Verify() && !Nx2Profile::Enabled()) {
        return;
    }
    ++draws_seen;
    plans_found += found ? 1 : 0;
    plans_matched += found && matched ? 1 : 0;
    plans_differed += found && !matched ? 1 : 0;
    if ((draws_seen & 4095) != 0) {
        return;
    }
    const u64 now = Nx2Profile::Now();
    if (window_start == 0) {
        window_start = now;
        return;
    }
    if (now - window_start < 10'000'000'000ULL) {
        return;
    }
    const auto outcome = [this](PipelineOutcome which) {
        return pipeline_outcomes[static_cast<size_t>(which)];
    };
    const auto result = [this](PlannerResult which) {
        return planner_results[static_cast<size_t>(which)].exchange(0, std::memory_order_relaxed);
    };
    LOG_INFO(HW_GPU,
             "Plan workers in {:.1f} s: {} draws, plans ready for {:.1f}% ({} written, {} "
             "without a synced mirror, {} overruns) | planned pipelines: {} used, {} stale, {} "
             "matched the lookup, {} differed; unplanned with the lookup's key {}, another key {} "
             "| planner: {} found, no memory {}, unknown registers {}, mesh {}, unmapped {}, "
             "shader not registered {}, not in the index {}{}",
             static_cast<double>(now - window_start) / 1e9, draws_seen,
             draws_seen == 0 ? 0.0
                             : 100.0 * static_cast<double>(plans_found) /
                                   static_cast<double>(draws_seen),
             plans_written.exchange(0, std::memory_order_relaxed),
             plans_unsynced.exchange(0, std::memory_order_relaxed),
             overruns.exchange(0, std::memory_order_relaxed), outcome(PipelineOutcome::Used),
             outcome(PipelineOutcome::Stale), outcome(PipelineOutcome::Matched),
             outcome(PipelineOutcome::Differed), outcome(PipelineOutcome::SameKeyUnplanned),
             outcome(PipelineOutcome::OtherKeyUnplanned), result(PlannerResult::Found),
             result(PlannerResult::NoMemory), result(PlannerResult::UnknownRegisters),
             result(PlannerResult::Mesh), result(PlannerResult::Unmapped),
             result(PlannerResult::NotRegistered), result(PlannerResult::NotInIndex),
             Verify() ? fmt::format(" | verify: {} mirrors matched, {} differed", plans_matched,
                                    plans_differed)
                      : std::string{});
    window_start = now;
    draws_seen = plans_found = plans_matched = plans_differed = 0;
    pipeline_outcomes = {};
    // TEST: what made plans' guest memory stale.
    std::string bumps;
    for (size_t reason = 0; reason < Nx2FastDraw::g_plan_bumps.size(); ++reason) {
        if (const u64 count = Nx2FastDraw::g_plan_bumps[reason].exchange(0); count != 0) {
            bumps += fmt::format(" {} {}", Nx2FastDraw::reason_names[reason], count);
        }
    }
    std::vector<std::pair<u32, u32>> methods;
    for (u32 method = 0; method < Nx2FastDraw::g_plan_bump_methods.size(); ++method) {
        if (const u32 count = Nx2FastDraw::g_plan_bump_methods[method].exchange(0); count != 0) {
            methods.emplace_back(count, method);
        }
    }
    std::ranges::sort(methods, std::greater{});
    std::string top;
    for (size_t i = 0; i < methods.size() && i < 10; ++i) {
        top += fmt::format(" {:#x}:{}", methods[i].second, methods[i].first);
    }
    LOG_INFO(HW_GPU, "Plan memory serial bumps in the window:{} | 3D methods:{}", bumps, top);
}

} // namespace VideoCommon
