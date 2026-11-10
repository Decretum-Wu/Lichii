// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <bitset>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/plan_workers.h"
#include "video_core/rasterizer_interface.h"

namespace Core {
class System;
}

namespace Tegra {
struct MacroEngine;
}

namespace VideoCommon {

/// EDEN_NX2_PREP_LANE=1, read once: the split's back runs as two lanes (gpu_split.h).
[[nodiscard]] bool PrepLaneEnabled();

/// The 3D engine on a back of two lanes (EDEN_NX2_PREP_LANE). The method lane runs the engine:
/// its registers, macros, draw manager and constant buffer state. The rasterizer and its caches
/// read a mirror of that state instead, which the preparation lane brings up to date, in order,
/// before each piece of work that reads it. This is the engine's rasterizer: each call it makes
/// becomes work for the preparation lane on the method lane, or runs at once on any other thread
/// (the split stopped, or the preparation lane running a call the method lane waits for). Work
/// that writes guest memory goes the same way: the method lane neither writes guest memory nor
/// touches the caches.
class Maxwell3DLane final : public VideoCore::RasterizerInterface {
public:
    using Maxwell3D = Tegra::Engines::Maxwell3D;

    Maxwell3DLane(Maxwell3D& engine, Maxwell3D& mirror);
    ~Maxwell3DLane() override;

    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// Brings the mirror up to date (EngineInterface::lane_flush).
    void Flush();

    /// Method lane: a macro that reads memory (Maxwell3D::FrontRunOnBack) runs on the
    /// preparation lane, on the mirror, in order: its memory reads see what the work before it
    /// wrote without the method lane waiting. The registers in `writes` are the mirror's until the
    /// engine writes them again (their value here is MacroOwned); in a tracking shadow RAM mode
    /// their shadow RAM entries too. Under Replay the macro runs on the engine while the method
    /// lane waits. Returns whether it ran on the mirror.
    bool RunMacro(Core::System& system, Tegra::MacroEngine& macro_engine, u32 start,
                  std::span<const u32> writes, const std::bitset<Maxwell3D::Regs::NUM_REGS>* write_set,
                  std::span<const u32> parameters, std::atomic<bool>* escaped,
                  std::atomic<u32>* escapes);
    /// The engine's register value as the method lane holds it: a register a macro the mirror
    /// ran wrote has the mirror's value (any thread once the preparation lane ran the macro).
    [[nodiscard]] u32 RegisterAfterMacro(u32 method, u32 value) const {
        return value == MacroOwned ? mirror.regs.reg_array[method] : value;
    }
    /// As RegisterAfterMacro, a shadow RAM entry.
    [[nodiscard]] u32 ShadowAfterMacro(u32 method, u32 value) const {
        return mirror_shadows[method].load(std::memory_order_acquire)
                   ? mirror.shadow_state.reg_array[method]
                   : value;
    }
    /// Method lane (Maxwell3D::ProcessShadowRam): the engine tracked a write of `method` into its
    /// shadow RAM, which is the engine's again.
    void ShadowWritten(u32 method) {
        mirror_shadows[method].store(false, std::memory_order_release);
    }
    /// Method lane, or the preparation lane while it waits (Maxwell3D::ProcessShadowRam under
    /// Replay): the shadow RAM entry of `method`, from the mirror if a macro it ran wrote it.
    [[nodiscard]] u32 TakeShadow(u32 method);
    /// With nothing running on either lane: the registers and shadow RAM entries the mirror holds
    /// newer values of (macros it ran) are the engine's again. `drained`: the preparation lane ran
    /// everything the method lane sent (else the split stopped, and only those the macros wrote).
    void Reclaim(bool drained);

    /// The engine's value of a register a macro the mirror ran wrote, until the engine writes it
    /// (no write of the register holds it: not a mode, an aligned address or a word count).
    static constexpr u32 MacroOwned = 0xffffffff;

    /// The engine's work that writes guest memory, or runs on the mirror's state. `note`: the
    /// written words go in the fast draw delta (Nx2FastDraw::NoteWrittenData), there.
    void WriteConstantWords(GPUVAddr address, std::span<const u32> words, bool note);
    void WriteConstantBlock(GPUVAddr address, std::span<const u32> words, bool note);
    void QueryCondition();
    void UploadExec(bool is_linear);
    void UploadData(u32 data, bool is_last_call);
    void UploadData(std::span<const u32> data);

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                   u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force) override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void FlushRegionForRead(DAddr addr, u64 size) override;
    void FinishRead(DAddr addr, u64 size) override;
    bool NoteReportPending(DAddr addr, u64 size) override;
    void FinishReport(DAddr addr, u64 size) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out, bool past_sync,
                                             u64* pending_tick) override;
    bool IsGpuTickDone(u64 tick) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    [[nodiscard]] bool Nx2ListSlotsSupported() const override;
    [[nodiscard]] std::optional<u32> Nx2AcquireListSlot() override;
    void Nx2ReleaseListSlot(u32 slot) override;
    void Nx2SnapshotListWords(DAddr device_addr, u32 count, u32 slot) override;
    void Nx2NoteOwedListWord(DAddr device_addr, u32 value) override;
    void Nx2NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) override;
    void Nx2ReadListSlots(std::span<const u32> slots, std::span<u32> words) override;
    void Nx2CheckListSlot(u32 slot, std::span<const u32> words) override;
    void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    bool MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    [[nodiscard]] bool HasPendingQueries(DAddr addr, u64 size) override;
    [[nodiscard]] bool HoldsQueries(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(PAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(PAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    [[nodiscard]] bool AccelerateSurfaceCopy(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Config& copy_config) override;
    [[nodiscard]] bool AccelerateTemporalUpscale(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Surface& motion_vectors,
        const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
        std::array<f32, 2> motion_scale, bool use_history) override;
    [[nodiscard]] bool SupportsGameDlssNetwork() const override;
    [[nodiscard]] bool UsesNvidiaDlss() const override;
    [[nodiscard]] bool CanProbeGameDlssOutput() const override;
    [[nodiscard]] std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    [[nodiscard]] VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    [[nodiscard]] Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;
    void RegisterTransformFeedback(GPUVAddr tfb_object_addr) override;
    bool HasDrawTransformFeedback() override;

private:
    /// What a piece of work reads of the mirror: its state, and the draw manager's for draws.
    enum Reads : u32 {
        ReadsNothing = 0,
        ReadsState = 1,
        ReadsDraw = 3,
    };

    /// Runs `func()` after bringing what it reads of the mirror up to date: on the preparation
    /// lane from the method lane, at once elsewhere. `func` must be trivially copyable.
    template <typename Func>
    void Send(Reads reads, const Func& func);
    /// As Send, `func(data)` with a copy of `data`.
    template <typename Func>
    void SendWords(Reads reads, std::span<const u32> data, const Func& func);
    /// As Send, but the method lane waits for it; `func` may refer to the caller's stack.
    template <typename Func>
    void Wait(const Func& func);

    /// Sends (method lane) or applies (anywhere else) the changes of the engine's state since
    /// the last time.
    void SyncMirror(Reads reads);
    /// Fills `record` with the changes since the last record.
    void Build(Reads reads);
    /// Brings the mirror up to `record` (preparation lane, or the thread that built it).
    void Apply(std::span<const u32> record);
    /// Preparation lane: the plan a worker made for draw `seq` (PlanWorkers), if any, counted;
    /// with EDEN_NX2_PREP_PLAN_VERIFY its mirror is checked against this one.
    const DrawPlan* CheckPlan(u64 seq);

    Maxwell3D& engine;
    Maxwell3D& mirror;
    VideoCore::RasterizerInterface* rasterizer{};

    // What the mirror holds, as of the last record built.
    std::unique_ptr<Maxwell3D::Regs> sent_regs;
    Maxwell3D::State sent_state{};
    std::array<std::array<u32, 64>, 8> sent_root_tables{};
    u32 sent_root_selected{};
    u32 sent_root_offset{};
    bool sent_root_loaded{};
    Maxwell3D::EngineHint sent_engine_state{};
    ::Common::unordered_map<u64, Maxwell3D::HLEReplacementAttributeType> sent_replace_table;
    std::vector<u32> record;
    /// Records Build made since it last compared every register block, and blocks that it found
    /// changed then without a noted write.
    static constexpr u32 FullScanEvery = 64;
    u32 builds_since_full_scan{};
    u64 missed_writes{};
    /// Registers whose shadow RAM entry a macro the mirror ran wrote (RunMacro), read by the front
    /// once the preparation lane ran it.
    std::unique_ptr<std::atomic<bool>[]> mirror_shadows;

    // EDEN_NX2_PREP_ORDER_VERIFY: mirrors that differ from their engine after a record.
    bool verify{};
    u64 verify_checks{};
    u64 verify_differences{};
    /// Registers a macro run on the mirror wrote (RunMacro) and the engine hasn't written since:
    /// the plan workers don't know their values.
    std::bitset<Maxwell3D::Regs::NUM_REGS> macro_owned;
    u32 macro_owned_count{};
};

} // namespace VideoCommon
