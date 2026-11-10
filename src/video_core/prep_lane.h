// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/rasterizer_interface.h"

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

    /// The engine's work that writes guest memory, or runs on the mirror's state.
    void WriteConstantWords(GPUVAddr address, std::span<const u32> words);
    void WriteConstantBlock(GPUVAddr address, std::span<const u32> words);
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

    // EDEN_NX2_PREP_ORDER_VERIFY: mirrors that differ from their engine after a record.
    bool verify{};
    u64 verify_checks{};
    u64 verify_differences{};
};

} // namespace VideoCommon
