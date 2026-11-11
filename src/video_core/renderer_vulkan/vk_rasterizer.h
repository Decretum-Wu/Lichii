// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string_view>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "common/dkb_cpu_trace.h"
#include "common/dkb_frame_trace.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2_back_stages.h"
#include "video_core/oot_network.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core {
class System;
}

namespace Core::Frontend {
class EmuWindow;
}

namespace Tegra {

namespace Engines {
class Maxwell3D;
}

} // namespace Tegra

namespace Vulkan {

struct FramebufferTextureInfo;

class StateTracker;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache,
                           Scheduler& scheduler);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;

    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;

    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;

    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& src,
                            const Tegra::DMA::ImageOperand& dst);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    Scheduler& scheduler;
};

class RasterizerVulkan final : public VideoCore::RasterizerInterface,
                               protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                              Tegra::MaxwellDeviceMemoryManager& device_memory_,
                              const Device& device_, MemoryAllocator& memory_allocator_,
                              StateTracker& state_tracker_, Scheduler& scheduler_);
    ~RasterizerVulkan() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void FlushRegionForRead(DAddr addr, u64 size) override;
    void FinishRead(DAddr addr, u64 size) override;
    bool NoteReportPending(DAddr addr, u64 size) override;
    void FinishReport(DAddr addr, u64 size) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out, bool past_sync,
                                             u64* pending_tick = nullptr) override;
    bool IsGpuTickDone(u64 tick) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    bool HasPendingQueries(DAddr addr, u64 size) override;
    bool HoldsQueries(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    bool HasDrawTransformFeedback() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    bool SupportsGameDlssNetwork() const override;
    bool UsesNvidiaDlss() const override;
    bool CanProbeGameDlssOutput() const override {
        return true;
    }
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth,
                                   std::array<f32, 2> jitter,
                                   std::array<f32, 2> motion_scale, bool use_history,
                                   std::optional<std::array<u32, 2>> render_region) override;
    VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void InitializeChannel(Tegra::Control::ChannelState& channel) override;

    void BindChannel(Tegra::Control::ChannelState& channel) override;

    void ReleaseChannel(s32 channel_id) override;
    std::optional<FramebufferTextureInfo> AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                            VAddr framebuffer_addr,
                                                            u32 pixel_stride);

private:
    static constexpr const u64 NEEDS_D24[] = {
        0x01006A800016E000ULL, // SSBU
        0x0100E95004038000ULL, // XC2
        0x0100A6301214E000ULL, // FE:Engage
    };
    static constexpr size_t MAX_TEXTURES = 192;
    static constexpr size_t MAX_IMAGES = 48;
    static constexpr size_t MAX_IMAGE_VIEWS = MAX_TEXTURES + MAX_IMAGES;

    static constexpr VkDeviceSize DEFAULT_BUFFER_SIZE = 4 * sizeof(float);

    /// Stage A builds draw_packet from the registers, stage B records it (docs/nx2/cmdlist.md).
    /// `draw_func` records the draw command from the packet.
    template <typename Func>
    void PrepareDraw(bool is_indexed, u32 instance_count, Func&& draw_func,
                     bool is_indirect = false);

    void CaptureIndirectTrace();
    void CollectIndirectTrace();
    void TraceDrawFailure(std::string_view reason);
    /// Remembers render target 0 of a draw with one of Ocarina of Time's tonemap shaders.
    void TrackOotTonemappedImage(const GraphicsPipeline& pipeline);
    bool TryNx2IndirectReset(bool is_indexed, u32 instance_count);

    void FlushWork();

    void TraceRenderState(bool is_indexed, const char* outcome, const Framebuffer* framebuffer);

    /// Records the copies of the command list pages the work just recorded wrote and submits, so
    /// the GPU makes them now (BufferCache::Nx2CopyListPages).
    void SubmitNx2ListCopies();

    /// Stage A: takes the dynamic state dirty flags of the draw being prepared with `pipeline`
    /// (null for DrawTexture's blit) and reads the values of those groups into `out`.
    void UpdateDynamicStates(const GraphicsPipeline* pipeline, DynamicStatePacket& out);

    /// Stage B: records the groups `state` carries.
    void RecordDynamicStates(const DynamicStatePacket& state);

    void HandleTransformFeedback(bool enabled);

    /// EDEN_NX2_DRAW_PACKET_VERIFY: compares the packet with the draw resolved from the registers
    /// again.
    void VerifyDrawPacket(const DrawPacket& packet);

    /// The draw commands of Draw and DrawIndirect, from the packet (stage B).
    void RecordDirectDraw(const DrawPacket& packet);
    void RecordIndirectDraw(const DrawPacket& packet);

    /// Stage 4 (EDEN_NX2_BACK_STAGES=1): stage B on its own thread. A fills the slot it holds
    /// with the draw's packet and the commands it captures, and hands it over at the draw's end.
    struct BackSlot {
        DrawPacket packet;
        Scheduler::PacketList list;
        bool indirect{};
        bool native_turing{}; ///< A native macro launched the draw
        u64 committed{};      ///< The packet's position in the ring
        u64 map_serial{};     ///< The channel's address space as of stage A
        /// EDEN_NX2_BACK_STAGES_VERIFY: the device ranges B reads, hashed at A.
        struct Read {
            u64 begin;
            u64 size;
            u64 hash;
            u32 slot;
            bool fresh;
            bool streamed;
        };
        std::vector<Read> reads;
        std::chrono::steady_clock::time_point committed_at{};
    };
    void StartBackStages();
    void StopBackStages();
    /// Stage A: notes what B reads for the slot, hands it to B and takes the next.
    void CommitBackSlot(bool indirect);
    /// After a join, with B idle: records the commands A captured since its last packet.
    void SpliceCapturedCommands();
    /// Stage B's thread.
    void ConsumeBackSlot(u32 index);
    void RecordBackDraw(BackSlot& slot);
    /// Stage A: the ranges B reads for the packet, so writes over them join while it is in
    /// flight (and their hashes for EDEN_NX2_BACK_STAGES_VERIFY).
    void NoteBackReads(BackSlot& slot);
    /// Stage B, EDEN_NX2_BACK_STAGES_VERIFY: the packet's order and the pages it reads.
    void VerifyBackReads(const BackSlot& slot);
    /// Stage A, B on its own thread: the repacked vertex buffers as of the draw being prepared.
    [[nodiscard]] u32 RepackedVertexBuffers() const {
        return back_ring ? back_repack_mask : buffer_cache_runtime.RepackedVertexBuffers();
    }

    /// Stage A's view of the scheduler state B records with (cmdlist-plan ruling 4): what the
    /// packets A handed over bound, until A asks for a change in its lists or uses the scheduler.
    struct BackView {
        const GraphicsPipeline* pipeline{};
        const Framebuffer* framebuffer{};
        bool needs_state_enable_refresh{};
        u64 render_pass{}; ///< Changes where a packet draws into another target
        u64 changes{};     ///< Scheduler::AViewChanges at the last packet

        /// Scheduler::RefreshesStateEnable as B will see it.
        [[nodiscard]] bool RefreshesStateEnable(const GraphicsPipeline* next) const;
        /// A packet drawing with `next` into `target` was handed over.
        void Bind(const GraphicsPipeline* next, const Framebuffer* target);
    };
    [[nodiscard]] u64 RenderPassSerialOfA() const {
        return back_ring ? back_view.render_pass + Scheduler::AViewChanges()
                         : scheduler.RenderPassSerial();
    }
    [[nodiscard]] bool RenderPassActiveOfA() const {
        return back_ring ? back_view.framebuffer != nullptr &&
                               back_view.changes == Scheduler::AViewChanges()
                         : scheduler.IsRenderPassActive();
    }
    [[nodiscard]] const GraphicsPipeline* BoundPipelineOfA() const {
        if (!back_ring) {
            return scheduler.BoundGraphicsPipeline();
        }
        return back_view.changes == Scheduler::AViewChanges() ? back_view.pipeline : nullptr;
    }

    /// Clears a 3D color target whose slices the framebuffer can't reach. Returns true when handled.
    bool ClearVolumeSlices(const Framebuffer* framebuffer, u32 first_slice, u32 slice_count);
    void FrameDumpPass(const Framebuffer* framebuffer);
    /// EDEN_NX2_PENDING_WRITE_TRACE: logs bound buffer ranges with collected, unapplied CPU writes.
    void TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes);
    bool OcarinaTemporalEnabled() const;
    void TrackOcarinaCamera();
    /// Offsets the viewport of Ocarina's scene draws (depth-tested, into the scene depth) by a
    /// sub-pixel amount that changes every frame while FSR replaces the game's upscale, so FSR
    /// gets new sample positions each frame (the game renders without jitter). The offset goes to
    /// FSR with the frame.
    void UpdateOcarinaJitter();
    void TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect);
    bool TryOcarinaUpscale(const ComputePipeline& pipeline, bool replace);
    void ReadOcarinaActor(struct OcarinaActorMesh& mesh);
    /// Records the uniform and storage buffers of the current draw or dispatch for the frame dump.
    bool FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;

    const Device& device;
    MemoryAllocator& memory_allocator;
    StateTracker& state_tracker;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    DescriptorPool descriptor_pool;
    GuestDescriptorQueue guest_descriptor_queue;
    ComputePassDescriptorQueue compute_pass_descriptor_queue;
    /// With EDEN_NX2_BACK_STAGES=1 (stage B on its own thread), B's own staging pool, the
    /// descriptor queue of the buffer cache's passes and the queue B fills A's entries through.
    std::unique_ptr<StagingBufferPool> back_staging_pool;
    std::unique_ptr<ComputePassDescriptorQueue> back_compute_pass_descriptor_queue;
    std::unique_ptr<GuestDescriptorQueue> back_descriptor_queue;
    DescriptorBufferRing descriptor_buffer_ring;
    BlitImageHelper blit_image;
    RenderPassCache render_pass_cache;

    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    QueryCacheRuntime query_cache_runtime;
    QueryCache query_cache;
    PipelineCache pipeline_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;
    FrameDump frame_dump;

    boost::container::static_vector<u32, MAX_IMAGE_VIEWS> image_view_indices;
    std::array<VideoCommon::ImageViewId, MAX_IMAGE_VIEWS> image_view_ids;
    boost::container::static_vector<VkSampler, MAX_TEXTURES> sampler_handles;

    u32 draw_counter = 0;
    std::optional<u64> oot_camera_sample_frame;
    mutable std::optional<VideoCommon::OcarinaUpscaleBackend> oot_backend;
    struct OcarinaMotionTrace {
        const Tegra::MemoryManager* camera_channel{};
        const Tegra::MemoryManager* depth_channel{};
        u64 depth_frame{};
        GPUVAddr depth_address{};
        u32 depth_tic{};
        VkExtent2D extent{};
        std::array<f32, 16> current{};
        std::array<f32, 16> previous{};
    } oot_motion;
    std::unique_ptr<class OcarinaUpscaler> oot_live_dlss;
    bool oot_live_failed{};
    const char* oot_fallback_reason{};
    u32 oot_fallback_repeats{};
    std::optional<u64> oot_dlss_frame;
    std::array<f32, 2> oot_jitter_applied{}; ///< in the viewport state now (render pixels)
    std::array<f32, 2> oot_jitter_offset{};  ///< what oot_jitter_frame's scene draws used
    std::optional<u64> oot_jitter_frame;
    u32 oot_jitter_phases{8}; ///< Halton sequence length for the last FSR frame's scale
    std::optional<u64> oot_4k_scene_frame;
    VkImage oot_4k_guest_present{};
    std::optional<u64> oot_effects_frame;
    GPUVAddr oot_effects_address{};
    VkImageView oot_effects_scene{};
    struct OcarinaActorPose {
        u64 frame{};
        const Tegra::MemoryManager* channel{};
        GPUVAddr vertex{};
        GPUVAddr indices{};
        s32 base_vertex{};
        std::array<f32, 4> origin{};
        std::array<f32, 280 * 12> bones{};
    } oot_actor, oot_previous_actor;
    u64 render_trace_frame = 0;
    u32 render_trace_draws = 0;
    u32 render_trace_clears = 0;
    bool pending_native_work{};
    bool mesh_draw{}; ///< the draw being prepared uses a task/mesh pipeline
    /// The draw the back thread is preparing, reused for its lists' capacity.
    DrawPacket draw_packet;
    u64 draw_sequence{};
    std::unique_ptr<VideoCommon::DrawStages::BackRing> back_ring;
    std::unique_ptr<BackSlot[]> back_slots;
    BackSlot* back_slot{}; ///< The slot stage A fills
    std::thread::id back_owner{};
    bool back_splicing{};
    u64 back_resets_seen{};
    u32 back_repack_mask{};
    /// Stage A: the device range B reads for each read slot as of the latest packet.
    std::array<std::pair<u64, u64>, VideoCommon::DrawStages::SlotCount> back_held{};
    std::vector<u32> back_active; ///< Read slots in back_held with a range
    std::array<bool, VideoCommon::DrawStages::SlotCount> back_read_active{};
    BackView back_view;
    /// Stage B: the latest value of each dynamic state group A took.
    DynamicStatePacket back_dynamic;
    u64 back_b_resets{}; ///< Stage B: its own submits it recorded the dynamic state again after
    /// Stage B could not record a draw a native macro launched: A sets the macro engine's flag.
    std::atomic<bool> back_native_failed{};
    /// Stage B: EDEN_NX2_BACK_STAGES_VERIFY results and the ring's 10 s report.
    struct {
        u64 next{};
        u64 packets{};
        u64 ranges{};
        u64 differing{};
        u64 out_of_order{};
        u64 remapped{};  ///< Packets recorded after the address space changed
        u64 b_submits{}; ///< Packets B recorded all of the dynamic state for after its own submit
        u64 vertex_cached{};
        u32 logged{};
        std::chrono::steady_clock::time_point last_report{};
    } back_verify;
    struct {
        u64 draws{};
        u64 differing{};
        u64 bytes{};
        u32 logged{};
        std::chrono::steady_clock::time_point last_report{};
    } packet_verify;
    /// Fragment output components the dynamic color write mask was last recorded with
    u32 last_color_output_mask{};
    VideoCommon::OotNetwork::BypassState oot_network_bypass;
    u32 mesh_emulation_groups{};             ///< mesh workgroups of the emulated draw
    u32 mesh_emulation_vertices_per_group{}; ///< 0 unless the draw is an emulated mesh draw
    /// Mesh emulation diagnostics with a task program: one draw's distribution arguments.
    struct {
        std::span<const u8> args;
        u64 tick{};
        u32 tasks{};
        u32 capacity{};
        bool pending{};
    } mesh_task_probe;
    /// With a task program: the vkCmdDrawIndirect arguments the distribution pass writes.
    struct {
        VkBuffer buffer{};
        VkDeviceSize offset{};
    } mesh_emulation_indirect;
    /// Mesh emulation diagnostics: one draw's output, read back once the GPU has written it.
    struct MeshEmulationProbe {
        std::span<const u8> data;
        u64 tick{};
        u64 frame{};
        u32 groups{};
        u32 block_words{};
        u32 vertex_lines{};
        u32 max_vertices{};
        u32 max_primitives{};
        std::array<u32, 4> position_lines{};
        bool pending{};
    } mesh_probe;
    void ReadMeshEmulationProbe();
    /// Mesh emulation with a task program: the output arena every task draw of a submission
    /// allocates its mesh workgroups from, and its host-visible counter of allocated words.
    struct MeshArena {
        vk::Buffer buffer;
        vk::Buffer counter;
        u64 size{};
        u64 tick{};
        bool counted{}; ///< the counter of its last submission was read
    };
    std::vector<MeshArena> mesh_arenas;
    size_t mesh_arena_current{~size_t{}};
    u64 mesh_arena_size{64ULL << 20};
    /// The current submission's arena, after its counter is reset.
    const MeshArena& AcquireMeshArena();

    /// Fast draw (nx2_fast_draw.h): what the last recorded draw left bound.
    struct ReusableDraw {
        GraphicsPipeline* pipeline{};
        const Tegra::Engines::Maxwell3D* maxwell3d{};
        u64 serial{};
        u64 map_serial{};
        u64 tick{};
        u64 render_pass{};
        Maxwell::PrimitiveTopology topology{};
        bool is_indexed{};
    };
    enum class FullDraw : u8 {
        NoCandidate,
        Kind,
        StateChanged,
        Submitted,
        RenderPass,
        DeltaPipeline,
        HandleBind,
        HandleData,
        SampledImages,
        Count,
    };
    /// Why a recorded draw can't be followed by a delta draw.
    enum class Ineligible : u8 {
        Indirect,
        Mesh,
        NotBuilt,
        WritesImages,
        TransformFeedback,
        InlineIndex,
        Quads,
        Repacked,
        NoRenderPass,
        PipelineNotBound,
        OffBackThread,
        Count,
    };
    enum class DrawPath : u8 { Full, Delta, Count };
    struct FastDrawStats {
        u64 delta{};
        std::array<u64, static_cast<size_t>(FullDraw::Count)> full{};
        std::array<u64, static_cast<size_t>(Ineligible::Count)> ineligible{};
        /// PrepareDraw time and draws per path.
        std::array<u64, static_cast<size_t>(DrawPath::Count)> prepare_ns{};
        std::array<u64, static_cast<size_t>(DrawPath::Count)> prepare_draws{};
        /// EDEN_NX2_FAST_DRAW_VERIFY
        u64 verified{};
        u64 pipeline_mismatches{};
        u64 view_mismatches{};
        u64 binding_mismatches{};
        /// DeltaPipeline draws by GraphicsPipeline::DeltaDrawBlocker.
        std::array<u64, static_cast<size_t>(GraphicsPipeline::DeltaBlocker::Count)> delta_blockers{};
        /// StateChanged draws by the reason of the last break, and by its 3D method.
        std::array<u64, static_cast<size_t>(VideoCommon::Nx2FastDraw::Reason::Count)> breaks{};
        std::array<u32, Maxwell::NUM_REGS> break_methods{};
        u64 window_start{};
    };
    /// The pipeline the draw can take the delta path with, or null for a full resolve. Clears
    /// the remembered draw either way: only a recorded draw remembers one again.
    GraphicsPipeline* TakeReusableDraw(bool is_indexed, bool is_indirect);
    /// Whether the draw TakeReusableDraw took still holds once the caches are locked (serial and
    /// map_serial as read then), and the images it samples need no preparing.
    bool DeltaStillHolds(u64 serial, u64 map_serial);
    void NoteResolvedDraw(GraphicsPipeline* pipeline, bool is_indexed, bool is_indirect, u64 serial,
                          u64 map_serial);
    void ReportFastDraws();
    ReusableDraw reusable_draw;
    ReusableDraw taken_draw; ///< The draw TakeReusableDraw last returned the pipeline of
    /// Device memory of the constant buffers TakeReusableDraw checks for texture handle writes.
    std::array<std::array<VideoCommon::Nx2FastDraw::DeviceSpans, Maxwell::MaxConstBuffers>,
               Maxwell::MaxShaderStage>
        handle_spans;
    /// The constant buffer binds and data since the previous draw.
    VideoCommon::Nx2FastDraw::Delta taken_delta;
    FastDrawStats fast_draw_stats;
    /// Nx2FastDraw::Enabled and VerifyEnabled as the drawing thread last read them.
    bool fast_draw_on{};
    bool fast_draw_verify{};

    struct IndirectTrace {
        vk::Buffer readback;
        u64 tick{};
        u64 frame{};
        GPUVAddr address{};
        GPUVAddr count_address{};
        u32 max_draws{};
        u32 sampled_commands{};
        bool pending{};
        bool include_count{};
        bool byte_count{};
    };
    std::array<IndirectTrace, 8> indirect_traces;
    Common::DkbFrameTrace::RateLimit trace_rate;
    Common::DkbFrameTrace::RateLimit trace_cpu_rate;
    std::optional<Common::DkbCpuTrace::Snapshot> trace_cpu_snapshot;
    u64 trace_cpu_frame{};
    bool trace_sample_frame{true};
    u64 trace_frame{};
    u64 trace_window_frames{};
    u64 trace_draws{};
    u64 trace_skipped{};
    u64 trace_indirect{};
    u64 trace_zero_direct{};
    /// Vertex and fragment hashes of the draw being recorded, for GPU timing.
    std::pair<u64, u64> trace_draw_hashes{};
    u32 trace_captures{};
    u32 trace_capture_dropped{};
    u32 trace_failure_details{};
};

} // namespace Vulkan
