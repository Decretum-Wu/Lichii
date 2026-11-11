// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <queue>
#include <source_location>
#include <span>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/polyfill_thread.h"
#include "common/settings.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_back_stages.h"
#include "video_core/nx2_draw_stages.h"
#include "video_core/renderer_vulkan/fragment_barrier_state.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
template <typename Trait>
class QueryCacheBase;
}

namespace Vulkan {

class CommandPool;
class Device;
class Framebuffer;
class GraphicsPipeline;
class StateTracker;
class UploadBatch;

struct QueryCacheParams;

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Device& device, StateTracker& state_tracker);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    u64 Flush(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
              std::source_location location = std::source_location::current());

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
                std::source_location location = std::source_location::current());

    /// On the split GPU thread, a fence or a command list copy submits only if the last
    /// submission is older than a window (EDEN_NX2_FENCE_FLUSH_US, default 500 us, 0 submits at
    /// once); otherwise it goes with the next one. The back submits deferred work before it
    /// sleeps and when another thread asks (WaitSubmitted). True when the flush was deferred.
    bool DeferFlush();
    /// Submits deferred work (the split's back, from its idle hook).
    void FlushDeferred();
    /// Threads other than the split's back: waits until tick is submitted, asking the back to
    /// submit deferred work (logged every 2 s while it waits). False only when the back thread
    /// is not running, so nothing will submit it: the caller then submits it itself.
    bool WaitSubmitted(u64 tick);
    /// Whether everything up to tick has been handed to the worker for submission.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept {
        return tick < master_semaphore->CurrentTick();
    }

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    enum class FrameMark : u8 { None, Start, End, Both };
    /// GPU thread, once per present: the frame-end mark this present gets. EDEN_DKB_FRAME_TRACE
    /// marks every present; otherwise only the two presents after RequestGpuBusySample.
    [[nodiscard]] FrameMark NextFrameMark();
    void MarkFrameEnd(u64 frame, FrameMark mark);
    /// Any thread: asks for one host GPU timestamp sample of the frame after the next present.
    void RequestGpuBusySample() {
        gpu_busy_marks.store(2, std::memory_order_relaxed);
    }
    /// Host GPU busy time of the last sampled frame, nullopt when none was taken in 3 s.
    [[nodiscard]] std::optional<double> GpuBusyMs() const;
    void TraceGraphicsPipeline(const std::array<u64, 6>& hashes);
    void TraceComputePipeline(u64 hash);
    /// Worker thread, inside a recorded command: timestamps one draw or dispatch of a sampled
    /// submission. A draw passes its vertex and fragment hashes, a dispatch its compute hash.
    template <typename Func>
    void TraceGpuWork(vk::CommandBuffer cmdbuf, u64 hash, u64 fragment_hash,
                      std::array<u32, 3> size, Func&& work) {
        if (!trace_gpu_recording || !trace_work_pool || trace_work_count == TraceMaxWork) {
            work();
            return;
        }
        const u32 query = trace_work_count * 2;
        trace_work_items[trace_work_count++] = {hash, fragment_hash, size};
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query);
        work();
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query + 1);
    }

    /// Sends currently recorded work to the worker thread.
    void DispatchWork();

    /// Requests to begin a renderpass.
    void RequestRenderpass(const Framebuffer* framebuffer,
                           std::source_location location = std::source_location::current());

    /// Defers a full-attachment color clear so it becomes the next render pass.
    bool DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot, const VkClearValue& value);

    /// Defers a full depth/stencil clear so it becomes the next render pass.
    bool DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value);

    /// Realizes any pending deferred clear before its framebuffer can be moved or freed.
    void FlushDeferredClear();

    /// Requests the current execution context to be able to execute operations only allowed outside
    /// of a renderpass.
    void RequestOutsideRenderPassOperationContext(
        std::source_location location = std::source_location::current());

    void DeferFragmentBarrier() {
        if (DeferToList(SpliceOp::DeferFragmentBarrier)) {
            return;
        }
        fragment_barrier.Defer(state.renderpass != VK_NULL_HANDLE);
    }

    bool HasFragmentBarrier() const {
        VideoCommon::DrawStages::CountInA(VideoCommon::DrawStages::Join::ASchedulerRead);
        return fragment_barrier.Pending();
    }

    void NoteRenderPassStores() {
        render_pass_has_stores = true;
    }

    bool RenderPassHasStores() const {
        return render_pass_has_stores;
    }

    /// A dispatch was recorded: the next render pass must see what it wrote.
    void NoteComputeWrites() {
        compute_writes_pending = true;
    }

    /// Returns true when a render pass is currently active in the scheduler state.
    bool IsRenderPassActive() const {
        VideoCommon::DrawStages::CountInA(VideoCommon::DrawStages::Join::ASchedulerRead);
        return state.renderpass != VK_NULL_HANDLE;
    }

    /// Changes whenever a render pass begins or ends.
    u64 RenderPassSerial() const {
        VideoCommon::DrawStages::CountInA(VideoCommon::DrawStages::Join::ASchedulerRead);
        return render_pass_serial;
    }

    /// The graphics pipeline bound in the current command buffer, if the scheduler knows it.
    const GraphicsPipeline* BoundGraphicsPipeline() const {
        return state.graphics_pipeline;
    }

    /// Update the pipeline to the current execution context.
    bool UpdateGraphicsPipeline(GraphicsPipeline* pipeline);

    /// Whether UpdateGraphicsPipeline(pipeline) would make the state enable flag dirty.
    [[nodiscard]] bool RefreshesStateEnable(const GraphicsPipeline* pipeline) const;

    /// Worker thread only: whether the bound graphics pipeline failed to build, so the draws
    /// recorded after it must be dropped instead of reaching the driver without a pipeline.
    void SetGraphicsPipelineMissing(bool missing) {
        graphics_pipeline_missing = missing;
    }
    [[nodiscard]] bool GraphicsPipelineMissing() const {
        return graphics_pipeline_missing;
    }

    /// Update the rescaling state. Returns true if the state has to be updated.
    bool UpdateRescaling(bool is_rescaling, bool mesh_layout);

    /// Returns true when the descriptor buffer chunk has to be bound into the command buffer.
    bool UpdateDescriptorBufferChunk(u32 descriptor_chunk);

    /// Invalidates current command buffer state except for render passes
    void InvalidateState();

    /// Assigns the query cache.
    void SetQueryCache(VideoCommon::QueryCacheBase<QueryCacheParams>& query_cache_) {
        query_cache = &query_cache_;
    }

    // Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// Consecutive in-order staging uploads share barriers until their destinations overlap,
    /// a source depends on a pending copy, or the batch is full. Call outside a render pass.
    void RecordUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies);
    /// Not while a draw's commands are captured: the batch would be recorded after them.
    [[nodiscard]] bool BatchesUploads() const noexcept {
        return Capturing() == nullptr && !capturing;
    }

    /// Draw packets, stage A (docs/nx2/cmdlist.md): until SpliceCommandList, recorded commands
    /// go to the draw's command list. A submit or dispatch records the list first.
    void BeginCommandList();
    /// Stage 4 (EDEN_NX2_BACK_STAGES=1, stage B on its own thread): the commands stage A records
    /// for a packet, with the scheduler state changes A asked for between them. B applies those
    /// where A asked for them as it splices the list.
    struct PacketList;
    enum class SpliceOp : u8 {
        EndRenderPass,         ///< RequestOutsideRenderPassOperationContext
        FlushDeferredClear,
        InvalidateState,       ///< The scheduler's part (A took the state tracker's)
        DeferFragmentBarrier,
        DispatchWork,
    };
    struct ListOp {
        u32 index; ///< Applied before the command with this index
        SpliceOp op;
    };
    /// Stage A, B on its own thread: the change goes to the packet's list (false otherwise).
    bool DeferToList(SpliceOp op);
    /// The calling thread (stage A) records into `list`; null records into the scheduler.
    static void CaptureInto(PacketList* list) noexcept {
        t_capture = list;
    }
    /// The list the calling thread records into: only the thread that owns the running ring
    /// has one, so a capture left on a thread whose ring stopped is never used.
    [[nodiscard]] static PacketList* Capturing() noexcept {
        return VideoCommon::DrawStages::OwnsRing() ? t_capture : nullptr;
    }
    /// Stage B, or stage A while B is idle: records `list` in order and empties it.
    void SpliceList(PacketList& list);
    /// Stage A: changes its lists asked for that may end the render pass or unbind the pipeline,
    /// and its direct uses of the scheduler.
    [[nodiscard]] static u64 AViewChanges() noexcept {
        return t_a_changes;
    }
    /// Submits stage B's thread made: A takes the state tracker's part of each (the command
    /// buffer state it holds), B the scheduler's.
    [[nodiscard]] u64 BackResets() const noexcept {
        return back_resets.load(std::memory_order_acquire);
    }

    /// With stage B on its own thread, on stage A's thread: waits until B consumed every packet
    /// (a join at `site`), then records into the scheduler directly while it lives.
    class DirectScope {
    public:
        explicit DirectScope(VideoCommon::DrawStages::Join site)
            : join{site, Capturing() != nullptr} {
            if (VideoCommon::DrawStages::OwnsRing()) {
                saved = std::exchange(t_capture, nullptr);
                t_a_changes += saved != nullptr ? 1 : 0;
            }
        }
        ~DirectScope() {
            if (saved != nullptr) {
                t_capture = saved;
            }
        }
        DirectScope(const DirectScope&) = delete;
        DirectScope& operator=(const DirectScope&) = delete;

    private:
        VideoCommon::DrawStages::JoinScope join;
        PacketList* saved{};
    };

    /// Stage B: records the draw's command list, in order, ahead of its own commands.
    void SpliceCommandList() {
        capturing = false;
        if (!command_list->Empty()) {
            SpliceCommands();
        }
    }

    /// Send work to a separate thread.
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer, vk::CommandBuffer>
    void RecordWithUploadBuffer(T&& command) {
        if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
            VideoCommon::GpuSplit::CheckOwner("Scheduler::Record");
        }
        if (cmdlist_verify) {
            NoteRecorded(CommandTag<std::remove_cvref_t<T>>(), HashCommand(command));
        }
        if (Capturing() != nullptr) {
            Capture(command);
            return;
        }
        if (capturing) {
            if (command_list->Record(command)) {
                return;
            }
            ++cmdlist_stats.full_joins;
            VideoCommon::DrawStages::Count(VideoCommon::DrawStages::Join::AListFull);
            SpliceCommands();
            (void)command_list->Record(command);
            return;
        }
        VideoCommon::DrawStages::Expect(VideoCommon::DrawStages::Stage::B, "Scheduler::Record");
        if (uploads_pending) {
            // Pending uploads precede every other recorded command, including upload-buffer work.
            FlushUploads();
        }
        ++record_serial;
        if (chunk->Record(command)) {
            return;
        }
        DispatchWork();
        (void)chunk->Record(command);
    }

    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void Record(T&& c) {
        this->RecordWithUploadBuffer(
            [command = std::move(c)](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
                command(cmdbuf);
            });
    }

    /// Whether anything was recorded since the last Flush or Finish, or waits to be: batched
    /// uploads and a deferred clear reach the command buffer only at the next record or submit.
    [[nodiscard]] bool RecordedSinceSubmit() const noexcept {
        return record_serial != submitted_serial || uploads_pending ||
               deferred_clear.framebuffer != nullptr;
    }
    /// Counts a flush skipped because nothing was recorded (GPU profile).
    static void CountSkippedFlush();

    /// Whether the recording thread submitted `tick` within 20 ms.
    [[nodiscard]] bool WaitForSubmission(u64 tick) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{20};
        while (tick >= master_semaphore->CurrentTick()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
        return true;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Changes whenever recorded command buffer state may have been reset (InvalidateState, the
    /// end of a render pass), so a binding recorded at the same epoch is still bound.
    [[nodiscard]] u64 RecordingEpoch() const noexcept {
        return recording_epoch;
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Waits for the given GPU tick, optionally pacing frames. Threads other than the one that
    /// records (the async present thread) pass may_flush = false: flushing from there records
    /// into the current chunk while that thread does (corrupt command buffers, crashes). They
    /// wait for the recording thread to submit the tick instead, and give up after 20 ms (it may
    /// be waiting for them).
    void Wait(u64 tick, double target_fps = 0.0, bool may_flush = true) {
        // Stage A's flushes go to B in its packets: B submitted them once it is idle.
        const DirectScope direct{VideoCommon::DrawStages::Join::Fence};
        const bool trace = Common::DkbFrameTrace::Enabled() && target_fps > 0.0;
        const auto trace_start = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (tick > 0 && (may_flush || WaitForSubmission(tick))) {
            if (tick >= master_semaphore->CurrentTick()) {
                Flush();
            }
            master_semaphore->Wait(tick);
        }
        const auto trace_waited = trace ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        double trace_requested_ms{};
        double trace_sleep_ms{};
        double trace_slept_ms{};
        if (Settings::values.use_speed_limit.GetValue() && target_fps > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (last_target_fps != target_fps) {
                frame_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / target_fps));
                max_frame_count = static_cast<int>(0.1 * target_fps);
                last_target_fps = target_fps;
                frame_counter = 0;
                start_time = now;
            }
            frame_counter++;
            auto target_time = start_time + frame_interval * frame_counter;
            if (target_time >= now) {
                constexpr auto spin_tail = std::chrono::milliseconds(1);
                auto sleep_time = target_time - now;
                if (trace) {
                    trace_requested_ms =
                        std::chrono::duration<double, std::milli>(sleep_time).count();
                }
                if (sleep_time > spin_tail * 2) {
                    if (trace) {
                        trace_sleep_ms =
                            std::chrono::duration<double, std::milli>(sleep_time - spin_tail).count();
                    }
                    const auto sleep_start = trace ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                    std::this_thread::sleep_for(sleep_time - spin_tail);
                    if (trace) {
                        trace_slept_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - sleep_start).count();
                    }
                }
                while (std::chrono::steady_clock::now() < target_time) {
                    std::this_thread::yield();
                }
            } else if (frame_counter > max_frame_count) {
                frame_counter = 0;
                start_time = now;
            }
        }
        if (trace) {
            static thread_local Common::DkbFrameTrace::RateLimit rate;
            const auto end = std::chrono::steady_clock::now();
            if (rate.Ready(end)) {
                LOG_INFO(Render_Vulkan,
                         "DKB host_pace target_fps={} enabled={} may_flush={} "
                         "cpu_wait_wall_ms={:.3f} pace_requested_ms={:.3f} "
                         "sleep_requested_ms={:.3f} sleep_actual_wall_ms={:.3f} "
                         "pace_actual_wall_ms={:.3f}",
                         target_fps, Settings::values.use_speed_limit.GetValue(), may_flush,
                         std::chrono::duration<double, std::milli>(trace_waited - trace_start).count(),
                         trace_requested_ms, trace_sleep_ms, trace_slept_ms,
                         std::chrono::duration<double, std::milli>(end - trace_waited).count());
            }
        }
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore& GetMasterSemaphore() const noexcept {
        return *master_semaphore;
    }

    std::mutex submit_mutex;

private:
    std::chrono::steady_clock::time_point last_flush{};
    bool flush_pending{};
    u64 record_serial{};    ///< Commands recorded
    u64 submitted_serial{}; ///< record_serial at the last Flush or Finish

    class CommandChunk;

    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const = 0;

        /// Moves the command to the end of `chunk`; false when it doesn't fit.
        virtual bool MoveTo(CommandChunk& chunk) = 0;

        [[nodiscard]] virtual uintptr_t Tag() const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const override {

            command(cmdbuf, upload_cmdbuf);
        }

        bool MoveTo(CommandChunk& chunk) override;

        [[nodiscard]] uintptr_t Tag() const override {
            return CommandTag<T>();
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        void ExecuteAll(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf);

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        /// Empties the chunk, passing each command in order to `take` before destroying it.
        template <typename Func>
        void TakeAll(Func&& take) {
            Command* command = std::exchange(first, nullptr);
            last = nullptr;
            command_offset = 0;
            while (command != nullptr) {
                Command* const next = command->GetNext();
                take(*command);
                command->~Command();
                command = next;
            }
        }

        bool Empty() const {
            return command_offset == 0;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        size_t command_offset = 0;
        bool submit = false;
        alignas(std::max_align_t) std::array<u8, 0x8000> data{};
    };

    struct State {
        VkRenderPass renderpass{};
        VkFramebuffer framebuffer{};
        VkExtent2D render_area = {0, 0};
        GraphicsPipeline* graphics_pipeline = nullptr;
        bool is_rescaling = false;
        bool rescaling_defined = false;
        bool mesh_layout = false;
        bool needs_state_enable_refresh = false;
        u32 descriptor_buffer_chunk = 0;
        bool descriptor_buffer_bound = false;
    };

    struct DeferredClear {
        const Framebuffer* framebuffer = nullptr;
        u32 color_clear_mask = 0;
        std::array<VkClearValue, 8> color_values{};
        bool depth_stencil = false;
        VkClearValue depth_stencil_value{};
    };

    /// Begins a render pass for the given framebuffer, optionally with clear values.
    void BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                             const VkClearValue* clear_values, u32 clear_value_count);

    /// If a deferred clear is pending.
    void RealizeDeferredClear();

    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffer();

    void TraceGpuSubmission();
    void ReportWorkTimes();
    void TraceShader(u64 hash, u32 stage);

    u64 SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore);

    void AllocateNewContext();

    void EndPendingOperations();

    void EndRenderPass();

    void AcquireNewChunk();

    /// Records the command list into the chunk (a join when stage A is still capturing).
    void SpliceCommands();

    void ApplySpliceOp(const ListOp& op);
    /// The scheduler's part of InvalidateState.
    void InvalidateRecordingState();

    template <typename T>
    void Capture(T& command);

    /// One identity per command type, for EDEN_NX2_CMDLIST_VERIFY.
    template <typename T>
    static uintptr_t CommandTag() {
        static const char anchor{};
        return reinterpret_cast<uintptr_t>(&anchor);
    }

    /// The captured arguments' bytes, or only their size when the capture has padding.
    template <typename T>
    static u64 HashCommand(const T& command) {
        u64 hash = 0xcbf29ce484222325ULL ^ sizeof(T);
        if constexpr (std::has_unique_object_representations_v<T>) {
            const auto* const bytes = reinterpret_cast<const unsigned char*>(&command);
            for (size_t index = 0; index < sizeof(T); ++index) {
                hash = (hash ^ bytes[index]) * 0x100000001b3ULL;
            }
        }
        return hash;
    }

    void NoteRecorded(uintptr_t tag, u64 hash);
    void NoteInserted(uintptr_t tag, u64 hash, u64 sequence);

    const Device& device;
    StateTracker& state_tracker;

    std::unique_ptr<MasterSemaphore> master_semaphore;
    FragmentBarrierState fragment_barrier;
    bool render_pass_has_stores{};
    bool compute_writes_pending{};
    bool graphics_pipeline_missing{};
    u64 render_pass_serial{};
    std::unique_ptr<CommandPool> command_pool;

    std::unique_ptr<UploadBatch> upload_batch;
    bool uploads_pending{};
    /// Records pending uploads before another command, worker handoff, or submission tick.
    void FlushUploads();
    /// Hands a non-empty chunk to the worker.
    void QueueChunk();

    vk::QueryPool trace_query_pool;
    static constexpr u32 TraceMaxSubmissions = 512;
    struct SubmissionTrace {
        struct ShaderUse {
            u64 hash{};
            u32 stage{};
        };
        u64 tick{};
        u32 draw_attempts{};
        u32 dispatch_attempts{};
        u32 num_shaders{};
        u32 untracked_uses{};
        std::array<ShaderUse, 32> shaders{};
    };
    SubmissionTrace trace_work;
    std::vector<SubmissionTrace> trace_submissions;
    static constexpr u32 TraceMaxWork = 4096;
    struct WorkTrace {
        u64 hash{};
        u64 fragment_hash{};
        /// Dispatch grid, or vertex and instance counts of a draw; zero when indirect.
        std::array<u32, 3> size{};
    };
    vk::QueryPool trace_work_pool;
    std::vector<WorkTrace> trace_work_items;
    u32 trace_work_count{};
    Common::DkbFrameTrace::RateLimit trace_gpu_rate;
    u64 trace_gpu_tick{};
    u64 trace_frame_end{};
    u64 trace_sample_frame{};
    u32 trace_gpu_count{};
    u64 trace_gpu_skipped{};
    u32 trace_timestamp_bits{};
    float trace_timestamp_period{};
    bool trace_gpu_recording{};
    bool trace_gpu_sampling{};
    bool trace_gpu_pending{};
    bool trace_gpu_failed{};
    FrameMark trace_frame_mark{};
    /// The pending sample covers exactly one frame, so its busy time can be shown.
    bool trace_gpu_publish{};

    std::atomic<u32> gpu_busy_marks{};
    struct GpuBusySample {
        double ms;
        std::chrono::steady_clock::time_point taken;
    };
    mutable std::mutex gpu_busy_mutex;
    std::optional<GpuBusySample> gpu_busy;

    VideoCommon::QueryCacheBase<QueryCacheParams>* query_cache = nullptr;

    vk::CommandBuffer current_cmdbuf;
    vk::CommandBuffer current_upload_cmdbuf;

    DeferredClear deferred_clear;

    std::unique_ptr<CommandChunk> chunk;
    std::function<void()> on_submit;

    /// Stage A's commands for the current draw packet, while capturing.
    std::unique_ptr<CommandChunk> command_list = std::make_unique<CommandChunk>();
    bool capturing{};
    const bool cmdlist_verify{std::getenv("EDEN_NX2_CMDLIST_VERIFY") != nullptr};
    struct CommandListVerifyEntry {
        uintptr_t tag;
        u64 hash;
        u64 sequence;
    };
    struct {
        u64 draws{};
        u64 lists{};          ///< Draws whose list held commands
        u64 commands{};       ///< Commands spliced
        u64 flush_joins{};    ///< Lists recorded early by a submit or dispatch during stage A
        u64 full_joins{};     ///< Lists recorded early because the list was full
        u64 recorded{};       ///< Verify: commands in recording order
        u64 inserted{};       ///< Verify: commands in command buffer order
        u64 differences{};
        std::vector<CommandListVerifyEntry> pending; ///< Verify: the list's commands
        std::chrono::steady_clock::time_point last_report{};
    } cmdlist_stats;

    State state;
    u64 recording_epoch = 0;

    static inline thread_local PacketList* t_capture{};
    static inline thread_local u64 t_a_changes{};
    std::atomic<u64> back_resets{};
    std::chrono::steady_clock::time_point list_verify_report{};

    u32 num_renderpass_images = 0;
    std::array<VkImage, 9> renderpass_images{};
    std::array<VkImageSubresourceRange, 9> renderpass_image_ranges{};

    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable_any event_cv;
    bool worker_waiting{}; ///< The worker is waiting for work (guarded by queue_mutex)
    std::jthread worker_thread;

    std::chrono::steady_clock::duration frame_interval{};
    std::chrono::steady_clock::time_point start_time{};
    double last_target_fps{};
    u64 max_frame_count{};
    u64 frame_counter{};
};

template <typename T>
bool Scheduler::TypedCommand<T>::MoveTo(CommandChunk& chunk) {
    return chunk.Record(command);
}

struct Scheduler::PacketList {
    PacketList() : commands{std::make_unique<CommandChunk>()} {}

    [[nodiscard]] bool Empty() const noexcept {
        return count == 0 && ops.empty();
    }

    std::unique_ptr<CommandChunk> commands;
    u32 count{}; ///< Commands in `commands`
    std::vector<ListOp> ops;
    /// EDEN_NX2_CMDLIST_VERIFY: what was recorded, in order.
    std::vector<CommandListVerifyEntry> pending;
};

template <typename T>
void Scheduler::Capture(T& command) {
    const uintptr_t tag = cmdlist_verify ? CommandTag<std::remove_cvref_t<T>>() : 0;
    const u64 hash = cmdlist_verify ? HashCommand(command) : 0;
    if (!t_capture->commands->Record(command)) {
        // A join splices the list while B is idle.
        ++cmdlist_stats.full_joins;
        VideoCommon::DrawStages::JoinB(VideoCommon::DrawStages::Join::AListFull);
        (void)t_capture->commands->Record(command);
    }
    if (cmdlist_verify) {
        t_capture->pending.push_back({tag, hash, t_capture->count});
    }
    ++t_capture->count;
}

} // namespace Vulkan
