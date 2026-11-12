// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/front_qmd.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_stats.h"
#include "video_core/nx2.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_list_stats.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

namespace {

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        const VideoCommon::StutterTrace::ThreadScope stutter_thread{
            VideoCommon::StutterTrace::Thread::Front};
        Common::SetCurrentThreadName("GPU");
        VideoCommon::GpuSplit::MarkGpuThread();
#ifdef __linux__
        // The pending-channel waits below sleep 500 us; the default timer slack adds up to 50 more.
        prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Settings::values.pin_gpu_threads.GetValue()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the waits below).
        auto last_progress = std::chrono::steady_clock::now();
        // For this long after that the thread waits in hardware, then it blocks in the OS. In
        // hardware it looks at the words the channels wait on every WordCheckPeriod and polls the
        // channels in full every PendingPollPeriod.
        constexpr std::chrono::nanoseconds PendingPoll = std::chrono::microseconds{500};
        constexpr std::chrono::nanoseconds PendingPollPeriod = std::chrono::microseconds{20};
        constexpr std::chrono::nanoseconds WordCheckPeriod = std::chrono::microseconds{2};
        // How the last wait ended, counted when a channel moves after it.
        enum class WaitEnd { None, Woken, PollTimedOut, SleepTimedOut };
        WaitEnd last_wait = WaitEnd::None;
        auto& wake = Tegra::Nx2PendingWake::Get();
        auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
        namespace AcquireStats = VideoCommon::Nx2AcquireStats;
        using FrontWait = AcquireStats::FrontWait;
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        namespace WaitStats = VideoCommon::GpuWaitStats;
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            WaitStats::MaybeReport(split.Active());
            VideoCommon::Nx2DrainCheck::MaybeReport();
            VideoCommon::Nx2AcquireStats::MaybeReport();
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores and list copies: a new command, a GPU
                // release or a finished copy wakes the wait below at once (Nx2PendingWake), so
                // arm it before looking.
                wake.Arm();
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                if (split.Active()) {
                    // The back may have run everything sent meanwhile.
                    VideoCommon::FrontQmdView::Get().MaybeTakeOver(
                        split, AcquireStats::QmdTakeOver::BetweenCommands);
                }
                const bool popped = state.queue.TryPop(next);
                if (resumed || popped) {
                    if (last_wait == WaitEnd::PollTimedOut) {
                        AcquireStats::NoteFrontWait(FrontWait::FoundPolling);
                    } else if (last_wait == WaitEnd::SleepTimedOut) {
                        AcquireStats::NoteFrontWait(FrontWait::FoundLate);
                    }
                    last_wait = WaitEnd::None;
                    last_progress = std::chrono::steady_clock::now();
                }
                if (!popped) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    // Words sent since the last publish would wait out the waits below while the
                    // back runs out of work: hand them over first.
                    if (split.Active()) {
                        split.Publish();
                    }
                    // Mario Kart World's channels wait for semaphores its CPU releases, stores
                    // that send no wake, and the CPU usually releases soon after: for half a
                    // millisecond wait in hardware, watching the semaphore words, then block in
                    // the OS for 500 us at a time. A full poll every 20 us catches what sends no
                    // wake and changes no word, such as a release a shader wrote to the caches.
                    // Ocarina of Time waits here ~330 times a second and its present waits for the
                    // work behind them: with 1 ms sleeps the present waited ~17 ms a frame in
                    // Kokiri village, with 50 us ~1 ms.
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    const WaitStats::IdleScope idle{WaitStats::g_front_poll_ns};
                    const auto now = std::chrono::steady_clock::now();
                    bool woken = false;
                    if (now - last_progress < PendingPoll) {
                        AcquireStats::NoteFrontWait(FrontWait::Hardware);
                        last_wait = WaitEnd::PollTimedOut;
                        const auto poll_end =
                            std::min(now + PendingPollPeriod, last_progress + PendingPoll);
                        auto until = now;
                        do {
                            until = std::min(std::chrono::steady_clock::now() + WordCheckPeriod,
                                             poll_end);
                            if (wake.Poll(until)) {
                                woken = true;
                                break;
                            }
                            if (waiters.WordStored()) {
                                AcquireStats::NoteFrontWait(FrontWait::Stored);
                                last_wait = WaitEnd::Woken;
                                break;
                            }
                        } while (until < poll_end);
                    } else {
                        AcquireStats::NoteFrontWait(FrontWait::Os);
                        last_wait = WaitEnd::SleepTimedOut;
                        woken = wake.Sleep(std::chrono::microseconds{500});
                    }
                    if (woken) {
                        AcquireStats::NoteFrontWait(FrontWait::Woken);
                        last_wait = WaitEnd::Woken;
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                const WaitStats::IdleScope idle{WaitStats::g_front_idle_ns};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (split.Active()) {
                // Between commands the back may have run out of work, and everything sent.
                split.PublishSoon();
                VideoCommon::FrontQmdView::Get().MaybeTakeOver(
                    split, VideoCommon::Nx2AcquireStats::QmdTakeOver::BetweenCommands);
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                    // Maps and flushes may change what the snapshot copier reads.
                    Tegra::DmaPusher::WaitForSnapshotCopies();
                }
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    Profile::Scope scope{Profile::Zone::Push};
                    Tegra::Nx2ListStats::NoteListQueued(submit_list->channel, -1);
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    // The sync operations (the back may run them between any two entries too,
                    // GpuSplit::RequestTick): flushes write guest memory where GPU work wrote.
                    // TickWork ends fast draw reuse itself when it runs a request.
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); },
                                VideoCommon::GpuSplit::Writes::Caches,
                                VideoCommon::GpuSplit::FastDrawNeutral);
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
                        split.DeferFunction(std::move(run->func));
                        split.Drain(VideoCommon::DrainSite::Map);
                    } else {
                        split.DeferFunction(std::move(run->func));
                        split.TagUnplaced(VideoCommon::UnplacedKind::RunCommand); // TEST
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    // Writes guest memory with what the caches hold of it.
                    split.Defer([rasterizer_ptr, addr = flush->addr,
                                 size = flush->size] { rasterizer_ptr->FlushRegion(addr, size); },
                                VideoCommon::GpuSplit::Writes::Caches);
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer(
                        [rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                            rasterizer_ptr->OnCacheInvalidation(addr, size);
                        },
                        VideoCommon::GpuSplit::Writes::Noted);
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence,
                             block = next.block] { SignalCommand(*state_ptr, fence, block); },
                            VideoCommon::GpuSplit::Writes::Noted,
                            VideoCommon::GpuSplit::FastDrawNeutral);
                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                Tegra::DmaPusher::WaitForSnapshotCopies();
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Push};
                Tegra::Nx2ListStats::NoteListQueued(submit_list->channel, -1);
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    Tegra::Nx2ListStats::NoteListQueued(channel, 1);
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    Tegra::Nx2PendingWake::Get().Notify();

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
