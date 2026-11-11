// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "shader_recompiler/shader_info.h"
#include "tests/core/system_test_context.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_draw_stages.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace {

/// #version 450, layout(local_size_x = 1) in; void main() {}
constexpr std::array<u32, 70> EmptyCompute{
    0x07230203, 0x00010000, 0x0008000b, 0x0000000a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00040047, 0x00000009, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017,
    0x00000007, 0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x00000008, 0x00000001, 0x0006002c,
    0x00000007, 0x00000009, 0x00000008, 0x00000008, 0x00000008, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038,
};

} // Anonymous namespace

// Runs on any Vulkan driver, lavapipe included (mesa-vulkan-drivers):
//   ./tests "[vulkan_compute_descriptors]"
TEST_CASE("Vulkan compute descriptors: a dispatch's buffers go to the queue Configure counts "
          "while stage B has its own queue",
          "[vulkan_compute_descriptors]") {
    const auto library = Vulkan::OpenLibrary();
    REQUIRE(library);
    Vulkan::vk::InstanceDispatch dld;
    const Vulkan::vk::Instance instance = Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1);
    std::optional<Vulkan::Device> device;
    for (const VkPhysicalDevice physical : instance.EnumeratePhysicalDevices()) {
        try {
            device.emplace(*instance, Vulkan::vk::PhysicalDevice(physical, dld), VK_NULL_HANDLE,
                           dld);
            break;
        } catch (const Vulkan::vk::Exception&) {
            device.reset();
        }
    }
    if (!device) {
        SKIP("No Vulkan device");
    }
    INFO(device->GetDriverName() << " " << device->GetModelName());

    ScopedSystemTestContext context;
    Core::System system;
    system.Initialize();
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull null_rasterizer{gpu};
    device_memory.BindInterface(&null_rasterizer);
    Tegra::MemoryManager gpu_memory{system, device_memory, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&null_rasterizer);
    Tegra::Control::ChannelState channel{0};
    channel.memory_manager = &gpu_memory;
    channel.payload.emplace(system, gpu_memory, channel);
    Tegra::Engines::KeplerCompute& kepler_compute = channel.payload->kepler_compute;

    Vulkan::MemoryAllocator allocator{*device};
    Vulkan::StateTracker state_tracker;
    Vulkan::Scheduler scheduler{*device, state_tracker};
    Vulkan::StagingBufferPool staging_pool{*device, allocator, scheduler};
    Vulkan::DescriptorPool descriptor_pool{*device, scheduler};
    Vulkan::GuestDescriptorQueue guest_queue{*device,
                                             Vulkan::UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                                             device->IsExtDescriptorBufferSupported()};
    Vulkan::ComputePassDescriptorQueue compute_pass_queue{
        *device, Vulkan::UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE};
    // As RasterizerVulkan builds stage B's queue with EDEN_NX2_BACK_STAGES=1.
    Vulkan::GuestDescriptorQueue back_queue{*device, 1, device->IsExtDescriptorBufferSupported()};
    Vulkan::DescriptorBufferRing descriptor_buffer_ring{*device, allocator};
    Vulkan::BlitImageHelper blit_image{*device, scheduler, state_tracker, descriptor_pool};
    Vulkan::RenderPassCache render_pass_cache{*device};
    Vulkan::TextureCacheRuntime texture_runtime{*device,      scheduler,         allocator,
                                                staging_pool, blit_image,        render_pass_cache,
                                                descriptor_pool, compute_pass_queue};
    Vulkan::TextureCache texture_cache{texture_runtime, device_memory};
    Vulkan::BufferCacheRuntime buffer_runtime{*device,         allocator,          scheduler,
                                              staging_pool,    staging_pool,       guest_queue,
                                              compute_pass_queue, descriptor_pool};
    buffer_runtime.SetStageDescriptorQueue(back_queue);
    Vulkan::BufferCache buffer_cache{device_memory, buffer_runtime};
    buffer_cache.CreateChannel(channel);
    buffer_cache.BindToChannel(channel.bind_id);
    texture_cache.CreateChannel(channel);
    texture_cache.BindToChannel(channel.bind_id);

    // One uniform and one storage buffer, both read through constant buffer 0 (unmapped, so
    // they bind the null buffer: each still takes its descriptor entry).
    Shader::Info info{};
    info.constant_buffer_mask = 1;
    info.constant_buffer_used_sizes[0] = 0x100;
    info.constant_buffer_descriptors.push_back({.index = 0, .count = 1});
    info.storage_buffers_descriptors.push_back(
        {.cbuf_index = 0, .cbuf_offset = 0x10, .count = 1, .is_written = false});
    kepler_compute.launch_description.const_buffer_enable_mask.Assign(1);
    kepler_compute.launch_description.const_buffer_config[0].size.Assign(0x100);

    Vulkan::vk::ShaderModule module = device->GetLogical().CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = EmptyCompute.size() * sizeof(u32),
        .pCode = EmptyCompute.data(),
    });
    Vulkan::vk::PipelineCache pipeline_cache;
    Vulkan::ComputePipeline pipeline{*device,
                                     scheduler,
                                     pipeline_cache,
                                     nullptr,
                                     0,
                                     descriptor_pool,
                                     guest_queue,
                                     descriptor_buffer_ring,
                                     Vulkan::PipelineBuildQueue{},
                                     nullptr,
                                     nullptr,
                                     info,
                                     std::move(module),
                                     0x5eed,
                                     {}};

    std::optional<Vulkan::ByteAddressedShadow> shadow;
    bool configured{};
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        configured = pipeline.Configure(kepler_compute, gpu_memory, scheduler, buffer_cache,
                                        texture_cache, texture_runtime, shadow, nullptr);
    }
    REQUIRE(guest_queue.UpdateCount() == 2);
    REQUIRE(configured);
    REQUIRE(&buffer_runtime.StageDescriptorQueue() == &guest_queue);

    // Stage B's thread binds into its own queue, which fills the entries A left.
    const Vulkan::GuestDescriptorQueue* stage_b_queue{};
    std::thread stage_b([&] {
        VideoCommon::DrawStages::t_stage_b_thread = true;
        stage_b_queue = &buffer_runtime.StageDescriptorQueue();
    });
    stage_b.join();
    REQUIRE(stage_b_queue == &back_queue);
}
