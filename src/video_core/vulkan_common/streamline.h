// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline. Only Streamline's MIT-licensed
// headers are built in: NVIDIA's DLLs are loaded at run time from the user's PC, and none ship
// with the emulator.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Streamline {

/// NVIDIA's files DLSS needs, as found on this PC. Paths are empty when missing.
struct DlssFiles {
    std::filesystem::path folder;     ///< Where the loader is, or where it should go
    std::filesystem::path interposer; ///< sl.interposer.dll (Streamline's loader)
    std::filesystem::path model;      ///< nvngx_dlss.dll next to the loader
    std::filesystem::path app_model;  ///< Newest DLSS the NVIDIA app downloaded
    std::string interposer_version;
    std::string model_version;
    std::string app_model_version;
    /// File names of the Streamline plugins and NVIDIA files DLSS Frame Generation needs that
    /// are not in the loader's folder.
    std::vector<std::string> frame_gen_missing;
};

[[nodiscard]] DlssFiles FindDlssFiles();

/// The user folder's "streamline" folder.
[[nodiscard]] std::filesystem::path UserFolder();

/// Streamline's own modules carry a second, Streamline-only signature.
[[nodiscard]] bool IsStreamlineModule(const std::filesystem::path& path);

/// A valid Authenticode signature from NVIDIA Corporation (DLSS models).
[[nodiscard]] bool IsSignedByNvidia(const std::filesystem::path& path);

/// When the NVIDIA DLSS setting is on, loads the user's sl.interposer.dll (checking NVIDIA's
/// signature) and starts Streamline for DLSS. Call before creating the renderer's instance.
bool Start();

/// Whether Streamline is running and waiting for the renderer's device.
[[nodiscard]] bool WantsDevice();

/// Instance and device extensions DLSS asks for, and DLSS Frame Generation while it is still
/// possible (empty when Streamline isn't running).
[[nodiscard]] std::vector<std::string> InstanceExtensions();
[[nodiscard]] std::vector<std::string> DeviceExtensions();

/// The NVIDIA DLSS Frame Generation setting with NVIDIA DLSS on. EDEN_NX2_DLSS_FG=0/1 overrides
/// the setting.
[[nodiscard]] bool FrameGenRequested();

/// Whether frame generation is still possible on the device being created: Streamline loaded
/// DLSS-G, Reflex and PCL and nothing has turned it off since.
[[nodiscard]] bool FrameGenPending();

/// Turns frame generation off before the device is attached; the reason goes to the log.
void DropFrameGen(std::string_view reason);

struct QueueCounts {
    u32 graphics{};
    u32 compute{};
};

/// Queues Streamline needs for frame generation beyond the renderer's own, all taken from the
/// graphics family (zero unless FrameGenPending()).
[[nodiscard]] QueueCounts ExtraQueues();

/// Vulkan 1.2 and 1.3 features frame generation needs, named as in VkPhysicalDeviceVulkan12Features
/// and VkPhysicalDeviceVulkan13Features (empty unless FrameGenPending()).
[[nodiscard]] std::vector<std::string> FrameGenDeviceFeatures();

/// Hands the renderer's device to Streamline and routes the swapchain calls through it, as
/// Streamline's manual hooking needs. Stops Streamline and returns false when DLSS can't run.
/// first_extra_queue is the first queue of graphics_family created for ExtraQueues(): the
/// graphics ones, then the compute ones.
bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, u32 first_extra_queue, vk::DeviceDispatch& dld);

/// Whether DLSS can run on the attached device.
[[nodiscard]] bool IsDlssAvailable();

/// Whether DLSS Frame Generation can run on the attached device: DLSS-G, Reflex and PCL are
/// supported and GetApi() has their functions.
[[nodiscard]] bool FrameGenAvailable();

/// DLSS-G finds the game's window from the surface its swapchain is made for, and Streamline only
/// learns a surface's window when the surface is created through sl.interposer.dll, which works
/// only once the device is attached. Creates a surface for this HWND that way (VK_NULL_HANDLE on
/// failure); destroy it with the instance's own vkDestroySurfaceKHR.
[[nodiscard]] VkSurfaceKHR CreateFrameGenSurface(VkInstance instance, void* window);

/// Turns DLSS Frame Generation off for the rest of the session (unloads DLSS-G, so the swapchain
/// calls reach the driver's swapchain); the reason goes to the log.
void DisableFrameGen(std::string_view reason);

/// Shuts Streamline down; call before the device is destroyed.
void Stop();

} // namespace Vulkan::Streamline
