// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#include "common/common_types.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/nv01_timer.h"
#include "video_core/dma_pusher.h"

namespace Core {
class System;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace VideoCommon {
class Maxwell3DLane;
}

namespace Tegra {

class GPU;
class MemoryManager;

namespace Control {

struct ChannelState {
    explicit ChannelState(s32 bind_id);

    void Init(Core::System& system, u64 program_id);

    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    struct Payload {
        explicit Payload(Core::System& system, MemoryManager& memory_manager, ChannelState& channel_state);
        ~Payload();

        /// The 3D engine the rasterizer and its caches read: with the split's preparation lane
        /// (EDEN_NX2_PREP_LANE), a mirror of maxwell_3d's state (VideoCommon::Maxwell3DLane).
        [[nodiscard]] Engines::Maxwell3D& PrepMaxwell3D() {
            return prep_maxwell_3d ? *prep_maxwell_3d : maxwell_3d;
        }

        /// 3D engine
        Engines::Maxwell3D maxwell_3d;
        std::optional<Engines::Maxwell3D> prep_maxwell_3d;
        std::unique_ptr<VideoCommon::Maxwell3DLane> maxwell_3d_lane;
        /// 2D engine
        Engines::Fermi2D fermi_2d;
        /// Compute engine
        Engines::KeplerCompute kepler_compute;
        /// DMA engine
        Engines::MaxwellDMA maxwell_dma;
        /// Inline memory engine
        Engines::KeplerMemory kepler_memory;
        /// NV01 Timer
        Engines::Nv01Timer nv01_timer;
        DmaPusher dma_pusher;
    };
    std::optional<Payload> payload;
    MemoryManager* memory_manager = nullptr;

    s32 bind_id = -1;
    u64 program_id = 0;
    bool initialized = false;
};

} // namespace Control

} // namespace Tegra
