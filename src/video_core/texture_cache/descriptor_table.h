// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T, typename Id = u32>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        // Cleared on the next read.
        invalidated = true;
        ++changes;
    }

    /// The entries bound to these ids, and those bound to none, are read anew: the texture cache
    /// passes the views of each image it deletes, dozens of times a frame while its collector
    /// runs. Other entries keep their ids. Applied on the next read or ApplyForgotten.
    void Forget(std::span<const Id> ids) {
        for (const Id id : ids) {
            const u32 value = IdIndex(id);
            if (value / 64 >= forgotten_bits.size()) {
                forgotten_bits.resize(value / 64 + 1);
            }
            u64& word = forgotten_bits[value / 64];
            const u64 bit = 1ULL << (value % 64);
            // Once per id: a table nobody reads keeps its queue no longer than the ids in use.
            if ((word & bit) == 0) {
                word |= bit;
                forgotten_ids.push_back(id);
            }
        }
        has_forgotten = true;
    }

    void ApplyForgotten() noexcept {
        if (!has_forgotten || invalidated) {
            return;
        }
        // Entries without a view are read again too, as a full invalidation did: whether a
        // descriptor's address is mapped can change without its bytes changing.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        for (size_t word = 0; word < words; ++word) {
            for (u64 bits = read_descriptors[word]; bits != 0; bits &= bits - 1) {
                const u32 bit = static_cast<u32>(std::countr_zero(bits));
                Id& id = bound_ids[word * 64 + bit];
                const u32 value = IdIndex(id);
                if (id == Id{0} || (value / 64 < forgotten_bits.size() &&
                                    ((forgotten_bits[value / 64] >> (value % 64)) & 1) != 0)) {
                    id = Id{0};
                    read_descriptors[word] &= ~(1ULL << bit);
                }
            }
        }
        ClearForgotten();
    }

    std::span<Id> LiveBoundIds() noexcept {
        return std::span{bound_ids}.first(
            (std::min)(bound_ids.size(), size_t{current_limit} + 1));
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<T, bool> Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        } else if (has_forgotten) {
            ApplyForgotten();
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        ++changes;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors, Id{0});
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    std::vector<Id> forgotten_ids;   ///< Queued by Forget, to clear their bits once applied
    std::vector<u64> forgotten_bits; ///< One bit per forgotten id
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read
    bool has_forgotten{}; ///< Forget was called since the last apply
    /// Bumped by every Invalidate and Refresh: the ids and descriptors read before are only good
    /// while it stays the same. Forget leaves it alone.
    u64 changes{};

private:
    static constexpr u32 IdIndex(Id id) noexcept {
        if constexpr (std::is_integral_v<Id>) {
            return id;
        } else {
            return id.index;
        }
    }

    void ClearReadDescriptors() noexcept {
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
        ClearForgotten();
    }

    void ClearForgotten() noexcept {
        for (const Id id : forgotten_ids) {
            forgotten_bits[IdIndex(id) / 64] = 0;
        }
        forgotten_ids.clear();
        has_forgotten = false;
    }
};

} // namespace VideoCommon
