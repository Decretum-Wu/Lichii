// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "common/cityhash.h"
#include "common/common_types.h"

// Stage 4 of the draw packets (EDEN_NX2_BACK_STAGES=1, docs/nx2/cmdlist.md): stage B uploads the
// CPU-modified vertex, index and uniform bytes of a packet from a copy A made at the draw, where
// the GPU thread read them before stage B had its own thread.
namespace VideoCommon::DrawStages {

/// Device pages B found holding the guest's bytes in their buffer. Every CPU mark of the memory
/// tracker clears a page (under the buffer cache's lock); only B sets one, under the lock, once a
/// page it settled from a snapshot still holds the snapshot's bytes. Stage A reads them without
/// the lock: a page not set may be CPU-modified.
class CleanPages {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr u64 AddressBits = 36;
    static constexpr u64 Pages = u64{1} << (AddressBits - PageBits);

    void Enable() {
        if (!words) {
            words = std::make_unique<std::atomic<u64>[]>(Pages / 64);
        }
    }

    [[nodiscard]] bool Enabled() const noexcept {
        return words != nullptr;
    }

    void Clear(u64 addr, u64 size) noexcept {
        Apply(addr, size, [](std::atomic<u64>& word, u64 mask) {
            word.fetch_and(~mask, std::memory_order_release);
        });
    }

    void Set(u64 addr, u64 size) noexcept {
        Apply(addr, size, [](std::atomic<u64>& word, u64 mask) {
            word.fetch_or(mask, std::memory_order_release);
        });
    }

    /// The runs of pages over [addr, addr + size) not set, as page-aligned [begin, end).
    template <typename Func>
    void ForEachUnclean(u64 addr, u64 size, Func&& func) const {
        if (size == 0) {
            return;
        }
        const u64 first = addr >> PageBits;
        const u64 end = (std::min)(((addr + size - 1) >> PageBits) + 1, Pages);
        u64 run = end;
        u64 word = 0;
        for (u64 page = first; page < end; ++page) {
            if (page == first || (page & 63) == 0) {
                word = words ? words[page >> 6].load(std::memory_order_acquire) : 0;
                if (word == ~u64{0} && (page & 63) == 0 && page + 64 <= end) {
                    if (run != end) {
                        func(run << PageBits, page << PageBits);
                        run = end;
                    }
                    page += 63;
                    continue;
                }
            }
            const bool clean = ((word >> (page & 63)) & 1) != 0;
            if (!clean && run == end) {
                run = page;
            } else if (clean && run != end) {
                func(run << PageBits, page << PageBits);
                run = end;
            }
        }
        if (run != end) {
            func(run << PageBits, end << PageBits);
        }
    }

private:
    template <typename Op>
    void Apply(u64 addr, u64 size, Op&& op) noexcept {
        if (!words || size == 0) {
            return;
        }
        u64 page = addr >> PageBits;
        const u64 end = (std::min)(((addr + size - 1) >> PageBits) + 1, Pages);
        while (page < end) {
            const u64 bit = page & 63;
            const u64 count = (std::min)(64 - bit, end - page);
            const u64 mask = count == 64 ? ~u64{0} : ((u64{1} << count) - 1) << bit;
            op(words[page >> 6], mask);
            page += count;
        }
    }

    std::unique_ptr<std::atomic<u64>[]> words;
};

/// The bytes stage A copied for one packet: whole pages, sorted and disjoint. The packet's slot
/// owns it, so A writes it again only after B consumed the packet (BackRing::Acquire).
struct StageSnapshot {
    static constexpr size_t MaxKeptBytes = 4ULL << 20;

    struct Chunk {
        u64 begin;
        u64 size;
        u64 offset;
        u64 hash; ///< EDEN_NX2_BACK_STAGES_VERIFY: the bytes as A copied them
    };
    struct Range {
        u64 begin;
        u64 size;
        bool whole; ///< Every page, the clean ones too: B reads the range from guest memory
    };

    std::vector<Chunk> chunks;
    std::vector<u8> bytes;

    void Reset() {
        chunks.clear();
        if (bytes.capacity() > MaxKeptBytes) {
            bytes = {};
        }
        bytes.clear();
    }

    /// Stage A: copies the pages of `ranges` that are not clean with `read(addr, out, size)`.
    template <typename Read>
    void Capture(std::span<const Range> ranges, const CleanPages& clean, Read&& read, bool hash) {
        Reset();
        pending.clear();
        constexpr u64 Mask = CleanPages::PageSize - 1;
        for (const Range& range : ranges) {
            if (range.begin == 0 || range.size == 0) {
                continue;
            }
            if (range.whole) {
                pending.emplace_back(range.begin & ~Mask, (range.begin + range.size + Mask) & ~Mask);
                continue;
            }
            clean.ForEachUnclean(range.begin, range.size,
                                 [this](u64 begin, u64 end) { pending.emplace_back(begin, end); });
        }
        if (pending.empty()) {
            return;
        }
        std::ranges::sort(pending);
        u64 total = 0;
        for (const auto& [begin, end] : pending) {
            if (!chunks.empty() && begin <= chunks.back().begin + chunks.back().size) {
                Chunk& last = chunks.back();
                const u64 last_end = last.begin + last.size;
                if (end > last_end) {
                    total += end - last_end;
                    last.size = end - last.begin;
                }
                continue;
            }
            chunks.push_back({begin, end - begin, total, 0});
            total += end - begin;
        }
        bytes.resize(total);
        for (Chunk& chunk : chunks) {
            read(chunk.begin, bytes.data() + chunk.offset, chunk.size);
            if (hash) {
                chunk.hash = HashChunk(chunk);
            }
        }
    }

    /// The copy of [addr, addr + size) when one chunk holds all of it.
    [[nodiscard]] const u8* Find(u64 addr, u64 size) const noexcept {
        const auto it = std::ranges::upper_bound(chunks, addr, {}, &Chunk::begin);
        if (it == chunks.begin()) {
            return nullptr;
        }
        const Chunk& chunk = *std::prev(it);
        if (addr + size > chunk.begin + chunk.size) {
            return nullptr;
        }
        return bytes.data() + chunk.offset + (addr - chunk.begin);
    }

    /// Each part of [addr, addr + size) a chunk holds, with its copy.
    template <typename Func>
    void ForEachCovered(u64 addr, u64 size, Func&& func) const {
        for (const Chunk& chunk : chunks) {
            const u64 begin = (std::max)(addr, chunk.begin);
            const u64 end = (std::min)(addr + size, chunk.begin + chunk.size);
            if (begin < end) {
                func(begin, end - begin, bytes.data() + chunk.offset + (begin - chunk.begin));
            }
        }
    }

    [[nodiscard]] u64 HashChunk(const Chunk& chunk) const noexcept {
        return Common::CityHash64(reinterpret_cast<const char*>(bytes.data() + chunk.offset),
                                  chunk.size);
    }

    [[nodiscard]] u64 Bytes() const noexcept {
        return bytes.size();
    }

private:
    std::vector<std::pair<u64, u64>> pending;
};

/// Stage B, under the buffer cache's lock, for [addr, addr + size) of a buffer the packet reads.
/// `scan(begin, size, found)`, the tracker's upload scan that clears what it finds, runs only over
/// the pages the snapshot holds, and gives each CPU-modified run it finds to `found`, to be
/// uploaded from the snapshot. A page the snapshot doesn't hold was clean at A: a CPU mark there
/// came after A and stays for the next draw. Then each page the snapshot holds is clean while
/// guest memory (`read`) still has the snapshot's bytes, and `mark`ed CPU-modified again when a
/// write after A's copy changed it: the scan cleared its mark, and the write may not have made one.
template <typename Scan, typename Found, typename Read, typename Mark>
void SettleSnapshot(const StageSnapshot& snapshot, u64 addr, u64 size, CleanPages& clean,
                    Scan&& scan, Found&& found, Read&& read, Mark&& mark,
                    std::vector<u8>& scratch) {
    snapshot.ForEachCovered(addr, size,
                            [&](u64 begin, u64 length, const u8*) { scan(begin, length, found); });
    constexpr u64 Mask = CleanPages::PageSize - 1;
    const u64 first = addr & ~Mask;
    snapshot.ForEachCovered(
        first, ((addr + size + Mask) & ~Mask) - first, [&](u64 begin, u64 length, const u8* bytes) {
            scratch.resize(length);
            read(begin, scratch.data(), length);
            u64 run = 0;
            for (u64 offset = 0; offset < length; offset += CleanPages::PageSize) {
                if (std::memcmp(scratch.data() + offset, bytes + offset, CleanPages::PageSize) ==
                    0) {
                    continue;
                }
                if (offset > run) {
                    clean.Set(begin + run, offset - run);
                }
                mark(begin + offset, CleanPages::PageSize);
                run = offset + CleanPages::PageSize;
            }
            if (length > run) {
                clean.Set(begin + run, length - run);
            }
        });
}

} // namespace VideoCommon::DrawStages
