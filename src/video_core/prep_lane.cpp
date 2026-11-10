// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/prep_lane.h"

namespace VideoCommon {

namespace {

using Maxwell3D = Tegra::Engines::Maxwell3D;

/// A record's sections: a word with the tag in its low byte, then the section's words.
enum Tag : u32 {
    TagRegs = 1,        ///< Bits 8-15 count, 16-31 first register; the values
    TagDirty,           ///< The dirty flags set since the last record
    TagStage,           ///< Bits 8-15 stage; its constant buffer bindings
    TagRootTable,       ///< Bits 8-15 table; its 64 words
    TagRootState,       ///< Selected table, offset, loaded
    TagEngineState,     ///< EngineHint
    TagReplaceTable,    ///< Count, then key low, key high, value of each entry
    TagDraw,            ///< DrawState's words, then the inline index bytes' count and bytes
    TagBreak,           ///< A fast draw break of the method lane's
    TagFastDelta,       ///< The fast draw delta the method lane collected
    TagVerify,          ///< EDEN_NX2_PREP_ORDER_VERIFY: the engine's whole state
};

/// What a draw takes from the draw manager, without its inline index buffer.
struct DrawState {
    Maxwell3D::Regs::PrimitiveTopology topology;
    Maxwell3D::DrawManager::DrawMode draw_mode;
    u32 draw_indexed;
    u32 draw_element_count;
    u32 base_index;
    Maxwell3D::Regs::VertexBuffer vertex_buffer;
    Maxwell3D::Regs::IndexBuffer index_buffer;
    u32 base_instance;
    u32 instance_count;
    u32 drawn_instances;
    Maxwell3D::DrawManager::DrawTextureState draw_texture_state;
    // IndirectParams, field by field: its padding would make copies compare unequal.
    u32 indirect_flags; ///< is_byte_count, is_indexed, include_count in bits 0-2
    u32 padding;
    u64 count_start_address;
    u64 indirect_start_address;
    u64 buffer_size;
    u64 max_draw_counts;
    u64 stride;
};
static_assert(std::is_trivially_copyable_v<DrawState>);

constexpr size_t Words(size_t bytes) {
    return (bytes + sizeof(u32) - 1) / sizeof(u32);
}

constexpr size_t DrawStateWords = Words(sizeof(DrawState));
constexpr size_t StageWords = Words(sizeof(Maxwell3D::State::ShaderStageInfo));
constexpr size_t DeltaWords = Words(sizeof(Nx2FastDraw::Delta));
constexpr size_t RegBlock = 64;
static_assert(Maxwell3D::Regs::NUM_REGS % RegBlock == 0);
static_assert(std::is_trivially_copyable_v<Maxwell3D::State::ShaderStageInfo>);
static_assert(std::is_trivially_copyable_v<Nx2FastDraw::Delta>);
static_assert(std::is_trivially_copyable_v<Maxwell3D::DirtyState::Flags>);

template <typename T>
void Append(std::vector<u32>& out, const T& value) {
    const size_t at = out.size();
    out.resize(at + Words(sizeof(T)));
    std::memcpy(out.data() + at, &value, sizeof(T));
}

template <typename T>
T Read(const u32*& in) {
    T value;
    std::memcpy(&value, in, sizeof(T));
    in += Words(sizeof(T));
    return value;
}

DrawState CaptureDraw(const Maxwell3D& engine) {
    const auto& draw = engine.draw_manager;
    const auto& state = draw.draw_state;
    DrawState captured;
    std::memset(&captured, 0, sizeof(captured));
    captured.topology = state.topology;
    captured.draw_mode = state.draw_mode;
    captured.draw_indexed = state.draw_indexed ? 1U : 0U;
    captured.draw_element_count = state.draw_element_count;
    captured.base_index = state.base_index;
    captured.vertex_buffer = state.vertex_buffer;
    captured.index_buffer = state.index_buffer;
    captured.base_instance = state.base_instance;
    captured.instance_count = state.instance_count;
    captured.drawn_instances = state.drawn_instances;
    captured.draw_texture_state = draw.draw_texture_state;
    const auto& indirect = draw.indirect_state;
    captured.indirect_flags = (indirect.is_byte_count ? 1U : 0U) | (indirect.is_indexed ? 2U : 0U) |
                              (indirect.include_count ? 4U : 0U);
    captured.count_start_address = indirect.count_start_address;
    captured.indirect_start_address = indirect.indirect_start_address;
    captured.buffer_size = indirect.buffer_size;
    captured.max_draw_counts = indirect.max_draw_counts;
    captured.stride = indirect.stride;
    return captured;
}

void RestoreDraw(Maxwell3D& mirror, const DrawState& captured) {
    auto& draw = mirror.draw_manager;
    auto& state = draw.draw_state;
    state.topology = captured.topology;
    state.draw_mode = captured.draw_mode;
    state.draw_indexed = captured.draw_indexed != 0;
    state.draw_element_count = captured.draw_element_count;
    state.base_index = captured.base_index;
    state.vertex_buffer = captured.vertex_buffer;
    state.index_buffer = captured.index_buffer;
    state.base_instance = captured.base_instance;
    state.instance_count = captured.instance_count;
    state.drawn_instances = captured.drawn_instances;
    draw.draw_texture_state = captured.draw_texture_state;
    auto& indirect = draw.indirect_state;
    indirect.is_byte_count = (captured.indirect_flags & 1) != 0;
    indirect.is_indexed = (captured.indirect_flags & 2) != 0;
    indirect.include_count = (captured.indirect_flags & 4) != 0;
    indirect.count_start_address = captured.count_start_address;
    indirect.indirect_start_address = captured.indirect_start_address;
    indirect.buffer_size = static_cast<size_t>(captured.buffer_size);
    indirect.max_draw_counts = static_cast<size_t>(captured.max_draw_counts);
    indirect.stride = static_cast<size_t>(captured.stride);
}

[[nodiscard]] bool DeltaEmpty(const Nx2FastDraw::Delta& delta) {
    return delta.Empty() && delta.num_device_writes == 0 && !delta.no_uniform_keeps;
}

} // Anonymous namespace

bool PrepLaneEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PREP_LANE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

Maxwell3DLane::Maxwell3DLane(Maxwell3D& engine_, Maxwell3D& mirror_)
    : engine{engine_}, mirror{mirror_}, sent_regs{std::make_unique<Maxwell3D::Regs>()} {
    // Both start at the register defaults, every dirty flag set.
    *sent_regs = engine.regs;
    mirror.regs = engine.regs;
    sent_state = engine.state;
    mirror.state = engine.state;
    const char* value = std::getenv("EDEN_NX2_PREP_ORDER_VERIFY");
    verify = value != nullptr && value[0] == '1';
    engine.runs_on_method_lane = true;
    engine.lane_flush = [](Tegra::Engines::EngineInterface* flushed) {
        static_cast<Maxwell3D*>(flushed)->Lane()->Flush();
    };
}

Maxwell3DLane::~Maxwell3DLane() = default;

void Maxwell3DLane::Build(Reads reads) {
    record.clear();
    // Registers, by runs of changed words within blocks.
    const u32* const now = engine.regs.reg_array.data();
    u32* const sent = sent_regs->reg_array.data();
    if (std::memcmp(now, sent, sizeof(Maxwell3D::Regs)) != 0) {
        for (size_t block = 0; block < Maxwell3D::Regs::NUM_REGS; block += RegBlock) {
            if (std::memcmp(now + block, sent + block, RegBlock * sizeof(u32)) == 0) {
                continue;
            }
            for (size_t first = block; first < block + RegBlock;) {
                if (now[first] == sent[first]) {
                    ++first;
                    continue;
                }
                size_t last = first + 1;
                while (last < block + RegBlock && now[last] != sent[last]) {
                    ++last;
                }
                const size_t count = last - first;
                record.push_back(TagRegs | static_cast<u32>(count << 8) |
                                 static_cast<u32>(first << 16));
                record.insert(record.end(), now + first, now + last);
                std::memcpy(sent + first, now + first, count * sizeof(u32));
                first = last;
            }
        }
    }
    // The flags the engine set since the last record: the rasterizer clears the mirror's.
    if (engine.dirty.flags.any()) {
        record.push_back(TagDirty);
        Append(record, engine.dirty.flags);
        engine.dirty.flags.reset();
    }
    for (u32 stage = 0; stage < Maxwell3D::Regs::MaxShaderStage; ++stage) {
        const auto& info = engine.state.shader_stages[stage];
        if (std::memcmp(&info, &sent_state.shader_stages[stage], sizeof(info)) != 0) {
            record.push_back(TagStage | (stage << 8));
            Append(record, info);
            sent_state.shader_stages[stage] = info;
        }
    }
    const auto& root = engine.root_tables;
    for (u32 table = 0; table < root.tables.size(); ++table) {
        if (root.tables[table] != sent_root_tables[table]) {
            record.push_back(TagRootTable | (table << 8));
            record.insert(record.end(), root.tables[table].begin(), root.tables[table].end());
            sent_root_tables[table] = root.tables[table];
        }
    }
    if (root.selected != sent_root_selected || root.offset != sent_root_offset ||
        root.loaded != sent_root_loaded) {
        record.push_back(TagRootState);
        record.push_back(root.selected);
        record.push_back(root.offset);
        record.push_back(root.loaded ? 1U : 0U);
        sent_root_selected = root.selected;
        sent_root_offset = root.offset;
        sent_root_loaded = root.loaded;
    }
    if (engine.engine_state != sent_engine_state) {
        record.push_back(TagEngineState);
        record.push_back(static_cast<u32>(engine.engine_state));
        sent_engine_state = engine.engine_state;
    }
    if (engine.replace_table != sent_replace_table) {
        record.push_back(TagReplaceTable);
        record.push_back(static_cast<u32>(engine.replace_table.size()));
        for (const auto& [key, value] : engine.replace_table) {
            record.push_back(static_cast<u32>(key));
            record.push_back(static_cast<u32>(key >> 32));
            record.push_back(static_cast<u32>(value));
        }
        sent_replace_table = engine.replace_table;
    }
    if (reads == ReadsDraw) {
        record.push_back(TagDraw);
        Append(record, CaptureDraw(engine));
        const auto& indexes = engine.draw_manager.draw_state.inline_index_draw_indexes;
        record.push_back(static_cast<u32>(indexes.size()));
        const size_t at = record.size();
        record.resize(at + Words(indexes.size()));
        if (!indexes.empty()) {
            std::memcpy(record.data() + at, indexes.data(), indexes.size());
        }
    }
    if (GpuSplit::OnMethodLane()) {
        // The fast draw changes this lane collected land where the draws are resolved.
        if (u32 last; Nx2FastDraw::TakePendingBreak(last)) {
            record.push_back(TagBreak);
            record.push_back(last);
        }
        if (!DeltaEmpty(Nx2FastDraw::g_delta)) {
            record.push_back(TagFastDelta);
            Append(record, Nx2FastDraw::g_delta);
            Nx2FastDraw::g_delta = {};
        }
    }
    if (verify && reads == ReadsDraw) {
        record.push_back(TagVerify);
        record.insert(record.end(), now, now + Maxwell3D::Regs::NUM_REGS);
        Append(record, engine.state);
        for (const auto& table : root.tables) {
            record.insert(record.end(), table.begin(), table.end());
        }
        Append(record, CaptureDraw(engine));
    }
}

void Maxwell3DLane::Apply(std::span<const u32> data) {
    const u32* in = data.data();
    const u32* const end = in + data.size();
    while (in < end) {
        const u32 head = *in++;
        switch (head & 0xff) {
        case TagRegs: {
            const u32 count = (head >> 8) & 0xff;
            const u32 first = head >> 16;
            std::memcpy(mirror.regs.reg_array.data() + first, in, count * sizeof(u32));
            in += count;
            break;
        }
        case TagDirty:
            mirror.dirty.flags |= Read<Maxwell3D::DirtyState::Flags>(in);
            break;
        case TagStage:
            mirror.state.shader_stages[(head >> 8) & 0xff] =
                Read<Maxwell3D::State::ShaderStageInfo>(in);
            break;
        case TagRootTable: {
            auto& table = mirror.root_tables.tables[(head >> 8) & 0xff];
            std::memcpy(table.data(), in, sizeof(table));
            in += table.size();
            break;
        }
        case TagRootState:
            mirror.root_tables.selected = in[0];
            mirror.root_tables.offset = in[1];
            mirror.root_tables.loaded = in[2] != 0;
            in += 3;
            break;
        case TagEngineState:
            mirror.engine_state = static_cast<Maxwell3D::EngineHint>(*in++);
            break;
        case TagReplaceTable: {
            const u32 count = *in++;
            mirror.replace_table.clear();
            for (u32 i = 0; i < count; ++i, in += 3) {
                mirror.replace_table.emplace(
                    u64{in[0]} | (u64{in[1]} << 32),
                    static_cast<Maxwell3D::HLEReplacementAttributeType>(in[2]));
            }
            break;
        }
        case TagDraw: {
            RestoreDraw(mirror, Read<DrawState>(in));
            const u32 bytes = *in++;
            auto& indexes = mirror.draw_manager.draw_state.inline_index_draw_indexes;
            indexes.resize(bytes);
            if (bytes != 0) {
                std::memcpy(indexes.data(), in, bytes);
            }
            in += Words(bytes);
            break;
        }
        case TagBreak:
            Nx2FastDraw::BreakAs(*in++);
            break;
        case TagFastDelta:
            Nx2FastDraw::MergeDelta(Read<Nx2FastDraw::Delta>(in));
            break;
        case TagVerify: {
            ++verify_checks;
            const u32* const regs = in;
            in += Maxwell3D::Regs::NUM_REGS;
            const auto state = Read<Maxwell3D::State>(in);
            const u32* const tables = in;
            in += mirror.root_tables.tables.size() * 64;
            const auto draw = Read<DrawState>(in);
            // The rasterizer's own change (UpdateDynamicStates' AMD logic op workaround).
            constexpr u32 LogicOpEnable = MAXWELL3D_REG_INDEX(logic_op.enable);
            std::string difference;
            for (u32 reg = 0; reg < Maxwell3D::Regs::NUM_REGS && difference.empty(); ++reg) {
                if (reg != LogicOpEnable && mirror.regs.reg_array[reg] != regs[reg]) {
                    difference = fmt::format("register {:#x}: mirror {:#x}, engine {:#x}", reg,
                                             mirror.regs.reg_array[reg], regs[reg]);
                }
            }
            if (difference.empty() &&
                std::memcmp(&mirror.state, &state, sizeof(state)) != 0) {
                difference = "constant buffer bindings";
            }
            for (size_t table = 0; table < mirror.root_tables.tables.size() && difference.empty();
                 ++table) {
                if (std::memcmp(mirror.root_tables.tables[table].data(), tables + table * 64,
                                64 * sizeof(u32)) != 0) {
                    difference = fmt::format("root table {}", table);
                }
            }
            if (difference.empty()) {
                const DrawState mirrored = CaptureDraw(mirror);
                if (std::memcmp(&mirrored, &draw, sizeof(draw)) != 0) {
                    difference = "draw state";
                }
            }
            if (!difference.empty()) {
                ++verify_differences;
                if (verify_differences <= 16) {
                    LOG_ERROR(HW_GPU, "Prep lane verify: mirror differs from its engine: {}",
                              difference);
                }
            }
            if ((verify_checks & 0xffff) == 0) {
                LOG_INFO(HW_GPU, "Prep lane verify: {} draws checked, {} mirrors differed",
                         verify_checks, verify_differences);
            }
            break;
        }
        default:
            LOG_CRITICAL(HW_GPU, "Prep lane: bad record word {:#x}", head);
            return;
        }
    }
}

void Maxwell3DLane::SyncMirror(Reads reads) {
    if (GpuSplit::OnMethodLane()) {
        Build(reads);
        if (!record.empty()) {
            GpuSplit::Get().LaneCall([this](std::span<const u32> data) { Apply(data); }, record);
        }
        return;
    }
    // Elsewhere the engine is only still while nothing runs it: the split stopped, or the method
    // lane waits for the preparation lane to run this. A call that reaches the rasterizer from
    // another thread while the method lane runs (the "any thread" ones) reads nothing here.
    const auto& split = GpuSplit::Get();
    if (split.BackRunning() && split.Lane() &&
        !(GpuSplit::OnPrepThread() && split.MethodLaneWaiting())) {
        return;
    }
    Build(reads);
    if (!record.empty()) {
        Apply(record);
    }
}

void Maxwell3DLane::Flush() {
    SyncMirror(ReadsState);
}

template <typename Func>
void Maxwell3DLane::Send(Reads reads, const Func& func) {
    if (reads != ReadsNothing) {
        SyncMirror(reads);
    }
    if (GpuSplit::OnMethodLane()) {
        GpuSplit::Get().LaneCall([func](std::span<const u32>) { func(); });
    } else {
        func();
    }
}

template <typename Func>
void Maxwell3DLane::SendWords(Reads reads, std::span<const u32> data, const Func& func) {
    if (reads != ReadsNothing) {
        SyncMirror(reads);
    }
    if (GpuSplit::OnMethodLane()) {
        GpuSplit::Get().LaneCall(func, data);
    } else {
        func(data);
    }
}

template <typename Func>
void Maxwell3DLane::Wait(const Func& func) {
    SyncMirror(ReadsDraw);
    if (GpuSplit::OnMethodLane()) {
        GpuSplit::Get().LaneRunSync(func);
    } else {
        func();
    }
}

void Maxwell3DLane::WriteConstantWords(GPUVAddr address, std::span<const u32> words) {
    SendWords(ReadsNothing, words, [memory = &engine.GetMemoryManager(), address](
                                       std::span<const u32> data) {
        memory->WriteWordsCached(address, data.data(), static_cast<u32>(data.size()));
    });
}

void Maxwell3DLane::WriteConstantBlock(GPUVAddr address, std::span<const u32> words) {
    SendWords(ReadsNothing, words, [memory = &engine.GetMemoryManager(), address](
                                       std::span<const u32> data) {
        memory->WriteBlockCached(address, data.data(), data.size_bytes());
    });
}

void Maxwell3DLane::QueryCondition() {
    Send(ReadsState, [mirror = &mirror] { mirror->LaneQueryCondition(); });
}

void Maxwell3DLane::UploadExec(bool is_linear) {
    Send(ReadsState, [mirror = &mirror, is_linear] { mirror->LaneUpload().ProcessExec(is_linear); });
}

void Maxwell3DLane::UploadData(u32 data, bool is_last_call) {
    Send(ReadsState, [mirror = &mirror, data, is_last_call] {
        mirror->LaneUpload().ProcessData(data, is_last_call);
    });
}

void Maxwell3DLane::UploadData(std::span<const u32> data) {
    SendWords(ReadsState, data, [mirror = &mirror](std::span<const u32> words) {
        mirror->LaneUpload().ProcessData(words.data(), words.size());
    });
}

void Maxwell3DLane::Draw(bool is_indexed, u32 instance_count) {
    Send(ReadsDraw, [this, is_indexed, instance_count] {
        // The render enable condition is the mirror's (QueryCondition).
        if (mirror.ShouldExecute()) {
            rasterizer->Draw(is_indexed, instance_count);
        }
    });
}

void Maxwell3DLane::DrawIndirect() {
    Send(ReadsDraw, [this] {
        if (mirror.ShouldExecute()) {
            rasterizer->DrawIndirect();
        }
    });
}

void Maxwell3DLane::DrawTexture() {
    Send(ReadsDraw, [this] { rasterizer->DrawTexture(); });
}

void Maxwell3DLane::Clear(u32 layer_count) {
    Send(ReadsDraw, [this, layer_count] {
        if (mirror.ShouldExecute()) {
            rasterizer->Clear(layer_count);
        }
    });
}

void Maxwell3DLane::DispatchCompute() {
    Wait([this] { rasterizer->DispatchCompute(); });
}

void Maxwell3DLane::ResetCounter(VideoCommon::QueryType type) {
    Send(ReadsState, [this, type] { rasterizer->ResetCounter(type); });
}

void Maxwell3DLane::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                          VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    // The query cache reads the topology and transform feedback state of the draws.
    Send(ReadsDraw, [this, gpu_addr, type, flags, payload, subreport] {
        rasterizer->Query(gpu_addr, type, flags, payload, subreport);
    });
}

void Maxwell3DLane::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                              u32 size) {
    Send(ReadsNothing, [this, stage, index, gpu_addr, size] {
        rasterizer->BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
    });
}

void Maxwell3DLane::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    Send(ReadsNothing,
         [this, stage, index] { rasterizer->DisableGraphicsUniformBuffer(stage, index); });
}

void Maxwell3DLane::SignalFence(std::function<void()>&& func) {
    Wait([this, &func] { rasterizer->SignalFence(std::move(func)); });
}

void Maxwell3DLane::SyncOperation(std::function<void()>&& func) {
    Wait([this, &func] { rasterizer->SyncOperation(std::move(func)); });
}

void Maxwell3DLane::SignalSyncPoint(u32 value) {
    Send(ReadsState, [this, value] { rasterizer->SignalSyncPoint(value); });
}

void Maxwell3DLane::SignalReference() {
    Send(ReadsState, [this] { rasterizer->SignalReference(); });
}

void Maxwell3DLane::ReleaseFences(bool force) {
    Wait([this, force] { rasterizer->ReleaseFences(force); });
}

void Maxwell3DLane::FlushQueryResults(DAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->FlushQueryResults(addr, size); });
}

void Maxwell3DLane::FlushAll() {
    Wait([this] { rasterizer->FlushAll(); });
}

void Maxwell3DLane::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Wait([this, addr, size, which] { rasterizer->FlushRegion(addr, size, which); });
}

void Maxwell3DLane::FlushRegionForRead(DAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->FlushRegionForRead(addr, size); });
}

void Maxwell3DLane::FinishRead(DAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->FinishRead(addr, size); });
}

bool Maxwell3DLane::NoteReportPending(DAddr addr, u64 size) {
    bool result{};
    Wait([this, addr, size, &result] { result = rasterizer->NoteReportPending(addr, size); });
    return result;
}

void Maxwell3DLane::FinishReport(DAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->FinishReport(addr, size); });
}

VideoCore::RasterizerInterface::GpuWrittenList Maxwell3DLane::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync, u64* pending_tick) {
    GpuWrittenList result{};
    Wait([&] { result = rasterizer->ReadGpuWrittenCommandList(addr, out, past_sync, pending_tick); });
    return result;
}

bool Maxwell3DLane::IsGpuTickDone(u64 tick) {
    bool result{};
    Wait([this, tick, &result] { result = rasterizer->IsGpuTickDone(tick); });
    return result;
}

void Maxwell3DLane::NoteGpuWrittenCommandListSynced(DAddr page) {
    Wait([this, page] { rasterizer->NoteGpuWrittenCommandListSynced(page); });
}

bool Maxwell3DLane::Nx2ListSlotsSupported() const {
    return rasterizer->Nx2ListSlotsSupported();
}

std::optional<u32> Maxwell3DLane::Nx2AcquireListSlot() {
    return rasterizer->Nx2AcquireListSlot(); // Any thread
}

void Maxwell3DLane::Nx2ReleaseListSlot(u32 slot) {
    rasterizer->Nx2ReleaseListSlot(slot); // Any thread
}

void Maxwell3DLane::Nx2SnapshotListWords(DAddr device_addr, u32 count, u32 slot) {
    Wait([this, device_addr, count, slot] {
        rasterizer->Nx2SnapshotListWords(device_addr, count, slot);
    });
}

void Maxwell3DLane::Nx2NoteOwedListWord(DAddr device_addr, u32 value) {
    Wait([this, device_addr, value] { rasterizer->Nx2NoteOwedListWord(device_addr, value); });
}

void Maxwell3DLane::Nx2NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) {
    Wait([this, device_addr, x, y, z] { rasterizer->Nx2NoteInlineQmdGrid(device_addr, x, y, z); });
}

void Maxwell3DLane::Nx2ReadListSlots(std::span<const u32> slots, std::span<u32> words) {
    Wait([this, slots, words] { rasterizer->Nx2ReadListSlots(slots, words); });
}

void Maxwell3DLane::Nx2CheckListSlot(u32 slot, std::span<const u32> words) {
    Wait([this, slot, words] { rasterizer->Nx2CheckListSlot(slot, words); });
}

void Maxwell3DLane::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    Wait([this, pages] { rasterizer->FlushGpuWrittenCommandListPages(pages); });
}

bool Maxwell3DLane::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    bool result{};
    Wait([this, addr, size, which, &result] {
        result = rasterizer->MustFlushRegion(addr, size, which);
    });
    return result;
}

VideoCore::RasterizerDownloadArea Maxwell3DLane::GetFlushArea(DAddr addr, u64 size) {
    VideoCore::RasterizerDownloadArea result{};
    Wait([this, addr, size, &result] { result = rasterizer->GetFlushArea(addr, size); });
    return result;
}

bool Maxwell3DLane::HasPendingQueries(DAddr addr, u64 size) {
    bool result{};
    Wait([this, addr, size, &result] { result = rasterizer->HasPendingQueries(addr, size); });
    return result;
}

bool Maxwell3DLane::HoldsQueries(DAddr addr, u64 size) {
    return rasterizer->HoldsQueries(addr, size); // Any thread
}

void Maxwell3DLane::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Wait([this, addr, size, which] { rasterizer->InvalidateRegion(addr, size, which); });
}

void Maxwell3DLane::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Wait([this, addr, size, which] { rasterizer->PrepareRegionWrite(addr, size, which); });
}

void Maxwell3DLane::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    Wait([this, sequences] { rasterizer->InnerInvalidation(sequences); });
}

void Maxwell3DLane::OnCacheInvalidation(PAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->OnCacheInvalidation(addr, size); });
}

void Maxwell3DLane::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    Wait([this, ranges] { rasterizer->OnCacheInvalidations(ranges); });
}

bool Maxwell3DLane::TryCacheInvalidation(DAddr addr, u64 size) {
    bool result{};
    Wait([this, addr, size, &result] { result = rasterizer->TryCacheInvalidation(addr, size); });
    return result;
}

bool Maxwell3DLane::OnCPUWrite(PAddr addr, u64 size) {
    bool result{};
    Wait([this, addr, size, &result] { result = rasterizer->OnCPUWrite(addr, size); });
    return result;
}

void Maxwell3DLane::InvalidateGPUCache() {
    Send(ReadsState, [this] { rasterizer->InvalidateGPUCache(); });
}

void Maxwell3DLane::UnmapMemory(DAddr addr, u64 size) {
    Wait([this, addr, size] { rasterizer->UnmapMemory(addr, size); });
}

void Maxwell3DLane::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    Wait([this, as_id, addr, size] { rasterizer->ModifyGPUMemory(as_id, addr, size); });
}

void Maxwell3DLane::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                             VideoCommon::CacheType which) {
    Wait([this, addr, size, which] { rasterizer->FlushAndInvalidateRegion(addr, size, which); });
}

void Maxwell3DLane::WaitForIdle() {
    Send(ReadsState, [this] { rasterizer->WaitForIdle(); });
}

void Maxwell3DLane::FragmentBarrier() {
    Send(ReadsState, [this] { rasterizer->FragmentBarrier(); });
}

void Maxwell3DLane::TiledCacheBarrier() {
    Send(ReadsState, [this] { rasterizer->TiledCacheBarrier(); });
}

void Maxwell3DLane::FlushCommands() {
    Wait([this] { rasterizer->FlushCommands(); });
}

void Maxwell3DLane::TickFrame() {
    Wait([this] { rasterizer->TickFrame(); });
}

bool Maxwell3DLane::AccelerateConditionalRendering() {
    bool result{};
    Wait([this, &result] { result = rasterizer->AccelerateConditionalRendering(); });
    return result;
}

bool Maxwell3DLane::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                          const Tegra::Engines::Fermi2D::Surface& dst,
                                          const Tegra::Engines::Fermi2D::Config& copy_config) {
    bool result{};
    Wait([&] { result = rasterizer->AccelerateSurfaceCopy(src, dst, copy_config); });
    return result;
}

bool Maxwell3DLane::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history) {
    bool result{};
    Wait([&] {
        result = rasterizer->AccelerateTemporalUpscale(src, dst, motion_vectors, depth, jitter,
                                                       motion_scale, use_history);
    });
    return result;
}

bool Maxwell3DLane::SupportsGameDlssNetwork() const {
    return rasterizer->SupportsGameDlssNetwork();
}

bool Maxwell3DLane::UsesNvidiaDlss() const {
    return rasterizer->UsesNvidiaDlss();
}

bool Maxwell3DLane::CanProbeGameDlssOutput() const {
    return rasterizer->CanProbeGameDlssOutput();
}

std::optional<VideoCore::GameDlssProbe> Maxwell3DLane::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst) {
    std::optional<VideoCore::GameDlssProbe> result;
    Wait([&] { result = rasterizer->ProbeGameDlssOutput(src, dst); });
    return result;
}

VideoCommon::OotNetwork::BypassState* Maxwell3DLane::OotNetworkBypass() {
    return rasterizer->OotNetworkBypass();
}

Tegra::Engines::AccelerateDMAInterface& Maxwell3DLane::AccessAccelerateDMA() {
    return rasterizer->AccessAccelerateDMA();
}

void Maxwell3DLane::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                             std::span<const u8> memory) {
    if (!GpuSplit::OnMethodLane()) {
        SyncMirror(ReadsState);
        rasterizer->AccelerateInlineToMemory(address, copy_size, memory);
        return;
    }
    std::vector<u32> words(Words(memory.size()));
    if (!memory.empty()) {
        std::memcpy(words.data(), memory.data(), memory.size());
    }
    const size_t bytes = memory.size();
    SendWords(ReadsState, words, [this, address, copy_size, bytes](std::span<const u32> data) {
        rasterizer->AccelerateInlineToMemory(
            address, copy_size,
            std::span<const u8>{reinterpret_cast<const u8*>(data.data()), bytes});
    });
}

void Maxwell3DLane::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    rasterizer->LoadDiskResources(title_id, stop_loading, callback);
}

void Maxwell3DLane::InitializeChannel(Tegra::Control::ChannelState& channel) {
    Wait([this, &channel] { rasterizer->InitializeChannel(channel); });
}

void Maxwell3DLane::BindChannel(Tegra::Control::ChannelState& channel) {
    Wait([this, &channel] { rasterizer->BindChannel(channel); });
}

void Maxwell3DLane::ReleaseChannel(s32 channel_id) {
    Wait([this, channel_id] { rasterizer->ReleaseChannel(channel_id); });
}

void Maxwell3DLane::RegisterTransformFeedback(GPUVAddr tfb_object_addr) {
    Send(ReadsState, [this, tfb_object_addr] {
        rasterizer->RegisterTransformFeedback(tfb_object_addr);
    });
}

bool Maxwell3DLane::HasDrawTransformFeedback() {
    return rasterizer->HasDrawTransformFeedback(); // What the device supports
}

} // namespace VideoCommon
