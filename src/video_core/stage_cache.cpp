// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <type_traits>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "video_core/stage_cache.h"

namespace VideoCommon {
namespace {

enum ReadKind : u8 {
    CbufValue,
    StoragePointer,
    IndirectStoragePointer,
    TextureType,
    TexturePixelFormat,
    TextureInteger,
    ViewportTransform,
    Replacement,
};

// Past this, the cache starts over (translations hold program metadata, emitted stages SPIR-V).
constexpr size_t MaxBytes = 256ULL << 20;

class Writer {
public:
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void Put(const T& value) {
        const auto* const begin = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    std::vector<u8> bytes;
};

u64 EncodeReplacement(const std::optional<Shader::ReplaceConstant>& value) {
    return value ? (u64{1} << 32) | static_cast<u32>(*value) : 0;
}

u64 SignatureHash(const std::vector<u8>& signature) {
    return Common::CityHash64(reinterpret_cast<const char*>(signature.data()), signature.size());
}

size_t EmittedBytes(const StageCache::Emitted& emitted) {
    return sizeof(StageCache::Emitted) + emitted.spirv.size() * sizeof(u32);
}

} // Anonymous namespace

StageCache::Recorder::Recorder(Shader::Environment& inner_) : inner{inner_} {
    sph = inner.SPH();
    gp_passthrough_mask = inner.GpPassthroughMask();
    mesh_registers = inner.MeshRegisters();
    mesh_primitive_header = inner.MeshPrimitiveHeader();
    stage = inner.ShaderStage();
    start_address = inner.StartAddress();
    is_proprietary_driver = inner.IsProprietaryDriver();
}

u64 StageCache::Recorder::Log(u8 kind, u32 a, u32 b, u32 c, u64 value) {
    // Only first reads: within one translation a repeated read gives the same answer.
    if (logged.insert({kind, a, b, c}).second) {
        reads.push_back({kind, a, b, c, value});
    }
    return value;
}

u64 StageCache::Recorder::ReadInstruction(u32 address) {
    return inner.ReadInstruction(address);
}

u32 StageCache::Recorder::TranslateProgramAddress(u32 address) const {
    return inner.TranslateProgramAddress(address);
}

bool StageCache::Recorder::IsNativeProgram() const {
    return inner.IsNativeProgram();
}

bool StageCache::Recorder::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    return Log(StoragePointer, index, offset, 0, inner.IsCbufStorageBufferPointer(index, offset)) != 0;
}

bool StageCache::Recorder::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) {
    return Log(IndirectStoragePointer, index, offset, indirect,
               inner.IsCbufIndirectStorageBufferPointer(index, offset, indirect)) != 0;
}

std::vector<std::pair<u32, u32>> StageCache::Recorder::NullStoragePointerSlots() const {
    return inner.NullStoragePointerSlots();
}

std::vector<u64> StageCache::Recorder::MappedStoragePointerKeys() const {
    return inner.MappedStoragePointerKeys();
}

u32 StageCache::Recorder::ReadCbufValue(u32 index, u32 offset) {
    return static_cast<u32>(Log(CbufValue, index, offset, 0, inner.ReadCbufValue(index, offset)));
}

Shader::TextureType StageCache::Recorder::ReadTextureType(u32 handle) {
    return static_cast<Shader::TextureType>(
        Log(TextureType, handle, 0, 0, static_cast<u64>(inner.ReadTextureType(handle))));
}

Shader::TexturePixelFormat StageCache::Recorder::ReadTexturePixelFormat(u32 handle) {
    return static_cast<Shader::TexturePixelFormat>(
        Log(TexturePixelFormat, handle, 0, 0, static_cast<u64>(inner.ReadTexturePixelFormat(handle))));
}

bool StageCache::Recorder::IsTexturePixelFormatInteger(u32 handle) {
    return Log(TextureInteger, handle, 0, 0, inner.IsTexturePixelFormatInteger(handle)) != 0;
}

u32 StageCache::Recorder::ReadViewportTransformState() {
    return static_cast<u32>(Log(ViewportTransform, 0, 0, 0, inner.ReadViewportTransformState()));
}

u32 StageCache::Recorder::TextureBoundBuffer() const {
    return inner.TextureBoundBuffer();
}

u32 StageCache::Recorder::LocalMemorySize() const {
    return inner.LocalMemorySize();
}

u32 StageCache::Recorder::SharedMemorySize() const {
    return inner.SharedMemorySize();
}

std::array<u32, 3> StageCache::Recorder::WorkgroupSize() const {
    return inner.WorkgroupSize();
}

bool StageCache::Recorder::HasHLEMacroState() const {
    return inner.HasHLEMacroState();
}

bool StageCache::Recorder::HasNativeInstructions() const {
    return inner.HasNativeInstructions();
}

std::optional<Shader::ReplaceConstant> StageCache::Recorder::GetReplaceConstBuffer(u32 bank,
                                                                                   u32 offset) {
    const auto value = inner.GetReplaceConstBuffer(bank, offset);
    Log(Replacement, bank, offset, 0, EncodeReplacement(value));
    return value;
}

void StageCache::Recorder::Dump(u64 pipeline_hash, u64 shader_hash) {
    inner.Dump(pipeline_hash, shader_hash);
}

std::vector<u8> StageCache::Signature(Shader::Environment& env, u64 unique_hash, size_t slot) {
    Writer writer;
    writer.Put(unique_hash);
    writer.Put(static_cast<u32>(slot));
    writer.Put(env.ShaderStage());
    writer.Put(env.StartAddress());
    writer.Put(env.IsProprietaryDriver());
    writer.Put(env.IsNativeProgram());
    writer.Put(env.HasNativeInstructions());
    writer.Put(env.SPH());
    writer.Put(env.GpPassthroughMask());
    writer.Put(env.MeshRegisters());
    writer.Put(env.MeshPrimitiveHeader());
    writer.Put(env.TextureBoundBuffer());
    writer.Put(env.LocalMemorySize());
    writer.Put(env.SharedMemorySize());
    writer.Put(env.WorkgroupSize());
    writer.Put(env.HasHLEMacroState());
    // What the recompiler reads from the settings, which can change during a session.
    const auto& resolution = Settings::values.resolution_info;
    writer.Put(resolution.up_scale);
    writer.Put(resolution.down_shift);
    writer.Put(resolution.up_factor);
    writer.Put(resolution.down_factor);
    writer.Put(resolution.active);
    writer.Put(Settings::values.renderer_debug.GetValue());
    writer.Put(Settings::values.disable_shader_loop_safety_checks.GetValue());
    writer.Put(Settings::values.fix_bloom_effects.GetValue());
    return std::move(writer.bytes);
}

std::shared_ptr<const StageCache::Translation> StageCache::Find(const std::vector<u8>& signature,
                                                                Shader::Environment& env) {
    std::vector<std::shared_ptr<const Translation>> candidates;
    {
        std::scoped_lock lock{mutex};
        const auto it = translations.find(SignatureHash(signature));
        if (it == translations.end()) {
            return nullptr;
        }
        candidates = it->second;
    }
    for (const auto& candidate : candidates) {
        if (candidate->signature != signature) {
            continue;
        }
        bool same = true;
        try {
            // In the order translation read them: up to the first difference, every read is one
            // translating here would make too.
            for (const Translation::Read& read : candidate->reads) {
                u64 value{};
                switch (read.kind) {
                case CbufValue:
                    value = env.ReadCbufValue(read.a, read.b);
                    break;
                case StoragePointer:
                    value = env.IsCbufStorageBufferPointer(read.a, read.b);
                    break;
                case IndirectStoragePointer:
                    value = env.IsCbufIndirectStorageBufferPointer(read.a, read.b, read.c);
                    break;
                case TextureType:
                    value = static_cast<u64>(env.ReadTextureType(read.a));
                    break;
                case TexturePixelFormat:
                    value = static_cast<u64>(env.ReadTexturePixelFormat(read.a));
                    break;
                case TextureInteger:
                    value = env.IsTexturePixelFormatInteger(read.a);
                    break;
                case ViewportTransform:
                    value = env.ReadViewportTransformState();
                    break;
                case Replacement:
                    value = EncodeReplacement(env.GetReplaceConstBuffer(read.a, read.b));
                    break;
                }
                if (value != read.value) {
                    same = false;
                    break;
                }
            }
        } catch (const Shader::Exception&) {
            // A disk cache environment without this read: translating it would throw here too.
            same = false;
        }
        if (same) {
            return candidate;
        }
    }
    return nullptr;
}

std::shared_ptr<const StageCache::Emitted> StageCache::FindEmitted(
    const Translation& translation, const Shader::RuntimeInfo& runtime_info,
    const Shader::Backend::Bindings& bindings_in) {
    std::scoped_lock lock{mutex};
    for (const auto& emitted : translation.emitted) {
        if (emitted->bindings_in == bindings_in && emitted->runtime_info == runtime_info) {
            return emitted;
        }
    }
    return nullptr;
}

void StageCache::Add(const std::shared_ptr<const Translation>& translation,
                     std::shared_ptr<const Emitted> emitted) {
    {
        std::scoped_lock lock{mutex};
        auto& candidates = translations[SignatureHash(translation->signature)];
        const Translation* target{};
        for (const auto& candidate : candidates) {
            if (candidate == translation || (candidate->signature == translation->signature &&
                                             candidate->reads == translation->reads)) {
                target = candidate.get();
                break;
            }
        }
        if (!target) {
            // New, or added again after the cache started over
            candidates.push_back(translation);
            target = translation.get();
            bytes += sizeof(Translation) + translation->signature.size() +
                     translation->reads.size() * sizeof(Translation::Read);
            for (const auto& existing : target->emitted) {
                bytes += EmittedBytes(*existing);
            }
            recorded_translations.fetch_add(1, std::memory_order_relaxed);
        }
        for (const auto& existing : target->emitted) {
            if (existing->bindings_in == emitted->bindings_in &&
                existing->runtime_info == emitted->runtime_info) {
                return;
            }
        }
        bytes += EmittedBytes(*emitted);
        target->emitted.push_back(std::move(emitted));
        recorded_spirv.fetch_add(1, std::memory_order_relaxed);
        if (bytes > MaxBytes) {
            // Translations in use elsewhere stay alive through their shared pointers.
            translations.clear();
            bytes = 0;
        }
    }
    Log();
}

Shader::IR::Program StageCache::Summary(const Shader::IR::Program& program) {
    Shader::IR::Program summary{program};
    summary.syntax_list.clear();
    summary.blocks.clear();
    summary.post_order_blocks.clear();
    return summary;
}

bool StageCache::SameSummary(const Shader::IR::Program& a, const Shader::IR::Program& b) {
    return a.info == b.info && a.stage == b.stage && a.workgroup_size == b.workgroup_size &&
           a.output_topology == b.output_topology && a.output_vertices == b.output_vertices &&
           a.invocations == b.invocations && a.local_memory_size == b.local_memory_size &&
           a.shared_memory_size == b.shared_memory_size &&
           a.is_geometry_passthrough == b.is_geometry_passthrough && a.mesh == b.mesh;
}

void StageCache::CountReusedPipeline(size_t stages) {
    reused_pipelines.fetch_add(1, std::memory_order_relaxed);
    reused_stages.fetch_add(stages, std::memory_order_relaxed);
    Log();
}

void StageCache::CountVerifyMismatch(u64 unique_hash) {
    verify_mismatches.fetch_add(1, std::memory_order_relaxed);
    LOG_ERROR(Render_Vulkan, "Stage cache: reused stage {:016x} differs from a fresh build",
              unique_hash);
}

void StageCache::Log() {
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = last_log.load(std::memory_order_relaxed);
    if (now - last < 10 || !last_log.compare_exchange_strong(last, now)) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Stage cache: {} pipelines built from reused stages ({} stages), {} translations and "
             "{} SPIR-V recorded, {} verify mismatches",
             reused_pipelines.load(), reused_stages.load(), recorded_translations.load(),
             recorded_spirv.load(), verify_mismatches.load());
}

} // namespace VideoCommon
