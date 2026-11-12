// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace VideoCommon {

/// Reuses the translated and emitted native shader stages of a pipeline for another pipeline.
///
/// Pipelines that differ only in fixed state (blending, depth, render targets) translate the same
/// programs again. A stage's IR depends only on its program, the environment's constant
/// properties, the recompiler settings and the answers to the reads translation makes (constant
/// buffer values, texture types and formats, storage pointer checks, ...); its SPIR-V also on the
/// runtime info and the bindings the earlier stages took. A translation matches another
/// environment when those reads, replayed in the order translation made them, give the same
/// answers: translation is deterministic, so every read replayed up to a difference is one it
/// would make too. Its SPIR-V matches when the runtime info and incoming bindings are also equal.
///
/// Thread safe: pipelines are built on the GPU thread, shader job workers and boot workers.
class StageCache {
public:
    /// A stage's SPIR-V for one runtime info and incoming bindings
    struct Emitted {
        Shader::RuntimeInfo runtime_info;
        Shader::Backend::Bindings bindings_in;
        std::vector<u32> spirv;
        Shader::IR::Program program; ///< After emission, without IR
        Shader::Backend::Bindings bindings_out;
    };

    struct Translation {
        struct Read {
            u8 kind;
            u32 a, b, c;
            u64 value;

            bool operator==(const Read&) const = default;
        };

        std::vector<u8> signature;
        std::vector<Read> reads;
        Shader::IR::Program program; ///< After translation, without IR
        /// Guarded by the cache's mutex
        mutable std::vector<std::shared_ptr<const Emitted>> emitted;
    };

    /// Records the reads translation makes through it, forwarding everything to `inner`.
    class Recorder final : public Shader::Environment {
    public:
        explicit Recorder(Shader::Environment& inner);

        u64 ReadInstruction(u32 address) override;
        u32 TranslateProgramAddress(u32 address) const override;
        bool IsNativeProgram() const override;
        bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;
        bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) override;
        std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;
        std::vector<u64> MappedStoragePointerKeys() const override;
        u32 ReadCbufValue(u32 index, u32 offset) override;
        Shader::TextureType ReadTextureType(u32 handle) override;
        Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;
        bool IsTexturePixelFormatInteger(u32 handle) override;
        u32 ReadViewportTransformState() override;
        u32 TextureBoundBuffer() const override;
        u32 LocalMemorySize() const override;
        u32 SharedMemorySize() const override;
        std::array<u32, 3> WorkgroupSize() const override;
        bool HasHLEMacroState() const override;
        bool HasNativeInstructions() const override;
        std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
        void Dump(u64 pipeline_hash, u64 shader_hash) override;

        [[nodiscard]] std::vector<Translation::Read> TakeReads() {
            return std::move(reads);
        }

    private:
        u64 Log(u8 kind, u32 a, u32 b, u32 c, u64 value);

        Shader::Environment& inner;
        std::vector<Translation::Read> reads;
        std::set<std::array<u32, 4>> logged;
    };

    /// The bytes translation depends on besides the reads: program, slot, environment constants and
    /// the settings the recompiler reads.
    [[nodiscard]] static std::vector<u8> Signature(Shader::Environment& env, u64 unique_hash,
                                                   size_t slot);

    /// A translation of `signature` whose reads `env` answers the same way, or null.
    [[nodiscard]] std::shared_ptr<const Translation> Find(const std::vector<u8>& signature,
                                                          Shader::Environment& env);

    /// The SPIR-V of `translation` for this runtime info and incoming bindings, or null.
    [[nodiscard]] std::shared_ptr<const Emitted> FindEmitted(
        const Translation& translation, const Shader::RuntimeInfo& runtime_info,
        const Shader::Backend::Bindings& bindings_in);

    /// Makes `emitted` findable as SPIR-V of `translation`, adding the translation unless the
    /// cache holds it or an equal one.
    void Add(const std::shared_ptr<const Translation>& translation,
             std::shared_ptr<const Emitted> emitted);

    /// Program metadata without its IR.
    [[nodiscard]] static Shader::IR::Program Summary(const Shader::IR::Program& program);

    /// Whether two programs have the same metadata (what Summary keeps).
    [[nodiscard]] static bool SameSummary(const Shader::IR::Program& a,
                                          const Shader::IR::Program& b);

    /// Counts for the periodic log line: a pipeline built from reused stages only, a reused stage
    /// EDEN_NX2_PERF_VERIFY compared with its fresh build and found different.
    void CountReusedPipeline(size_t stages);
    void CountVerifyMismatch(u64 unique_hash);

private:
    void Log();

    std::mutex mutex;
    std::unordered_map<u64, std::vector<std::shared_ptr<const Translation>>> translations;
    size_t bytes{};

    std::atomic<u64> reused_pipelines{};
    std::atomic<u64> reused_stages{};
    std::atomic<u64> recorded_translations{};
    std::atomic<u64> recorded_spirv{};
    std::atomic<u64> verify_mismatches{};
    std::atomic<s64> last_log{};
};

} // namespace VideoCommon
