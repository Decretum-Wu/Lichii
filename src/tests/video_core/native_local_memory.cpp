// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"

namespace {
namespace IR = Shader::IR;

class LocalEnvironment final : public VideoCommon::GenericEnvironment {
public:
    bool native{true};

    LocalEnvironment(u32 low, u32 high) {
        stage = Shader::Stage::Compute;
        local_memory_size = low + high;
        local_memory_high_size = high;
        code = {0, 0};
        cached_lowest = 0;
        cached_highest = 8;
        workgroup_size = {1, 1, 1};
    }
    bool IsNativeProgram() const override { return native; }
    u32 ReadCbufValue(u32, u32) override { throw Shader::LogicError("Unexpected cbuf read"); }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
};

struct LocalMachine {
    LocalEnvironment env;
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    std::array<u32, IR::NUM_REGS> registers{};
    std::vector<u32> memory;
    std::unordered_map<const IR::Inst*, u32> values;

    LocalMachine(u32 low, u32 high) : env{low, high}, memory((low + high) / 4) {}

    u32 Value(const IR::Value& value) {
        return value.IsImmediate() ? value.U32() : values.at(value.InstRecursive());
    }

    void Run() {
        for (const auto& inst : block) {
            const auto arg = [&](size_t i) { return Value(inst.Arg(i)); };
            u32 result{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetRegister:
                result = registers[static_cast<size_t>(inst.Arg(0).Reg())];
                break;
            case IR::Opcode::SetRegister:
                registers[static_cast<size_t>(inst.Arg(0).Reg())] = arg(1);
                break;
            case IR::Opcode::IAdd32: result = arg(0) + arg(1); break;
            case IR::Opcode::BitwiseAnd32: result = arg(0) & arg(1); break;
            case IR::Opcode::ShiftRightArithmetic32:
                result = static_cast<s32>(arg(0)) >> arg(1);
                break;
            case IR::Opcode::ULessThan: result = arg(0) < arg(1); break;
            case IR::Opcode::SelectU32: result = arg(0) ? arg(1) : arg(2); break;
            case IR::Opcode::LoadLocal:
                REQUIRE(arg(0) < memory.size());
                result = memory[arg(0)];
                break;
            case IR::Opcode::WriteLocal:
                REQUIRE(arg(0) < memory.size());
                memory[arg(0)] = arg(1);
                break;
            default: FAIL("Unexpected local memory IR opcode");
            }
            values.emplace(&inst, result);
        }
    }

    void StoreLoad(u32 address, u32 expected_word) {
        registers[1] = address;
        registers[2] = 0x1234abcd;
        visitor.STL(0xef54000000070102ULL);
        visitor.LDL(0xef44000000070103ULL);
        Run();
        CHECK(memory.at(expected_word) == 0x1234abcd);
        CHECK(registers[3] == 0x1234abcd);
    }
};
} // namespace

TEST_CASE("Native local memory maps the Fast Fusion stack inside its high allocation",
          "[video_core][native_local_memory]") {
    for (const u32 displacement : {0U, 76U}) {
        LocalMachine machine{0, 656};
        machine.StoreLoad(0xfffdc0 - 80 + displacement, displacement / 4);
    }
}

TEST_CASE("Native low and high local allocations occupy disjoint host words",
          "[video_core][native_local_memory]") {
    for (const auto& [address, word] : std::array<std::pair<u32, u32>, 4>{
             {{0xffffe0, 0}, {0xfffffff0, 4}, {0, 8}, {12, 11}}}) {
        LocalMachine machine{16, 32};
        machine.StoreLoad(address, word);
    }
    LocalMachine low_only{32, 0};
    low_only.StoreLoad(12, 3);
    LocalMachine maxwell{32, 0};
    maxwell.env.native = false;
    maxwell.StoreLoad(12, 3);
}

TEST_CASE("Native local memory without a high allocation translates as before",
          "[video_core][native_local_memory]") {
    const auto opcodes = [](bool native) {
        LocalMachine machine{32, 0};
        machine.env.native = native;
        machine.StoreLoad(12, 3);
        std::vector<IR::Opcode> result;
        for (const auto& inst : machine.block) {
            result.push_back(inst.GetOpcode());
        }
        return result;
    };
    CHECK(opcodes(true) == opcodes(false));
}

namespace {
std::string Record(u32 low, u32 high) {
    LocalEnvironment env{low, high};
    std::ostringstream out{std::ios::binary};
    env.Serialize(out, 783);
    return out.str();
}

/// The record as builds before the high allocation key wrote it.
std::string WithoutHighKey(std::string record) {
    const u64 key = 0xffffffff00000003ULL;
    const std::string key_bytes{reinterpret_cast<const char*>(&key), sizeof(key)};
    const auto position = record.find(key_bytes);
    REQUIRE(position != std::string::npos);
    record.erase(position, 12);
    u64 count;
    std::memcpy(&count, record.data() + 24, sizeof(count));
    --count;
    std::memcpy(record.data() + 24, &count, sizeof(count));
    return record;
}

VideoCommon::FileEnvironment Read(const std::string& bytes) {
    std::istringstream input{bytes, std::ios::binary};
    VideoCommon::FileEnvironment result;
    result.Deserialize(input, static_cast<std::streamoff>(bytes.size()));
    return result;
}
} // namespace

TEST_CASE("Recorded local layouts round trip and old incomplete layouts request recapture",
          "[video_core][native_local_memory]") {
    auto recorded = Read(Record(16, 656));
    REQUIRE(recorded.LocalMemorySize() == 672);
    REQUIRE(recorded.LocalMemoryHighSize() == 656);
    REQUIRE_FALSE(recorded.LacksRecordedValue(783));

    auto incomplete = Read(WithoutHighKey(Record(16, 656)));
    REQUIRE_THROWS_AS(incomplete.LocalMemoryHighSize(), Shader::LogicError);
    REQUIRE(incomplete.LacksRecordedValue(783));
    REQUIRE(incomplete.LacksRecordedValue(0));
}

TEST_CASE("One compute shader under two local layouts keeps two pipelines",
          "[video_core][native_local_memory]") {
    using Vulkan::ComputePipelineLookupKey;
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0xd037c025142cc326ULL, .shared_memory_size = 0, .workgroup_size{64, 1, 1}};
    const ComputePipelineLookupKey high_only{key, 592, 592};
    const ComputePipelineLookupKey low_only{key, 592, 0};
    std::unordered_map<ComputePipelineLookupKey, int> cache;
    cache.emplace(high_only, 1);
    cache.emplace(low_only, 2);
    CHECK(cache.size() == 2);
    // Any number of alternations finds each layout's own translation.
    for (int i = 0; i < 10; ++i) {
        CHECK(Vulkan::FindComputePipeline(cache, high_only)->second == 1);
        CHECK(Vulkan::FindComputePipeline(cache, low_only)->second == 2);
    }
    CHECK(Vulkan::FindComputePipeline(cache, ComputePipelineLookupKey{key, 672, 656}) ==
          cache.end());

    // A record's pipeline is found under the layout it was translated for.
    CHECK(Vulkan::RecordLookupKey(key, Read(Record(16, 656))) ==
          ComputePipelineLookupKey{key, 672, 656});
    CHECK(Vulkan::RecordLookupKey(key, Read(Record(32, 0))) ==
          ComputePipelineLookupKey{key, 32, 0});
    // Only Maxwell compute translates local accesses from a record without the high allocation.
    CHECK(Vulkan::RecordLookupKey(key, Read(WithoutHighKey(Record(32, 0)))) ==
          ComputePipelineLookupKey{key, 32, 0});
}

TEST_CASE("Compute pipelines without local accesses serve every local layout",
          "[video_core][native_local_memory]") {
    using Vulkan::ComputePipelineLookupKey;
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0x143a3470b3d02c14ULL, .shared_memory_size = 0, .workgroup_size{32, 1, 1}};
    std::unordered_map<ComputePipelineLookupKey, int> cache;
    cache.emplace(Vulkan::AnyLayoutKey(key), 1);
    // Xenoblade Chronicles 2 recorded it at 576 bytes and dispatches it at 592.
    CHECK(Vulkan::FindComputePipeline(cache, ComputePipelineLookupKey{key, 576, 576})->second == 1);
    CHECK(Vulkan::FindComputePipeline(cache, ComputePipelineLookupKey{key, 592, 592})->second == 1);
    // Another shader's key does not match.
    auto other{key};
    other.unique_hash ^= 1;
    CHECK(Vulkan::FindComputePipeline(cache, ComputePipelineLookupKey{other, 576, 576}) ==
          cache.end());
}

namespace {
/// The fields of a compute pipeline cache entry that publishing a rebuild reads and writes.
struct FakeEntry {
    std::unique_ptr<int> pipeline;
    bool uses_local_memory{};
    Vulkan::ComputePipelineLookupKey dispatch_key{};
    u32 rebuilds{};
    bool used{};
};

struct FakeCache {
    std::unordered_map<Vulkan::ComputePipelineLookupKey, FakeEntry> map;
    std::vector<int> retired;

    /// What the dispatch path finds for a layout, the translation's id or 0
    int Dispatch(const Vulkan::ComputePipelineLookupKey& lookup) {
        const auto it{Vulkan::FindComputePipeline(map, lookup)};
        return it != map.end() && it->second.pipeline ? *it->second.pipeline : 0;
    }

    /// Publishes `rebuilt` over the entry dispatch `lookup` finds, as CurrentComputePipeline does,
    /// and returns whether that entry still serves the dispatch.
    bool Publish(const Vulkan::ComputePipelineLookupKey& lookup, FakeEntry&& rebuilt) {
        auto it{Vulkan::FindComputePipeline(map, lookup)};
        REQUIRE(it != map.end());
        it = Vulkan::PublishComputeRebuild(map, it, std::move(rebuilt),
                                           [&](std::unique_ptr<int>&& pipeline) {
                                               retired.push_back(*pipeline);
                                           });
        return Vulkan::ServesLookup(it->first, lookup);
    }
};

FakeEntry Translation(int id, bool uses_local_memory,
                      const Vulkan::ComputePipelineLookupKey& dispatch_key) {
    return {.pipeline = std::make_unique<int>(id),
            .uses_local_memory = uses_local_memory,
            .dispatch_key = dispatch_key};
}
} // namespace

TEST_CASE("A rebuild with local accesses leaves the layout-free key for its own layout",
          "[video_core][native_local_memory]") {
    using Vulkan::ComputePipelineLookupKey;
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0xd037c025142cc326ULL, .shared_memory_size = 0, .workgroup_size{64, 1, 1}};
    const ComputePipelineLookupKey a{key, 592, 592};
    const ComputePipelineLookupKey b{key, 608, 608};
    FakeCache cache;
    // The first translation dropped its local accesses with a null pointer: it serves any layout.
    cache.map.emplace(Vulkan::AnyLayoutKey(key), Translation(1, false, a));
    REQUIRE(cache.Dispatch(a) == 1);
    REQUIRE(cache.Dispatch(b) == 1);

    SECTION("Immediate rebuild by the dispatch it was translated for") {
        CHECK(cache.Publish(a, Translation(2, true, a)));
        CHECK(cache.Dispatch(a) == 2);
        CHECK(cache.Dispatch(b) == 0); // B translates its own
        CHECK(cache.retired == std::vector<int>{1});
        CHECK(cache.map.at(a).rebuilds == 1);
    }
    SECTION("A queues the rebuild, B finds it built") {
        CHECK_FALSE(cache.Publish(b, Translation(2, true, a)));
        // B's dispatch looks up again: neither lookup path hands it A's translation.
        CHECK(cache.Dispatch(b) == 0);
        CHECK(cache.Dispatch(a) == 2);
        cache.map.emplace(b, Translation(3, true, b));
        CHECK(cache.Dispatch(b) == 3);
        CHECK(cache.Dispatch(a) == 2);
        CHECK(cache.retired == std::vector<int>{1});
    }
    SECTION("A rebuild without local accesses keeps serving every layout") {
        CHECK(cache.Publish(b, Translation(2, false, a)));
        CHECK(cache.Dispatch(a) == 2);
        CHECK(cache.Dispatch(b) == 2);
    }
    SECTION("A layout's own translation is replaced, not kept beside the rebuild") {
        cache.map.emplace(a, Translation(4, true, a));
        auto it{cache.map.find(Vulkan::AnyLayoutKey(key))};
        it = Vulkan::PublishComputeRebuild(cache.map, it, Translation(2, true, a),
                                           [&](std::unique_ptr<int>&& pipeline) {
                                               cache.retired.push_back(*pipeline);
                                           });
        CHECK(it->first == a);
        CHECK(cache.Dispatch(a) == 2);
        CHECK(cache.Dispatch(b) == 0);
        CHECK(cache.retired == std::vector<int>{1, 4});
    }
}
