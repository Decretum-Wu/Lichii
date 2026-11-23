// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <sstream>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
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

TEST_CASE("Recorded local layouts round trip and old incomplete layouts request recapture",
          "[video_core][native_local_memory]") {
    LocalEnvironment env{16, 656};
    std::ostringstream out{std::ios::binary};
    env.Serialize(out, 783);
    const auto read = [](const std::string& bytes) {
        std::istringstream input{bytes, std::ios::binary};
        VideoCommon::FileEnvironment result;
        result.Deserialize(input, static_cast<std::streamoff>(bytes.size()));
        return result;
    };
    auto recorded = read(out.str());
    REQUIRE(recorded.LocalMemorySize() == 672);
    REQUIRE(recorded.LocalMemoryHighSize() == 656);
    REQUIRE_FALSE(recorded.LacksRecordedValue(783));

    auto old = out.str();
    const u64 key = 0xffffffff00000003ULL;
    const std::string key_bytes{reinterpret_cast<const char*>(&key), sizeof(key)};
    const auto position = old.find(key_bytes);
    REQUIRE(position != std::string::npos);
    old.erase(position, 12);
    u64 count;
    std::memcpy(&count, old.data() + 24, sizeof(count));
    --count;
    std::memcpy(old.data() + 24, &count, sizeof(count));
    auto incomplete = read(old);
    REQUIRE_THROWS_AS(incomplete.LocalMemoryHighSize(), Shader::LogicError);
    REQUIRE(incomplete.LacksRecordedValue(783));
    REQUIRE(incomplete.LacksRecordedValue(0));
}
