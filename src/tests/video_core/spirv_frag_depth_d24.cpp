// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"

namespace {
namespace IR = Shader::IR;

constexpr u32 OpExtInstImport = 11;
constexpr u32 OpExtInst = 12;
constexpr u32 GLSLstd450RoundEven = 2;

size_t CountRoundEven(const std::vector<u32>& code) {
    constexpr std::string_view glsl{"GLSL.std.450"};
    u32 glsl_set{};
    size_t count{};
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        REQUIRE(word_count > 0);
        REQUIRE(offset + word_count <= code.size());
        const u32 opcode = code[offset] & 0xffff;
        if (opcode == OpExtInstImport && word_count >= 3) {
            const auto* name = reinterpret_cast<const char*>(&code[offset + 2]);
            if (std::strncmp(name, glsl.data(), glsl.size()) == 0) {
                glsl_set = code[offset + 1];
            }
        } else if (opcode == OpExtInst && word_count >= 5 && code[offset + 3] == glsl_set &&
                   code[offset + 4] == GLSLstd450RoundEven) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

enum class Depth { Constant, FragCoordZ, MinFragCoordZ };

std::vector<u32> EmitFragDepth(bool quantize, Depth source) {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = Shader::Stage::Fragment;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    IR::IREmitter ir{block};
    switch (source) {
    case Depth::Constant:
        ir.SetFragDepth(ir.Imm32(0.25f));
        break;
    case Depth::FragCoordZ:
        ir.SetFragDepth(ir.GetAttribute(IR::Attribute::PositionZ));
        program.info.loads.Set(IR::Attribute::PositionZ, true);
        break;
    case Depth::MinFragCoordZ:
        ir.SetFragDepth(ir.FPMin(ir.GetAttribute(IR::Attribute::PositionZ), ir.Imm32(0.25f)));
        program.info.loads.Set(IR::Attribute::PositionZ, true);
        break;
    }
    program.info.stores_frag_depth = true;
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;
    Shader::RuntimeInfo runtime_info{};
    runtime_info.quantize_frag_depth_d24 = quantize;
    Shader::Backend::Bindings bindings;
    return Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}
} // Anonymous namespace

TEST_CASE("SPIR-V FragDepth is rounded to D24 only when the host emulates D24",
          "[video_core][spirv][frag_depth_d24]") {
    for (const Depth source : {Depth::Constant, Depth::FragCoordZ, Depth::MinFragCoordZ}) {
        CAPTURE(source);
        CHECK(CountRoundEven(EmitFragDepth(false, source)) == 0);
    }
    CHECK(CountRoundEven(EmitFragDepth(true, Depth::Constant)) == 1);
    CHECK(CountRoundEven(EmitFragDepth(true, Depth::MinFragCoordZ)) == 1);
}

TEST_CASE("SPIR-V FragDepth that stores FragCoord.z as it is stays unrounded",
          "[video_core][spirv][frag_depth_d24]") {
    CHECK(CountRoundEven(EmitFragDepth(true, Depth::FragCoordZ)) == 0);
}
