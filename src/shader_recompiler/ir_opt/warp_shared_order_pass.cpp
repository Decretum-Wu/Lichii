// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <unordered_map>

#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
bool StoresShared(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::WriteSharedU8:
    case IR::Opcode::WriteSharedU16:
    case IR::Opcode::WriteSharedU32:
    case IR::Opcode::WriteSharedU64:
    case IR::Opcode::WriteSharedU128:
        return true;
    default:
        return std::string_view{IR::NameOf(opcode)}.starts_with("SharedAtomic");
    }
}

bool LoadsShared(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadSharedU8:
    case IR::Opcode::LoadSharedS8:
    case IR::Opcode::LoadSharedU16:
    case IR::Opcode::LoadSharedS16:
    case IR::Opcode::LoadSharedU32:
    case IR::Opcode::LoadSharedU64:
    case IR::Opcode::LoadSharedU128:
        return true;
    default:
        return false;
    }
}

bool OrdersShared(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::Barrier:
    case IR::Opcode::WorkgroupMemoryBarrier:
    case IR::Opcode::DeviceMemoryBarrier:
    case IR::Opcode::VotedBarrier:
    case IR::Opcode::WarpSharedMemoryBarrier:
        return true;
    default:
        return false;
    }
}

// Whether a shared store may still be unordered when control leaves the block.
bool Transfer(const IR::Block& block, bool pending) {
    for (const IR::Inst& inst : block) {
        const IR::Opcode opcode{inst.GetOpcode()};
        if (OrdersShared(opcode) || (pending && LoadsShared(opcode))) {
            pending = false;
        } else if (StoresShared(opcode)) {
            pending = true;
        }
    }
    return pending;
}
} // Anonymous namespace

// NVIDIA guest code reads words other threads of its warp stored in shared memory with no barrier
// in between (warp-synchronous scans and reductions): the warp runs in step. Vulkan orders such
// accesses only through a barrier. A shared load that a store can reach with no barrier on the way
// gets a subgroup memory barrier before it, and the program runs whole guest warps.
void WarpSharedOrderPass(IR::Program& program) {
    std::unordered_map<const IR::Block*, bool> pending_out;
    const auto pending_in = [&](const IR::Block& block) {
        for (const IR::Block* const pred : block.ImmPredecessors()) {
            const auto it{pending_out.find(pred)};
            if (it != pending_out.end() && it->second) {
                return true;
            }
        }
        return false;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (const IR::Block* const block : program.blocks) {
            const bool out{Transfer(*block, pending_in(*block))};
            auto [it, inserted]{pending_out.try_emplace(block, out)};
            if (inserted || it->second != out) {
                it->second = out;
                changed = true;
            }
        }
    }
    for (IR::Block* const block : program.blocks) {
        bool pending{pending_in(*block)};
        for (auto it = block->begin(); it != block->end(); ++it) {
            const IR::Opcode opcode{it->GetOpcode()};
            if (OrdersShared(opcode)) {
                pending = false;
            } else if (pending && LoadsShared(opcode)) {
                block->PrependNewInst(it, IR::Opcode::WarpSharedMemoryBarrier);
                program.info.orders_warp_shared_memory = true;
                pending = false;
            } else if (StoresShared(opcode)) {
                pending = true;
            }
        }
    }
}

} // namespace Shader::Optimization
