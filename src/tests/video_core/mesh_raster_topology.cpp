// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"

namespace {
using Regs = Tegra::Engines::Maxwell3D::Regs;
using Topology = Regs::PrimitiveTopology;

TEST_CASE("Mesh draws rasterize SET_MESH_SHADER_A's topology", "[video_core]") {
    const auto regs = std::make_unique<Regs>();
    REQUIRE(regs->RasterTopology(Topology::Points) == Topology::Points);
    REQUIRE(regs->RasterTopology(Topology::TriangleStrip) == Topology::TriangleStrip);

    regs->reg_array[Regs::MeshControl] = 1;
    regs->reg_array[Regs::MeshShaderA] = (64u << 13) | (64u << 4) | 4;
    REQUIRE(regs->RasterTopology(Topology::Points) == Topology::Triangles);
    regs->reg_array[Regs::MeshShaderA] = (64u << 4) | 1;
    REQUIRE(regs->RasterTopology(Topology::Points) == Topology::Lines);
    regs->reg_array[Regs::MeshShaderA] = 0;
    REQUIRE(regs->RasterTopology(Topology::Triangles) == Topology::Points);
}

} // Anonymous namespace
