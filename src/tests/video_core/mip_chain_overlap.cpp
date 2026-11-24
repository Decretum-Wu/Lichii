// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/util.h"

namespace {

using namespace VideoCommon;

constexpr GPUVAddr Address = 0x100000;
constexpr DAddr CpuAddress = 0x200000;

ImageInfo MakeImage(Extent3D size, s32 levels, Extent3D block, s32 layers = 1) {
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = PixelFormat::R16G16B16A16_FLOAT;
    info.size = size;
    info.resources = {.levels = levels, .layers = layers};
    info.block = block;
    info.layer_stride = CalculateLayerStride(info);
    info.maybe_unaligned_layer_stride = CalculateLayerSize(info);
    return info;
}

} // namespace

TEST_CASE("A bloom target and its mip chain share level zero", "[video_core][texture_cache]") {
    const auto target = MakeImage({640, 360, 1}, 1, {0, 4, 0});
    const auto chain = MakeImage(target.size, 5, target.block);
    REQUIRE(chain.layer_stride != target.layer_stride);
    REQUIRE(chain.maybe_unaligned_layer_stride != target.maybe_unaligned_layer_stride);
    CHECK(IsLayerStrideCompatible(chain, target));
    CHECK(IsLayerStrideCompatible(target, chain));

    const ImageBase target_image{target, Address, CpuAddress};
    const auto joined = ResolveOverlap(chain, Address, CpuAddress, target_image, true, false, false);
    REQUIRE(joined);
    CHECK(joined->resources.levels == 5);
    const ImageBase chain_image{chain, Address, CpuAddress};
    const auto base = FindSubresource(target, chain_image, Address, {}, false, false);
    REQUIRE(base);
    CHECK(base->level == 0);

    CHECK_FALSE(IsLayerStrideCompatible(chain, MakeImage(target.size, 1, {0, 3, 0})));
    CHECK_FALSE(IsLayerStrideCompatible(target, MakeImage(target.size, 5, target.block, 2)));
    auto maxwell = target;
    maxwell.maxwell_gob_layout = true;
    CHECK_FALSE(IsLayerStrideCompatible(chain, maxwell));
}

TEST_CASE("An interior mip is reused as a view without replacing its parent",
          "[video_core][texture_cache]") {
    const auto chain = MakeImage({64, 64, 1}, 2, {});
    const ImageBase chain_image{chain, Address, CpuAddress};
    const u32 mip1 = chain_image.mip_level_offsets[1];
    REQUIRE(mip1 != 0);

    const auto matching_mip = MakeImage({32, 32, 1}, 1, {});
    const auto view = FindSubresource(matching_mip, chain_image, Address + mip1, {}, false, false);
    REQUIRE(view);
    CHECK(view->level == 1);
    CHECK_FALSE(ResolveOverlap(matching_mip, Address + mip1, CpuAddress + mip1, chain_image, true,
                               false, false));

    auto wrong_mip = MakeImage({128, 128, 1}, 1, {});
    wrong_mip.layer_stride = 0;
    CHECK_FALSE(ResolveOverlap(wrong_mip, Address + mip1, CpuAddress + mip1, chain_image, true,
                               false, false));
    CHECK_FALSE(FindSubresource(wrong_mip, chain_image, Address + mip1, {}, false, false));
}

TEST_CASE("An interior chain cannot replace the parent with smaller base dimensions",
          "[video_core][texture_cache]") {
    const auto parent = MakeImage({4, 4, 1}, 2, {});
    const auto child = MakeImage({2, 2, 1}, 2, {});
    REQUIRE(parent.layer_stride == child.layer_stride);
    const ImageBase parent_image{parent, Address, CpuAddress};
    const u32 mip1 = parent_image.mip_level_offsets[1];
    REQUIRE(mip1 != 0);
    CHECK_FALSE(FindSubresource(child, parent_image, Address + mip1, {}, false, false));
    CHECK_FALSE(ResolveOverlap(child, Address + mip1, CpuAddress + mip1, parent_image, true,
                               false, false));
}
