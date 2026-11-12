// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/draw_template.h"

namespace {
using VideoCommon::DrawTemplateCheck;
using VideoCommon::DrawTemplateContext;
using Template = VideoCommon::DrawTemplate<u64, 5, 2>;
using Table = VideoCommon::DrawTemplateTable<Template>;

DrawTemplateContext Context(u64 owner) {
    return {
        .owner = owner,
        .memory = 0x1000,
        .image_table = 0x2000,
        .tic_address = 0x40000,
        .tsc_address = 0x50000,
        .tic_limit = 127,
        .tsc_limit = 63,
        .via_header_index = 0,
        .native_tiling = 1,
    };
}

Template* Find(Table& table, const DrawTemplateContext& context, const std::vector<u32>& handles) {
    return table.Find(context, handles, VideoCommon::HashDrawTemplate(context, handles));
}

Template& Insert(Table& table, const DrawTemplateContext& context,
                 const std::vector<u32>& handles) {
    return table.Insert(context, handles, VideoCommon::HashDrawTemplate(context, handles));
}
} // namespace

TEST_CASE("Draw template: only the same pipeline, tables and handle words find a template",
          "[video_core]") {
    Table table{4};
    const DrawTemplateContext context = Context(table.NewOwner());
    const std::vector<u32> handles{0x10002, 0x30004};
    Insert(table, context, handles);
    REQUIRE(Find(table, context, handles) != nullptr);

    CHECK(Find(table, context, {0x10002, 0x30005}) == nullptr);
    CHECK(Find(table, context, {0x10002}) == nullptr);
    CHECK(Find(table, Context(table.NewOwner()), handles) == nullptr);
    for (auto change : {&DrawTemplateContext::tic_address, &DrawTemplateContext::tsc_address,
                        &DrawTemplateContext::memory, &DrawTemplateContext::image_table}) {
        DrawTemplateContext other = context;
        other.*change += 0x100;
        CHECK(Find(table, other, handles) == nullptr);
    }
    for (auto change :
         {&DrawTemplateContext::tic_limit, &DrawTemplateContext::tsc_limit,
          &DrawTemplateContext::via_header_index, &DrawTemplateContext::native_tiling}) {
        DrawTemplateContext other = context;
        other.*change ^= 1;
        CHECK(Find(table, other, handles) == nullptr);
    }
    // A hash collision alone never finds another key's template.
    const std::vector<u32> other_handles{0x10002, 0x30006};
    CHECK(table.Find(context, other_handles, VideoCommon::HashDrawTemplate(context, handles)) ==
          nullptr);
}

TEST_CASE("Draw template: a full set replaces its least recently used template", "[video_core]") {
    Table table{1};
    const DrawTemplateContext context = Context(table.NewOwner());
    for (u32 key = 0; key < Table::WAYS; ++key) {
        Insert(table, context, {key});
    }
    REQUIRE(Find(table, context, {0}) != nullptr);
    Insert(table, context, {Table::WAYS});
    CHECK(Find(table, context, {0}) != nullptr);
    CHECK(Find(table, context, {1}) == nullptr);
    CHECK(Find(table, context, {Table::WAYS}) != nullptr);

    // The same key keeps its slot.
    Template& kept = Insert(table, context, {2});
    CHECK(&Insert(table, context, {2}) == &kept);

    size_t found = 0;
    for (u32 key = 0; key < 64; ++key) {
        Insert(table, context, {100 + key});
    }
    for (u32 key = 0; key < 64; ++key) {
        found += Find(table, context, {100 + key}) != nullptr ? 1 : 0;
    }
    CHECK(found == table.Capacity());
}

TEST_CASE("Draw template: a change the key does not cover rejects the template", "[video_core]") {
    std::array<u8, sizeof(Tegra::Texture::TICEntry)> tic_memory{};
    std::array<u8, sizeof(Tegra::Texture::TSCEntry)> tsc_memory{};
    tic_memory[4] = 0x5a;
    tsc_memory[8] = 0xa5;
    std::array<u32, 4> generations{0, 3, 0, 0};
    const auto generation = [&](VideoCommon::ImageViewId id) { return generations[id.index]; };

    Template kept;
    kept.host_serial = 7;
    kept.scale_epoch = 2;
    VideoCommon::DrawTemplateTexture& texture = kept.textures.emplace_back();
    texture.tic_host = tic_memory.data();
    texture.tsc_host = tsc_memory.data();
    std::memcpy(&texture.tic, tic_memory.data(), sizeof(texture.tic));
    std::memcpy(&texture.tsc, tsc_memory.data(), sizeof(texture.tsc));
    texture.view = VideoCommon::ImageViewId{1};
    texture.view_generation = 3;
    REQUIRE(kept.Check(7, 2, generation) == DrawTemplateCheck::Hit);

    CHECK(kept.Check(8, 2, generation) == DrawTemplateCheck::HostMemory);
    CHECK(kept.Check(7, 3, generation) == DrawTemplateCheck::Scale);

    // The view was deleted and its slot holds another one.
    ++generations[1];
    CHECK(kept.Check(7, 2, generation) == DrawTemplateCheck::ViewDeleted);
    --generations[1];

    ++tic_memory[4];
    CHECK(kept.Check(7, 2, generation) == DrawTemplateCheck::Descriptors);
    --tic_memory[4];
    ++tsc_memory[31];
    CHECK(kept.Check(7, 2, generation) == DrawTemplateCheck::Descriptors);
    --tsc_memory[31];
    CHECK(kept.Check(7, 2, generation) == DrawTemplateCheck::Hit);
}

TEST_CASE("Draw template: a rejected template is gone for the next lookup", "[video_core]") {
    Table table{4};
    const DrawTemplateContext context = Context(table.NewOwner());
    const std::vector<u32> handles{0x10002};
    Template* const kept = &Insert(table, context, handles);
    REQUIRE(Find(table, context, handles) == kept);
    Table::Erase(*kept);
    CHECK(Find(table, context, handles) == nullptr);
    CHECK(&Insert(table, context, handles) == kept);
    CHECK(Find(table, context, handles) == kept);
}
