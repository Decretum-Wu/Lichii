// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/arm/dynarmic/address_tag.h"

TEST_CASE("Guest addresses lose their top byte tag", "[core][address_tag]") {
    // Shine Post: Be Your Idol! (#2647): reads at 0x2000001143fb588c and 0x20000011465f9b30
    // are the heap addresses 0x1143fb588c and 0x11465f9b30 under a tag in bits 63:56.
    CHECK(Core::StripAddressTag(0x2000001143fb588cULL) == 0x1143fb588cULL);
    CHECK(Core::StripAddressTag(0x20000011465f9b30ULL) == 0x11465f9b30ULL);
    CHECK(Core::StripAddressTag(0xff00000000001000ULL) == 0x1000ULL);
    CHECK(Core::StripAddressTag(0x11525b98b0ULL) == 0x11525b98b0ULL);
    CHECK(Core::StripAddressTag(0x0000000000000000ULL) == 0);
    // Bit 55 set selects the upper address range: such an address is not a tagged user address.
    CHECK(Core::StripAddressTag(0xff80000000001000ULL) == 0xff80000000001000ULL);
}
