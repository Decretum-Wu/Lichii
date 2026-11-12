// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "core/internal_network/local_redirect.h"

namespace LocalRedirect = Network::LocalRedirect;

namespace {

constexpr Network::SockAddrIn LoopbackHttps{Network::Domain::INET, {127, 0, 0, 1}, 443};
constexpr Network::SockAddrIn Public{Network::Domain::INET, {192, 0, 2, 7}, 443};

bool Same(const Network::SockAddrIn& a, const Network::SockAddrIn& b) {
    return a.family == b.family && a.ip == b.ip && a.portno == b.portno;
}

void ResetRedirectSettings() {
    Settings::values.online_service_redirect.SetValue(false);
    Settings::values.online_redirect_hosts.SetValue({});
    Settings::values.online_redirect_server.SetValue({});
    Settings::values.online_redirect_ca.SetValue({});
    Settings::values.online_redirect_loopback_only.SetValue(false);
    LocalRedirect::Configure();
}

} // Anonymous namespace

TEST_CASE("LocalRedirect: off by default changes nothing", "[core]") {
    REQUIRE_FALSE(Settings::values.online_service_redirect.GetDefault());
    REQUIRE_FALSE(Settings::values.online_redirect_loopback_only.GetDefault());

    ResetRedirectSettings();
    Settings::values.online_redirect_hosts = "a.example";
    Settings::values.online_redirect_server = "192.0.2.1:8443";
    Settings::values.online_redirect_loopback_only = true;
    LocalRedirect::Configure();

    REQUIRE_FALSE(LocalRedirect::Matches("a.example"));
    REQUIRE_FALSE(LocalRedirect::Resolve("a.example", "https").has_value());
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), LoopbackHttps));
    REQUIRE(LocalRedirect::Allowed(Public));
    REQUIRE_FALSE(LocalRedirect::LoopbackOnly());
    REQUIRE(LocalRedirect::TestCa().empty());

    ResetRedirectSettings();
}

TEST_CASE("LocalRedirect: host names map to the user's server", "[core]") {
    ResetRedirectSettings();
    Settings::values.online_service_redirect = true;
    Settings::values.online_redirect_hosts = " A.Example. ,, b.example";
    Settings::values.online_redirect_server = "192.0.2.1:8443";
    LocalRedirect::Configure();

    REQUIRE(LocalRedirect::Matches("a.example"));
    REQUIRE(LocalRedirect::Matches("API.a.example."));
    REQUIRE(LocalRedirect::Matches("b.example"));
    REQUIRE_FALSE(LocalRedirect::Matches("nota.example"));
    REQUIRE_FALSE(LocalRedirect::Matches("a.example.org"));
    REQUIRE_FALSE(LocalRedirect::Matches("example"));

    const auto resolved = LocalRedirect::Resolve("api.a.example", "https");
    REQUIRE(resolved.has_value());
    REQUIRE(resolved->size() == 1);
    REQUIRE(Same(resolved->front().addr, LoopbackHttps));
    REQUIRE(LocalRedirect::Resolve("a.example", "8080")->front().addr.portno == 8080);
    REQUIRE_FALSE(LocalRedirect::Resolve("c.example", "https").has_value());

    const Network::SockAddrIn server{Network::Domain::INET, {192, 0, 2, 1}, 8443};
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), server));
    const Network::SockAddrIn loopback_http{Network::Domain::INET, {127, 0, 0, 1}, 80};
    REQUIRE(Same(LocalRedirect::ConnectTarget(loopback_http), loopback_http));
    REQUIRE(Same(LocalRedirect::ConnectTarget(Public), Public));
    REQUIRE(LocalRedirect::Allowed(Public));

    Settings::values.online_redirect_loopback_only = true;
    LocalRedirect::Configure();
    REQUIRE_FALSE(LocalRedirect::Allowed(Public));
    REQUIRE(LocalRedirect::Allowed(LoopbackHttps));

    ResetRedirectSettings();
    REQUIRE_FALSE(LocalRedirect::Matches("a.example"));
}

TEST_CASE("LocalRedirect: malformed server and CA are ignored", "[core]") {
    ResetRedirectSettings();
    Settings::values.online_service_redirect = true;
    Settings::values.online_redirect_hosts = "a.example";
    Settings::values.online_redirect_server = "192.0.2.1";
    const auto ca_path = std::filesystem::temp_directory_path() / "lichii_redirect_test_ca.der";
    {
        std::ofstream file{ca_path, std::ios::binary};
        file << "-----BEGIN CERTIFICATE-----";
    }
    Settings::values.online_redirect_ca = ca_path.string();
    LocalRedirect::Configure();
    REQUIRE(Same(LocalRedirect::ConnectTarget(LoopbackHttps), LoopbackHttps));
    REQUIRE(LocalRedirect::TestCa().empty());

    {
        std::ofstream file{ca_path, std::ios::binary | std::ios::trunc};
        file.write("\x30\x03\x02\x01\x00", 5);
    }
    LocalRedirect::Configure();
    REQUIRE(LocalRedirect::TestCa().size() == 5);

    std::filesystem::remove(ca_path);
    ResetRedirectSettings();
}
