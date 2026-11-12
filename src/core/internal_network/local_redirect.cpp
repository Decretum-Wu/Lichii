// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>

#include "common/logging.h"
#include "common/settings.h"
#include "core/internal_network/local_redirect.h"

namespace Network::LocalRedirect {
namespace {

std::string Normalize(std::string_view name) {
    while (!name.empty() &&
           (name.back() == '.' || std::isspace(static_cast<unsigned char>(name.back())))) {
        name.remove_suffix(1);
    }
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) {
        name.remove_prefix(1);
    }
    std::string result{name};
    std::ranges::transform(result, result.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

std::optional<u16> ParsePort(std::string_view text) {
    unsigned value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0 || value > 65535) {
        return std::nullopt;
    }
    return static_cast<u16>(value);
}

std::optional<SockAddrIn> ParseServer(std::string_view text) {
    IPv4Address ip{};
    for (size_t i = 0; i < ip.size(); ++i) {
        const char separator = i + 1 < ip.size() ? '.' : ':';
        const size_t end = text.find(separator);
        if (end == std::string_view::npos) {
            return std::nullopt;
        }
        unsigned value{};
        const auto [stop, error] = std::from_chars(text.data(), text.data() + end, value);
        if (error != std::errc{} || stop != text.data() + end || value > 255) {
            return std::nullopt;
        }
        ip[i] = static_cast<u8>(value);
        text.remove_prefix(end + 1);
    }
    const std::optional<u16> port = ParsePort(text);
    if (!port) {
        return std::nullopt;
    }
    return SockAddrIn{Domain::INET, ip, *port};
}

struct Config {
    std::vector<std::string> hosts;
    std::optional<SockAddrIn> server;
    bool loopback_only{};
    std::vector<u8> test_ca;

    struct Empty {};
    explicit Config(Empty) {}

    [[nodiscard]] bool Any() const {
        return !hosts.empty() || server || loopback_only || !test_ca.empty();
    }

    Config() {
        if (const char* list = std::getenv("EDEN_NX2_ONLINE_HOSTS")) {
            std::string_view rest{list};
            while (!rest.empty()) {
                const size_t comma = rest.find(',');
                std::string host = Normalize(rest.substr(0, comma));
                if (!host.empty()) {
                    hosts.push_back(std::move(host));
                }
                rest =
                    comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
            }
        }
        if (const char* text = std::getenv("EDEN_NX2_ONLINE_SERVER"); text && *text) {
            server = ParseServer(text);
            if (!server) {
                LOG_WARNING(Network, "EDEN_NX2_ONLINE_SERVER={} is not <ip>:<port>; ignored", text);
            }
        }
        const char* loopback = std::getenv("EDEN_NX2_LOOPBACK_ONLY");
        loopback_only = loopback != nullptr && loopback[0] != '\0' && loopback[0] != '0';
        if (const char* path = std::getenv("EDEN_NX2_ONLINE_CA"); path && *path) {
            std::ifstream file{path, std::ios::binary};
            test_ca.assign(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
            // A DER certificate is an ASN.1 SEQUENCE; a PEM file or a missing one is not.
            if (test_ca.empty() || test_ca.front() != 0x30) {
                LOG_WARNING(Network, "EDEN_NX2_ONLINE_CA={}: not a DER certificate; ignored", path);
                test_ca.clear();
            }
        }
        if (Any()) {
            LOG_INFO(Network,
                     "Local redirect: {} host names, server {}, loopback only {}, test CA {} bytes",
                     hosts.size(), server ? "set" : "unset", loopback_only, test_ca.size());
        }
    }
};

const Config& Configured() {
    static const Config config;
    return config;
}

/// The environment's settings while "Redirect online services to custom servers" is ticked,
/// else none.
const Config& Get() {
    static const Config none{Config::Empty{}};
    const Config& config = Configured();
    if (Settings::values.online_service_redirect.GetValue()) {
        return config;
    }
    static std::once_flag hinted;
    if (config.Any()) {
        std::call_once(hinted, [] {
            LOG_WARNING(Network, "Local redirect: EDEN_NX2_ONLINE_* or EDEN_NX2_LOOPBACK_ONLY is "
                                 "set, but \"Redirect online services to custom servers\" is "
                                 "off in Configure > Network; ignored");
        });
    }
    return none;
}

bool IsLoopback(const SockAddrIn& address) {
    return address.ip[0] == 127;
}

} // Anonymous namespace

bool Matches(std::string_view host) {
    const auto& hosts = Get().hosts;
    if (hosts.empty()) {
        return false;
    }
    const std::string name = Normalize(host);
    return std::ranges::any_of(hosts, [&](const std::string& entry) {
        return name == entry || (name.size() > entry.size() && name.ends_with(entry) &&
                                 name[name.size() - entry.size() - 1] == '.');
    });
}

std::optional<std::vector<AddrInfo>> Resolve(std::string_view host,
                                             const std::optional<std::string>& service) {
    if (!Matches(host)) {
        return std::nullopt;
    }
    u16 port = 0;
    if (service) {
        if (*service == "https") {
            port = 443;
        } else if (*service == "http") {
            port = 80;
        } else {
            port = ParsePort(*service).value_or(0);
        }
    }
    LOG_INFO(Network, "Local redirect: {} resolves to 127.0.0.1", host);
    return std::vector<AddrInfo>{AddrInfo{
        .family = Domain::INET,
        .socket_type = Type::STREAM,
        .protocol = Protocol::TCP,
        .addr = {Domain::INET, {127, 0, 0, 1}, port},
        .canon_name = std::nullopt,
    }};
}

SockAddrIn ConnectTarget(const SockAddrIn& address) {
    const auto& server = Get().server;
    if (!server || address.ip != IPv4Address{127, 0, 0, 1} || address.portno != 443) {
        return address;
    }
    LOG_DEBUG(Network, "Local redirect: 127.0.0.1:443 -> {}.{}.{}.{}:{}", server->ip[0],
              server->ip[1], server->ip[2], server->ip[3], server->portno);
    return *server;
}

bool Allowed(const SockAddrIn& address) {
    return !Get().loopback_only || IsLoopback(address);
}

bool LoopbackOnly() {
    return Get().loopback_only;
}

std::span<const u8> TestCa() {
    return Get().test_ca;
}

} // namespace Network::LocalRedirect
