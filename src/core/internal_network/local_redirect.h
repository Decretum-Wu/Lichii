// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/socket_types.h"

/// Custom servers for a title's online services, set up through the environment (by a
/// launcher, a script or the user). They apply only while "Redirect online services to custom
/// servers" (Settings::values.online_service_redirect, Configure > Network) is ticked; all are
/// off when unset:
///   EDEN_NX2_ONLINE_HOSTS=a.example,b.example  these names and their subdomains resolve to
///                                              127.0.0.1 (and are no longer blocked)
///   EDEN_NX2_ONLINE_SERVER=<ip>:<port>         TCP connections to 127.0.0.1:443 go there
///   EDEN_NX2_LOOPBACK_ONLY=1                   other names do not resolve, and the guest's
///                                              connections and datagrams reach loopback only
///                                              (and EDEN_NX2_ONLINE_SERVER, through 127.0.0.1)
///   EDEN_NX2_ONLINE_CA=<file.der>              a CA certificate (DER) the guest's TLS trusts
///                                              too, through the system store and OpenSSL
namespace Network::LocalRedirect {

/// Whether `host` is one of EDEN_NX2_ONLINE_HOSTS or a subdomain of one (case and trailing dots
/// ignored).
[[nodiscard]] bool Matches(std::string_view host);

/// 127.0.0.1 for a redirected host, with the service's port (a number, "http" or "https").
[[nodiscard]] std::optional<std::vector<AddrInfo>> Resolve(
    std::string_view host, const std::optional<std::string>& service);

/// Where a TCP connection to `address` goes: EDEN_NX2_ONLINE_SERVER for 127.0.0.1:443.
[[nodiscard]] SockAddrIn ConnectTarget(const SockAddrIn& address);

/// Whether the guest may reach `address` (always, unless EDEN_NX2_LOOPBACK_ONLY is set). Checked
/// before ConnectTarget, so EDEN_NX2_ONLINE_SERVER stays reachable.
[[nodiscard]] bool Allowed(const SockAddrIn& address);

/// EDEN_NX2_LOOPBACK_ONLY.
[[nodiscard]] bool LoopbackOnly();

/// The EDEN_NX2_ONLINE_CA certificate (DER), empty when unset or unreadable.
[[nodiscard]] std::span<const u8> TestCa();

} // namespace Network::LocalRedirect
