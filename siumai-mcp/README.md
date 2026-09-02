# `siumai-mcp`

`siumai-mcp` provides lifecycle-safe Model Context Protocol client and tool bindings for Siumai.
It bounds discovery and messages, keeps remote annotations non-authoritative, and preserves
explicit host approval, concurrency, and recovery policy.

## Streamable HTTP routing

Streamable HTTP MCP uses `HttpTransportRoute::Direct` by default and ignores environment/system
proxy configuration. A host may instead select one explicit trusted CONNECT route. Route
construction validates configuration only and performs no network I/O:

```rust
use siumai_mcp::{HttpTransportRoute, McpClientConfig, ProxyEndpoint};

let route = HttpTransportRoute::trusted_connect(
    ProxyEndpoint::https("https://proxy.example.com")?,
);
let config = McpClientConfig::default().with_http_transport_route(route);

let _ = config;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Pass the configuration to `McpClient::from_http` only when the host is ready to connect. The same
route is applied to POST, GET/SSE, and DELETE/session-close requests. Stdio MCP ignores the route.

The route changes network reachability only. MCP retains its own endpoint policy, origin bearer
authentication, message limits, session lifecycle, and never-replay behavior. It does not accept
provider HTTP settings, provider credentials, provider authentication, or provider retry policy.

## Trust and credentials

Direct mode validates the MCP origin DNS answers and connected peer locally. Trusted CONNECT mode
instead validates the proxy endpoint and peer, then trusts that proxy for destination DNS/peer
selection. The logical MCP URL, inner TLS hostname and certificate, origin authentication, bounds,
and session lifecycle remain authoritative.

An optional `ProxyBasicCredential` is bounded, header-safe, and sent only during CONNECT to the
proxy audience. MCP-origin bearer authentication is sent only inside the tunnel. The proxy
credential is an immutable configuration snapshot; rotate it by rebuilding `McpClientConfig` and
connecting a new `McpClient`.

## Deliberate limits

The implemented claim is bounded CONNECT routing for public HTTPS MCP origins. It is not
certification for a named enterprise proxy product. The crate does not provide environment proxy
discovery, SOCKS, PAC, opaque or bearer/Negotiate/NTLM/Kerberos proxy authentication, URL-userinfo
credentials, raw client/custom-fetch injection, custom proxy CA or mTLS configuration, WebSocket
proxying, stdio proxying, or arbitrary-URL routing. A public proxy must use HTTPS. An explicitly
granted local cleartext proxy must be unauthenticated.
